# -*- coding: utf-8 -*-
"""
Eval GFLOPs + FPS (model-only & end-to-end) + COCO AP/AR + Table7-like
适配你的 RT-DETR 工程：
- cfg: src.core.yaml_config.YAMLConfig（注意大小写）
- evaluate: 自动识别是否需要 base_ds / output_dir 等不同函数签名
- ckpt: 支持 --epoch 从目录自动选中指定轮次；支持 --prefer-ema 选择是否优先加载 EMA
- FPS: 自适应 warmup，验证集很小也能给出稳定数值
- GFLOPs: 优先 fvcore，失败回退 thop
- Summary: 额外输出 mAP% / AP50% / AR% 百分比口径
- Table7-like: Params/M, GFLOPs/G, mAP@0.5/%, FPS_e2e, Latency/ms, 真实 P/% 与 R/%
"""
import argparse
import time
import inspect
import os
from pathlib import Path
from glob import glob

import torch
torch.backends.cudnn.benchmark = True


# =========================
# 构建 cfg（你的工程是 YAMLConfig）
# =========================
def build_cfg(cfg_path: str):
    try:
        from src.core.yaml_config import YAMLConfig  # 注意大小写
        # 直接构造
        try:
            return YAMLConfig(cfg_path)
        except Exception:
            pass
        # 类方法兼容
        for name in ("fromfile", "from_file", "load"):
            if hasattr(YAMLConfig, name):
                try:
                    return getattr(YAMLConfig, name)(cfg_path)
                except Exception:
                    pass
        # 关键字构造兼容
        for kw in ("yaml", "file", "path", "cfg"):
            try:
                return YAMLConfig(**{kw: cfg_path})
            except Exception:
                pass
    except Exception:
        pass
    raise RuntimeError("无法实例化配置类：请确认 src.core.yaml_config.YAMLConfig 可用。")


# =========================
# evaluate 自适应调用
# =========================
from src.solver.det_engine import evaluate


def _try_get_base_ds(val_loader):
    """尽力从 val_loader.dataset 提取 COCO 对象。"""
    ds = getattr(val_loader, "dataset", val_loader)

    # 1) 直接已有 coco 句柄
    for attr in ("coco", "coco_api"):
        obj = getattr(ds, attr, None)
        if obj is not None:
            return obj

    # 2) 用 ann_file 构造
    ann = getattr(ds, "ann_file", None) or getattr(ds, "annFile", None)
    if ann and os.path.exists(ann):
        try:
            from pycocotools.coco import COCO
            return COCO(ann)
        except Exception:
            pass

    # 3) 项目工具
    try:
        from src.data.coco.coco_utils import get_coco_api_from_dataset
        return get_coco_api_from_dataset(ds)
    except Exception:
        return None


def call_evaluate(model, criterion, postproc, val_loader, device, workdir: Path):
    """
    兼容多种签名：
      - evaluate(model, criterion, postproc, val_loader, device=..., output_dir=...)
      - evaluate(model, criterion, postproc, val_loader, base_ds, device=..., output_dir=...)
      - evaluate(model, criterion, postproc, val_loader, device, output_dir)
    """
    sig = inspect.signature(evaluate)
    names = list(sig.parameters.keys())
    base_ds = _try_get_base_ds(val_loader)
    outdir = workdir / "eval_out"
    outdir.mkdir(parents=True, exist_ok=True)

    kwargs = {}
    if "device" in names:
        kwargs["device"] = device
    if "output_dir" in names:
        kwargs["output_dir"] = str(outdir)

    # 情形 1：需要 base_ds
    if "base_ds" in names:
        if base_ds is None:
            raise RuntimeError("evaluate() 需要 base_ds，但无法从 val_dataloader.dataset 获取 COCO 对象。")
        return evaluate(model, criterion, postproc, val_loader, base_ds, **kwargs)

    # 情形 2：位置参数里含 output_dir（常见顺序：..., device, output_dir）
    try:
        if len(names) >= 6 and names[5] == "output_dir" and "output_dir" not in kwargs:
            return evaluate(model, criterion, postproc, val_loader, device, str(outdir))
    except Exception:
        pass

    # 情形 3：最简单的关键字/位置参数形式
    return evaluate(model, criterion, postproc, val_loader, **kwargs)


# =========================
# 通用小工具
# =========================
def sync_cuda():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def to_device(batch, device):
    """兼容 detection_collate: (images, targets) 或 dict"""
    if isinstance(batch, (list, tuple)):
        if len(batch) == 2:
            images, targets = batch
            if isinstance(images, (list, tuple)):
                images = [img.to(device, non_blocking=True) for img in images]
            else:
                images = images.to(device, non_blocking=True)
            return images, targets
        else:
            # 非标准返回：尝试把张量都搬到 device
            return [b.to(device, non_blocking=True) if torch.is_tensor(b) else b for b in batch], None
    elif isinstance(batch, dict):
        x = batch.get('images', batch.get('samples', batch))
        if isinstance(x, (list, tuple)):
            x = [t.to(device, non_blocking=True) for t in x]
        else:
            x = x.to(device, non_blocking=True)
        return x, batch.get('targets', None)
    else:
        return batch.to(device, non_blocking=True), None


@torch.no_grad()
def benchmark_fps(cfg, model, val_loader, device, max_iters=200, warmup=30):
    """
    自适应 warmup 的 FPS 统计：
    - warmup 最多占 1/3，总迭代受 max_iters 和验证集大小共同限制
    - 若统计失败，再做一次“无 warmup 的 10 批兜底”
    """
    model.eval()
    postproc = cfg.postprocessor

    # 估计可用批次数
    try:
        total = len(val_loader)
    except Exception:
        total = max_iters
    if not total or total <= 0:
        total = max_iters
    total = min(total, max_iters)
    if total <= 1:
        total = 1
    warmup = min(warmup, max(0, total // 3))
    if warmup >= total:
        warmup = max(0, total - 1)

    n_img, t_model, t_e2e, iters = 0, 0.0, 0.0, 0
    for batch in val_loader:
        iters += 1
        if iters > total:
            break
        t0 = time.time()
        images, targets = to_device(batch, device)

        sync_cuda()
        t1 = time.time()
        _outputs = model(images)   # model-only
        sync_cuda()
        t2 = time.time()

        try:
            _ = postproc(_outputs, targets) if targets is not None else postproc(_outputs, None)
        except Exception:
            pass
        sync_cuda()
        t3 = time.time()

        # 取 batch size
        if isinstance(images, (list, tuple)):
            bs = len(images)
        elif torch.is_tensor(images):
            bs = images.shape[0]
        else:
            bs = 1
        n_img += bs

        if iters > warmup:
            t_model += (t2 - t1)
            t_e2e += (t3 - t0)

    # 兜底（warmup 导致有效样本为 0 时）
    if n_img <= 0 or t_model <= 0 or t_e2e <= 0:
        n_img = t_model = t_e2e = 0.0
        iters = 0
        for batch in val_loader:
            iters += 1
            if iters > min(total, 10):
                break
            t0 = time.time()
            images, targets = to_device(batch, device)
            sync_cuda()
            t1 = time.time()
            _outputs = model(images)
            sync_cuda()
            t2 = time.time()
            try:
                _ = postproc(_outputs, targets) if targets is not None else postproc(_outputs, None)
            except Exception:
                pass
            sync_cuda()
            t3 = time.time()
            bs = len(images) if isinstance(images, (list, tuple)) else (images.shape[0] if torch.is_tensor(images) else 1)
            n_img += bs
            t_model += (t2 - t1)
            t_e2e += (t3 - t0)
        if n_img <= 0 or t_model <= 0 or t_e2e <= 0:
            return None, None, 0

    return n_img / t_model, n_img / t_e2e, int(n_img)


@torch.no_grad()
def estimate_gflops(model, device, h, w):
    """优先 fvcore；失败回退 thop；输入尝试 Tensor 与 list 两种，匹配你项目前向。"""
    model.eval()
    dummy_bchw = torch.zeros(1, 3, h, w, device=device)
    dummy_list = [torch.zeros(3, h, w, device=device)]

    # fvcore
    try:
        from fvcore.nn import FlopCountAnalysis
        try:
            flops = FlopCountAnalysis(model, (dummy_bchw,)).total()
        except Exception:
            flops = FlopCountAnalysis(model, (dummy_list,)).total()
        return flops / 1e9
    except Exception:
        pass

    # thop（1 MAC ≈ 2 FLOPs）
    try:
        from thop import profile
        try:
            macs, _ = profile(model, inputs=(dummy_bchw,), verbose=False)
        except Exception:
            macs, _ = profile(model, inputs=(dummy_list,), verbose=False)
        return (macs * 2.0) / 1e9
    except Exception:
        return None


# =========================
# 从 COCOeval 里抽取真正的 P/R（基于 TP/FP/FN）
# =========================
def compute_pr_from_cocoeval(cocoeval, iou_thr=0.5, score_thr=0.5,
                             area_range_lbl="all", max_dets=100):
    """
    从 pycocotools.cocoeval.COCOeval 抽取“真正的” Precision / Recall：
      - IoU 阈值固定为 iou_thr（默认 0.5）
      - 只看 area=all, maxDets=100（或你指定的）
      - 预测框分数 >= score_thr 才参与统计
    按 TP/FP/FN 统计：
      P = TP / (TP + FP)
      R = TP / (TP + FN)
    """
    if cocoeval is None or not hasattr(cocoeval, "evalImgs"):
        return 0.0, 0.0

    p = cocoeval
    params = p.params

    # IoU index
    iou_thrs = list(params.iouThrs)
    if len(iou_thrs) == 0:
        return 0.0, 0.0
    try:
        iou_idx = iou_thrs.index(iou_thr)
    except ValueError:
        # 没有精确 0.5，就找最接近的那个
        iou_idx = min(range(len(iou_thrs)), key=lambda i: abs(iou_thrs[i] - iou_thr))

    # area / maxDet index
    area_labels = list(params.areaRngLbl)
    try:
        area_idx = area_labels.index(area_range_lbl)
    except ValueError:
        area_idx = 0

    maxdets_list = list(params.maxDets)
    try:
        maxdets_idx = maxdets_list.index(max_dets)
    except ValueError:
        # 找一个最大的 maxDet
        maxdets_idx = max(range(len(maxdets_list)), key=lambda i: maxdets_list[i])

    TP = FP = FN = 0

    for evalImg in p.evalImgs:
        if evalImg is None:
            continue
        # 过滤 area / maxDet
        if evalImg["aRng"] != params.areaRng[area_idx]:
            continue
        if evalImg["maxDet"] != params.maxDets[maxdets_idx]:
            continue

        dtScores = evalImg["dtScores"]  # [D]
        dtMatches = evalImg["dtMatches"][iou_idx]  # [D]
        dtIgnore = evalImg["dtIgnore"][iou_idx]    # [D]
        gtIgnore = evalImg["gtIgnore"]            # [G]
        gtMatches = evalImg["gtMatches"][iou_idx]  # [G]

        # 预测：score >= score_thr & 非 ignore
        for s, m, ign in zip(dtScores, dtMatches, dtIgnore):
            if s < score_thr:
                continue
            if ign:
                continue
            if m > 0:   # 匹配到某个 gt（同一 iou_thr）
                TP += 1
            else:
                FP += 1

        # GT：非 ignore，未被匹配的算 FN
        for m, ign in zip(gtMatches, gtIgnore):
            if ign:
                continue
            if m == 0:
                FN += 1

    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    return precision, recall


# =========================
# ckpt 选择与加载
# =========================
def find_ckpt_by_epoch(ckpt_dir: str, epoch: int):
    """
    在 ckpt_dir 下按常见命名查找第 epoch 轮的 ckpt。
    支持的模式举例：
      epoch_264.pth, epoch-264.pth, ckpt_epoch_264.pth, model_epoch_264.pth,
      epoch=264*.pth, *-264.pth, *_264.pth
    """
    if not ckpt_dir or not os.path.isdir(ckpt_dir):
        return None
    ep = str(epoch)
    patterns = [
        f"epoch_{ep}.pth",
        f"epoch-{ep}.pth",
        f"ckpt_epoch_{ep}.pth",
        f"model_epoch_{ep}.pth",
        f"*epoch={ep}*.pth",
        f"*_{ep}.pth",
        f"*-{ep}.pth",
    ]
    for pat in patterns:
        paths = sorted(glob(os.path.join(ckpt_dir, pat)))
        if paths:
            return paths[-1]
    return None


def load_ckpt_with_policy(model, ckpt_path: str, prefer_ema: bool = True):
    import collections
    ckpt = torch.load(ckpt_path, map_location="cpu")

    # 1) 常见键优先
    common_ema = ["model_ema", "state_dict_ema", "ema", "ema_state_dict"]
    common_raw = ["model", "state_dict", "weights", "params", "net", "state", "model_state", "module"]
    cand_keys = (common_ema + common_raw) if prefer_ema else (common_raw + common_ema)

    def try_load(state):
        # 去掉常见前缀
        def strip_prefix(sd, prefix="module."):
            if not any(k.startswith(prefix) for k in sd.keys()):
                return sd
            return {k[len(prefix):] if k.startswith(prefix) else k: v for k, v in sd.items()}
        if isinstance(state, dict) and len(state) > 0 and all(isinstance(k, str) for k in state.keys()):
            sd = strip_prefix(state, "module.")
            missing, unexpected = model.load_state_dict(sd, strict=False)
            print(f"[ckpt] loaded: missing={len(missing)} unexpected={len(unexpected)}")
            return True
        return False

    if isinstance(ckpt, dict):
        # 1.1 直接命中
        for k in cand_keys:
            if k in ckpt and isinstance(ckpt[k], dict) and len(ckpt[k]) > 0:
                print(f"[ckpt] using key='{k}' ({len(ckpt[k])} params)")
                if try_load(ckpt[k]):
                    return
        # 1.2 递归搜最大“像 state_dict”的子字典
        best = None

        def score(d):
            if not isinstance(d, dict):
                return -1
            ks = list(d.keys())
            if not ks or not all(isinstance(x, str) for x in ks):
                return -1
            n = len(ks)
            hit = sum(1 for x in ks if x.endswith(".weight") or x.endswith(".bias"))
            return n + 5 * hit  # 权重名多的得分更高

        stack = [ckpt]
        seen = set()
        while stack:
            cur = stack.pop()
            if id(cur) in seen:
                continue
            seen.add(id(cur))
            if isinstance(cur, dict):
                sc = score(cur)
                if sc >= 200:  # 粗阈值：小字典直接忽略
                    if best is None or sc > best[0]:
                        best = (sc, cur)
                for v in cur.values():
                    if isinstance(v, (dict, collections.OrderedDict)):
                        stack.append(v)
        if best is not None:
            sc, cand = best
            print(f"[ckpt] using best nested dict (score={sc}, size={len(cand)})")
            if try_load(cand):
                return

    # 2) 兜底：原逻辑（不建议，但保留）
    print(f"[ckpt] using raw dict as state ({len(ckpt) if isinstance(ckpt, dict) else 'N/A'} items)")
    if isinstance(ckpt, dict):
        missing, unexpected = model.load_state_dict(ckpt, strict=False)
        print(f"[ckpt] loaded: missing={len(missing)} unexpected={len(unexpected)}")
    else:
        raise RuntimeError("Unrecognized checkpoint format")


# =========================
# Main
# =========================
def main():
    ap = argparse.ArgumentParser("Eval GFLOPs + FPS + COCO AP/AR + Table7-like")
    ap.add_argument("--config", required=True, help="YAML（val_dataloader 指向验证集）")
    ap.add_argument("--ckpt", default="", help="权重路径（可选）")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--max-iters", type=int, default=200)
    ap.add_argument("--warmup", type=int, default=30)
    ap.add_argument("--gflops-size", type=str, default="640,640", help="GFLOPs dummy 输入尺寸 H,W")

    # 按 epoch 选择权重、ckpt 文件夹、是否优先 EMA
    ap.add_argument("--epoch", type=int, default=None, help="按轮次选择权重，如 264")
    ap.add_argument("--ckpt-dir", default="", help="当使用 --epoch 时，从该目录下自动匹配 ckpt 文件")
    ap.add_argument("--prefer-ema", type=int, default=1, help="1=优先加载EMA, 0=不优先")

    # 新增：P/R 计算时使用的分数阈值
    ap.add_argument("--pr-score-thr", type=float, default=0.5,
                    help="计算 Precision/Recall 时的 score 阈值（默认 0.5）")

    args = ap.parse_args()

    device = torch.device(args.device)
    try:
        H, W = [int(x) for x in args.gflops_size.replace(' ', '').split(',')]
    except Exception:
        H, W = 640, 640

    # ---- cfg / model / loader ----
    cfg = build_cfg(args.config)
    model = cfg.model.to(device)

    # 统计参数量（只算 requires_grad 的）
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_params_m = n_parameters / 1e6

    # ---- 选择 ckpt 路径（优先 epoch 指定）----
    ckpt_path = args.ckpt
    if args.epoch is not None:
        search_dir = args.ckpt_dir
        if not search_dir:
            if ckpt_path and os.path.isfile(ckpt_path):
                search_dir = os.path.dirname(ckpt_path)
            else:
                # 兜底猜测一次常见目录，按需调整
                guess = os.path.join("output", "neu_det_ghostv2", "ckpts")
                search_dir = guess if os.path.isdir(guess) else "."
        cand = find_ckpt_by_epoch(search_dir, args.epoch)
        if not cand:
            raise FileNotFoundError(f"在目录 {search_dir} 未找到第 {args.epoch} 轮的 ckpt 文件")
        ckpt_path = cand
        print(f"[ckpt] epoch={args.epoch} → {ckpt_path}")

    # ---- 加载权重（支持优先 EMA）----
    if ckpt_path:
        load_ckpt_with_policy(model, ckpt_path, prefer_ema=bool(args.prefer_ema))

    val_loader = cfg.val_dataloader
    criterion = getattr(cfg, "criterion", None)
    postproc = getattr(cfg, "postprocessor", None)

    workdir = Path("./eval_tmp")
    workdir.mkdir(parents=True, exist_ok=True)

    # === 1) COCO AP/AR ===
    print("\n[Eval] COCO AP/AR：")
    with torch.autocast(device_type='cuda', enabled=(device.type == 'cuda')):
        test_stats, coco_evaluator = call_evaluate(model, criterion, postproc, val_loader, device, workdir)
    stats = coco_evaluator.coco_eval['bbox'].stats
    ap, ap50, ap75 = float(stats[0]), float(stats[1]), float(stats[2])
    ar1, ar10, ar100 = float(stats[6]), float(stats[7]), float(stats[8])
    print(f"[COCO] AP=.5:.95={ap:.3f} | AP50={ap50:.3f} | AP75={ap75:.3f}")
    print(f"[COCO] AR@1={ar1:.3f} | AR@10={ar10:.3f} | AR@100={ar100:.3f}")

    # === 2) FPS ===
    print("\n[Bench] 测 FPS（自适应 warmup，最多 {} 批）：".format(args.max_iters))
    fps_model, fps_e2e, n_img = benchmark_fps(cfg, model, val_loader, device, args.max_iters, args.warmup)
    if fps_model is None:
        print("[FPS] 无法统计（请检查 dataloader 批数据格式或减小 --warmup / 增大 --max-iters）")
        fps_model = fps_e2e = 0.0
    else:
        print(f"[FPS] model-only = {fps_model:.2f} img/s | end-to-end = {fps_e2e:.2f} img/s | images used = {n_img}")

    # === 3) GFLOPs ===
    print(f"\n[GFLOPs] 估算（dummy 输入 {H}x{W}）...")
    gflops = estimate_gflops(model, device, H, W)
    if gflops is None:
        print("[GFLOPs] 失败：fvcore/thop 皆未成功（可能未安装或 forward 需包装）")
        gflops = 0.0
    else:
        print(f"[GFLOPs] ≈ {gflops:.2f} G")

    # === 4) 真实 Precision / Recall（IoU=0.5, score>=pr_score_thr）===
    pr, rr = compute_pr_from_cocoeval(
        coco_evaluator.coco_eval['bbox'],
        iou_thr=0.5,
        score_thr=float(args.pr_score_thr),
        area_range_lbl="all",
        max_dets=100
    )
    print("\n[PR@IoU=0.5] (score >= {:.2f}) Precision={:.3f}, Recall={:.3f}"
          .format(args.pr_score_thr, pr, rr))

    # --- Summary ---
    print("\n[Summary] AP={:.3f}, AP50={:.3f}, AP75={:.3f}, AR@1={:.3f}, AR@10={:.3f}, AR@100={:.3f}, "
          "FPS_model={:.2f}, FPS_e2e={:.2f}, GFLOPs≈{:.2f}"
          .format(ap, ap50, ap75, ar1, ar10, ar100, fps_model, fps_e2e, gflops))
    print("[Summary %] mAP(.5:.95)={:.2f}% | AP50={:.2f}% | AR@100={:.2f}%"
          .format(100.0 * ap, 100.0 * ap50, 100.0 * ar100))
    print("[Hint] GFLOPs 基于 dummy 输入，务必用同一尺寸对比（默认 640x640，可用 --gflops-size 调整）。")

    # --- Table7-like ---
    latency_ms = 1000.0 / fps_e2e if fps_e2e > 0 else 0.0
    print("\n[Table7-like] 关键结果（模仿《钢铁缺陷检测》Table 7）：")
    print("  Params/M | GFLOPs/G | mAP@0.5/% | FPS_e2e/img/s | Latency/ms |   P/%   |   R/%")
    print(" {:8.2f} | {:8.2f} | {:10.2f} | {:14.2f} | {:10.2f} | {:7.2f} | {:7.2f}".format(
        n_params_m,
        gflops,
        100.0 * ap50,
        fps_e2e,
        latency_ms,
        100.0 * pr,
        100.0 * rr
    ))


if __name__ == "__main__":
    main()
