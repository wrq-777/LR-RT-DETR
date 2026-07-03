"""by lyuwenyu
修改说明：
- 新增：Teacher Distillation 支持（R50 → GhostV2）
  * 从 YAML 读取 KD 段（use_kd/teacher_cfg/teacher_ckpt/...）
  * 构建并冻结 Teacher，优先加载 model_ema → model → state_dict
  * 将 teacher / kd_cfg 注入 cfg，供 det_engine 或 criterion 使用
- 新增：仅保存最优权重（best.pth），可选保存 best_ema.pth
- 新增：覆盖写 last.pth（完整断点）以便需要时继续训练，但不攒文件
- 修复：state_dict(last_epoch) 的调用一致性；save() 默认保存完整断点
"""

import torch
import torch.nn as nn

from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Any

from src.misc import dist
from src.core import BaseConfig

# ★ 新增：YAMLConfig 用于按老师的结构文件构建 teacher 模型
try:
    from src.core.yaml_config import YAMLConfig
except Exception:
    YAMLConfig = None


class BaseSolver(object):
    def __init__(self, cfg: BaseConfig) -> None:
        self.cfg = cfg

        # 仅保存最优相关
        self.best_metric: float = float("-inf")
        self.best_path: Optional[Path] = None           # 在 setup() 中初始化
        self.best_ema_path: Optional[Path] = None       # 在 setup() 中初始化

        # 可选项（从 cfg 读取，不存在则给默认值）
        self.save_fp16: bool = bool(getattr(cfg, "save_fp16", False))                 # 仅权重保存时是否转 FP16
        self.keep_last_full: bool = bool(getattr(cfg, "keep_last_full", True))        # 是否覆盖写 last.pth（完整断点）
        self.best_metric_key_order = tuple(getattr(cfg, "best_metric_keys", (
            "map50", "mAP50", "mAP", "map", "box_ap"
        )))  # 依次尝试这些键从 val_stats 里取指标

        # ★ 新增：KD / Teacher 相关占位
        self.teacher: Optional[nn.Module] = None
        self.kd_cfg: Optional[dict] = None

    # -------------------- 基础流程 --------------------
    def setup(self, ):
        """Avoid instantiating unnecessary classes
        """
        cfg = self.cfg
        device = cfg.device
        self.device = device
        self.last_epoch = cfg.last_epoch

        self.model = dist.warp_model(cfg.model.to(device), cfg.find_unused_parameters, cfg.sync_bn)
        self.criterion = cfg.criterion.to(device)
        self.postprocessor = cfg.postprocessor

        # NOTE (lvwenyu): should load_tuning_state before ema instance building
        if self.cfg.tuning:
            print(f'Tuning checkpoint from {self.cfg.tuning}')
            self.load_tuning_state(self.cfg.tuning)

        self.scaler = cfg.scaler
        self.ema = cfg.ema.to(device) if cfg.ema is not None else None

        self.output_dir = Path(cfg.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 初始化 best/last 路径
        self.best_path = self.output_dir / "best.pth"
        self.best_ema_path = self.output_dir / "best_ema.pth"
        self.last_path = self.output_dir / "last.pth"

        # 打印保存策略
        print(f"[ckpt] best will be saved to: {self.best_path}")
        if self.ema is not None:
            print(f"[ckpt] best EMA will be saved to: {self.best_ema_path}")
        if self.keep_last_full:
            print(f"[ckpt] last full checkpoint will be saved to: {self.last_path}")

        # ★★★★★ 新增：Teacher Distillation 构建与注入 ★★★★★
        # 读取 KD 段配置（不存在则 None）
        self.kd_cfg = getattr(cfg, "KD", None)
        if self.kd_cfg and isinstance(self.kd_cfg, dict) and self.kd_cfg.get("use_kd", False):
            self._build_and_attach_teacher(self.kd_cfg)

    def train(self, ):
        self.setup()
        self.optimizer = self.cfg.optimizer
        self.lr_scheduler = self.cfg.lr_scheduler

        # NOTE instantiating order
        if self.cfg.resume:
            print(f'Resume checkpoint from {self.cfg.resume}')
            self.resume(self.cfg.resume)

        self.train_dataloader = dist.warp_loader(self.cfg.train_dataloader,
                                                 shuffle=self.cfg.train_dataloader.shuffle)
        self.val_dataloader = dist.warp_loader(self.cfg.val_dataloader,
                                               shuffle=self.cfg.val_dataloader.shuffle)

    def eval(self, ):
        self.setup()
        self.val_dataloader = dist.warp_loader(self.cfg.val_dataloader,
                                               shuffle=self.cfg.val_dataloader.shuffle)

        if self.cfg.resume:
            print(f'resume from {self.cfg.resume}')
            self.resume(self.cfg.resume)

    # -------------------- Teacher / KD 工具 --------------------
    def _build_and_attach_teacher(self, kd_cfg: dict):
        """按 kd_cfg 构建并冻结 Teacher；注入到 cfg 以供训练循环/损失使用。"""
        if YAMLConfig is None:
            print("[KD] warn: YAMLConfig not available, skip teacher building.")
            return
        teacher_cfg_path = kd_cfg.get("teacher_cfg", "")
        teacher_ckpt_path = kd_cfg.get("teacher_ckpt", "")
        if not teacher_cfg_path or not teacher_ckpt_path:
            print("[KD] warn: missing 'teacher_cfg' or 'teacher_ckpt', skip teacher building.")
            return

        print(f"[KD] building teacher from cfg: {teacher_cfg_path}")
        try:
            t_cfg = YAMLConfig(teacher_cfg_path)
            teacher = t_cfg.model  # 与学生相同工厂构建
        except Exception as e:
            print(f"[KD] ERROR: failed to build teacher model from {teacher_cfg_path}: {e}")
            return

        # 加载 teacher 权重（优先尝试 EMA）
        try:
            ckpt = torch.load(teacher_ckpt_path, map_location="cpu")
        except Exception as e:
            print(f"[KD] ERROR: failed to load teacher ckpt {teacher_ckpt_path}: {e}")
            return

        state = None
        if isinstance(ckpt, dict):
            for k in ("model_ema", "ema", "ema_state_dict", "state_dict_ema", "model", "state_dict"):
                if k in ckpt and isinstance(ckpt[k], dict):
                    state = ckpt[k]
                    print(f"[KD] load teacher using key='{k}'")
                    break
        if state is None:
            # 兜底：如果整个 ckpt 本身就是 state_dict
            if isinstance(ckpt, dict):
                state = ckpt
                print("[KD] load teacher using raw dict (fallback)")
            else:
                print("[KD] ERROR: bad teacher checkpoint format.")
                return

        try:
            missing, unexpected = teacher.load_state_dict(state, strict=False)
            print(f"[KD] teacher loaded: missing={len(missing)} unexpected={len(unexpected)}")
            if len(missing) < 10:
                for m in missing:
                    print("  - missing:", m)
            if len(unexpected) < 10:
                for u in unexpected:
                    print("  - unexpected:", u)
        except Exception as e:
            print(f"[KD] ERROR: teacher load_state_dict failed: {e}")
            return

        teacher.eval()
        for p in teacher.parameters():
            p.requires_grad_(False)

        self.teacher = teacher.to(self.device)
        # 对下游开放（训练循环 / criterion 可通过 cfg.teacher / cfg.kd_cfg 访问）
        try:
            setattr(self.cfg, "teacher", self.teacher)
            setattr(self.cfg, "kd_cfg", self.kd_cfg)
            print("[KD] teacher attached to cfg as 'cfg.teacher'; kd config as 'cfg.kd_cfg'.")
            # 友好提示
            print("[KD] NOTE: 要生效的前提是训练循环/损失中实际使用 cfg.teacher/cfg.kd_cfg 计算 KD loss。")
        except Exception:
            pass

    # -------------------- 状态编解码 --------------------
    def state_dict(self, last_epoch: Optional[int] = None) -> Dict[str, Any]:
        """完整断点：模型 + optimizer + lr_scheduler + ema + scaler + last_epoch"""
        state: Dict[str, Any] = {}
        state['model'] = dist.de_parallel(self.model).state_dict()
        state['date'] = datetime.now().isoformat()
        state['last_epoch'] = int(self.last_epoch if last_epoch is None else last_epoch)

        if getattr(self, 'optimizer', None) is not None:
            state['optimizer'] = self.optimizer.state_dict()

        if getattr(self, 'lr_scheduler', None) is not None:
            state['lr_scheduler'] = self.lr_scheduler.state_dict()

        if self.ema is not None:
            state['ema'] = self.ema.state_dict()

        if getattr(self, 'scaler', None) is not None:
            state['scaler'] = self.scaler.state_dict()

        return state

    def load_state_dict(self, state: Dict[str, Any]):
        """load state dict"""
        if 'last_epoch' in state:
            self.last_epoch = state['last_epoch']
            print('Loading last_epoch')

        if getattr(self, 'model', None) is not None and 'model' in state:
            if dist.is_parallel(self.model):
                self.model.module.load_state_dict(state['model'])
            else:
                self.model.load_state_dict(state['model'])
            print('Loading model.state_dict')

        if getattr(self, 'ema', None) is not None and 'ema' in state:
            self.ema.load_state_dict(state['ema'])
            print('Loading ema.state_dict')

        if getattr(self, 'optimizer', None) is not None and 'optimizer' in state:
            self.optimizer.load_state_dict(state['optimizer'])
            print('Loading optimizer.state_dict')

        if getattr(self, 'lr_scheduler', None) is not None and 'lr_scheduler' in state:
            self.lr_scheduler.load_state_dict(state['lr_scheduler'])
            print('Loading lr_scheduler.state_dict')

        if getattr(self, 'scaler', None) is not None and 'scaler' in state:
            self.scaler.load_state_dict(state['scaler'])
            print('Loading scaler.state_dict')

    # -------------------- 保存/恢复（提供三类保存） --------------------
    def save(self, path: str, epoch: Optional[int] = None):
        """保存完整断点（可能很大）。建议只在需要恢复训练时使用。
        """
        state = self.state_dict(last_epoch=epoch)
        dist.save_on_master(state, path)

    def save_last(self, epoch: int):
        """覆盖式保存完整断点到 last.pth（不会越攒越多）"""
        if not self.keep_last_full:
            return
        state = self.state_dict(last_epoch=epoch)
        dist.save_on_master(state, str(self.last_path))
        print(f"[ckpt] last checkpoint saved @ epoch={epoch}: {self.last_path}")

    def _save_best_weights_only(self, path: Path, use_ema: bool = False):
        """仅保存权重（体积最小），可选 FP16；若 use_ema=True 则保存 EMA 模型权重。"""
        module = None
        if use_ema and self.ema is not None:
            # EMA 的 state_dict 结构通常是 {'module': model_sd, ...}
            ema_state = self.ema.state_dict()
            if isinstance(ema_state, dict) and 'module' in ema_state:
                module_sd = ema_state['module']
            else:
                # 兜底：直接从 ema 的属性取
                module = getattr(self.ema, 'module', None)
        if module is None:
            module = dist.de_parallel(self.model)

        if use_ema and self.ema is not None and module is None:
            # 从 ema_state['module'] 直接保存（已在 CPU 上）
            sd = {}
            for k, v in self.ema.state_dict()['module'].items():
                if self.save_fp16 and isinstance(v, torch.Tensor) and v.is_floating_point():
                    sd[k] = v.half().cpu()
                else:
                    sd[k] = v.cpu()
            dist.save_on_master(sd, str(path))
            return

        # 常规：从 module.state_dict() 提取
        sd = {}
        for k, v in module.state_dict().items():
            t = v.detach().cpu() if isinstance(v, torch.Tensor) else v
            if self.save_fp16 and isinstance(t, torch.Tensor) and t.is_floating_point():
                t = t.half()
            sd[k] = t
        dist.save_on_master(sd, str(path))

    def resume(self, path: str):
        """load resume（完整断点）"""
        state = torch.load(path, map_location='cpu')  # for cuda:0 memory
        self.load_state_dict(state)

    # -------------------- Tuning 预加载 --------------------
    def load_tuning_state(self, path, ):
        """only load model for tuning and skip missed/dismatched keys
        """
        if 'http' in path:
            state = torch.hub.load_state_dict_from_url(path, map_location='cpu')
        else:
            state = torch.load(path, map_location='cpu')

        module = dist.de_parallel(self.model)

        # TODO hard code
        if 'ema' in state:
            stat, infos = self._matched_state(module.state_dict(), state['ema']['module'])
        else:
            stat, infos = self._matched_state(module.state_dict(), state['model'])

        module.load_state_dict(stat, strict=False)
        print(f'Load model.state_dict, {infos}')

    @staticmethod
    def _matched_state(state: Dict[str, torch.Tensor], params: Dict[str, torch.Tensor]):
        missed_list = []
        unmatched_list = []
        matched_state = {}
        for k, v in state.items():
            if k in params:
                if v.shape == params[k].shape:
                    matched_state[k] = params[k]
                else:
                    unmatched_list.append(k)
            else:
                missed_list.append(k)

        return matched_state, {'missed': missed_list, 'unmatched': unmatched_list}

    # -------------------- 仅保存最优的对外接口 --------------------
    def _metric_from_val_stats(self, val_stats: Dict[str, Any]) -> Optional[float]:
        """从验证统计里抽取用于“是否变更最优”的指标。默认尝试 map50/mAP 等键。"""
        if not isinstance(val_stats, dict):
            return None
        for k in self.best_metric_key_order:
            if k in val_stats:
                try:
                    return float(val_stats[k])
                except Exception:
                    continue
        return None

    def maybe_save_best(self, val_stats: Dict[str, Any], epoch: int):
        """在每次验证后调用；当指标提升时：
           - 保存仅权重 best.pth（小文件）
           - 若存在 EMA，同时保存 best_ema.pth
           - 覆盖写 last.pth（完整断点，便于必要时继续训练）
        """
        metric = self._metric_from_val_stats(val_stats)
        if metric is None:
            print("[ckpt] warn: no valid metric found in val_stats, skip best saving.")
            # 即使没有 metric，也可以仍然覆盖写 last.pth，防止断电丢失
            self.save_last(epoch)
            return

        improved = (metric > self.best_metric)
        if improved:
            old = self.best_metric
            self.best_metric = metric
            # 仅权重：best.pth
            self._save_best_weights_only(self.best_path, use_ema=False)
            print(f"[ckpt] new best={metric:.4f} (prev={old:.4f}) -> {self.best_path}")
            # 仅权重：best_ema.pth（如有 EMA）
            if self.ema is not None:
                self._save_best_weights_only(self.best_ema_path, use_ema=True)
                print(f"[ckpt] new best EMA saved -> {self.best_ema_path}")

        # 总是覆盖写 last.pth（完整断点），便于 resume；不会攒文件
        self.save_last(epoch)

    # -------------------- 子类接口 --------------------
    def fit(self, ):
        raise NotImplementedError('')

    def val(self, ):
        raise NotImplementedError('')
