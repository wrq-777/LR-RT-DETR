"""
by lyuwenyu  (modified: save best by AP50, keep eval/best consistent)
"""
import os
import time
import json
import datetime
from typing import Dict, Any

import torch

from src.misc import dist
from src.data import get_coco_api_from_dataset

from .solver import BaseSolver
from .det_engine import train_one_epoch, evaluate


class DetSolver(BaseSolver):

    def fit(self, ):
        print("Start training")
        self.train()

        args = self.cfg

        # 统计可训练参数量
        n_parameters = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print('number of params:', n_parameters)

        # COCO 基准（评测需要）
        base_ds = get_coco_api_from_dataset(self.val_dataloader.dataset)

        # 日志中维护的最好统计（可选）
        best_stat: Dict[str, Any] = {'epoch': -1}

        # ★ 用 AP50 作为 best 判据（历史最优），断点续训可从已有 best.pth 里恢复
        self.best_ap50 = getattr(self, "best_ap50", -1.0)
        try:
            # 若存在旧 best.pth，读取其中的 ap50 作为基线，避免续训后首次评估就误判
            best_pth = self.output_dir / "best.pth"
            if best_pth.exists():
                _ckpt = torch.load(best_pth, map_location="cpu")
                self.best_ap50 = float(
                    _ckpt.get(
                        "ap50",
                        _ckpt.get("stats", [0, -1])[1]
                        if isinstance(_ckpt.get("stats"), (list, tuple))
                        else -1.0,
                    )
                )
                print(f"[resume-best] restore best_ap50={self.best_ap50:.4f} from best.pth")
        except Exception:
            pass

        start_time = time.time()
        for epoch in range(self.last_epoch + 1, args.epoches):
            if dist.is_dist_available_and_initialized():
                self.train_dataloader.sampler.set_epoch(epoch)

            # 1) 训练一轮
            train_stats = train_one_epoch(
                self.model,
                self.criterion,
                self.train_dataloader,
                self.optimizer,
                self.device,
                epoch,
                args.clip_max_norm,
                print_freq=args.log_step,
                ema=self.ema,
                scaler=self.scaler,
            )

            # 2) 学习率步进
            self.lr_scheduler.step()

            # 3) 按间隔评估（或最后一轮强制评估）
            #    评估口径：若存在 EMA，则默认评 EMA（更稳）；否则评主模型
            eval_interval = getattr(args, "eval_interval", 1)
            do_eval = ((epoch + 1) % eval_interval == 0) or ((epoch + 1) == args.epoches)

            eval_module = self.ema.module if self.ema is not None else self.model

            if do_eval:
                test_stats, coco_evaluator = evaluate(
                    eval_module,
                    self.criterion,
                    self.postprocessor,
                    self.val_dataloader,
                    base_ds,
                    self.device,
                    self.output_dir,
                )
            else:
                test_stats, coco_evaluator = ({}, None)

            # 4) 从评估结果抽取用于“是否刷新最好”的指标
            #    stats[0] = mAP(.5:.95)；stats[1] = AP50
            val_stats: Dict[str, float] = {}
            if coco_evaluator is not None and ("bbox" in coco_evaluator.coco_eval):
                stats = coco_evaluator.coco_eval["bbox"].stats
                if stats is not None:
                    try:
                        val_stats = {"mAP": float(stats[0]), "map50": float(stats[1])}
                    except Exception:
                        val_stats = {}

            # 若没有 coco_evaluator，就尽量从 test_stats 兜底
            if not val_stats and "coco_eval_bbox" in test_stats:
                v = test_stats["coco_eval_bbox"]
                try:
                    if isinstance(v, (list, tuple)) and len(v) > 1:
                        val_stats = {"mAP": float(v[0]), "map50": float(v[1])}
                    elif torch.is_tensor(v) and v.numel() > 1:
                        val_stats = {"mAP": float(v[0].item()), "map50": float(v[1].item())}
                except Exception:
                    pass

            # 5) 只保存“更好”的 best（小文件，仅权重）
            self.maybe_save_best(val_stats, epoch, eval_module)

            # 6) 维护一下原本的 best_stat 日志（可选，保持你原来的风格）
            for k in test_stats.keys():
                try:
                    val = test_stats[k][0]
                except Exception:
                    # 兼容 tensor / float
                    try:
                        val = float(test_stats[k])
                    except Exception:
                        continue

                if k in best_stat:
                    if val > best_stat[k]:
                        best_stat["epoch"] = epoch
                        best_stat[k] = val
                else:
                    best_stat["epoch"] = epoch
                    best_stat[k] = val
            print("best_stat: ", best_stat)

            # 7) 训练/验证日志落盘
            log_stats = {
                **{f"train_{k}": v for k, v in train_stats.items()},
                **{f"test_{k}": v for k, v in test_stats.items()},
                "epoch": epoch,
                "n_parameters": n_parameters,
            }

            if self.output_dir and dist.is_main_process():
                with (self.output_dir / "log.txt").open("a", encoding="utf-8") as f:
                    f.write(json.dumps(log_stats, ensure_ascii=False) + "\n")

                # 降频保存评估原始结果，避免评估大文件每轮都写
                save_eval = getattr(args, "save_eval", True)
                eval_save_interval = getattr(args, "eval_save_interval", 50)
                if save_eval and coco_evaluator is not None and ("bbox" in coco_evaluator.coco_eval):
                    (self.output_dir / "eval").mkdir(exist_ok=True)
                    if ((epoch + 1) % eval_save_interval == 0) or ((epoch + 1) == args.epoches):
                        torch.save(
                            coco_evaluator.coco_eval["bbox"].eval,
                            self.output_dir / "eval" / f"{epoch + 1:03}.pth",
                        )

        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print("Training time {}".format(total_time_str))

    # ★ 以 AP50 作为 best 判据的保存逻辑
    def maybe_save_best(self, val_stats: Dict[str, float], epoch: int, eval_module: torch.nn.Module):
        """
        按 AP50（map50）作为 best 判据：
          - 刷新时保存 best.pth（eval_module 的权重：若评 EMA，则 best.pth 就是 EMA）
          - 如存在 EMA，同时额外保存 best_ema.pth（self.ema.module），两者通常相同
        """
        if not val_stats:
            return

        # 读取 AP50；若取不到则直接返回
        cur_ap50 = val_stats.get("map50", None)
        cur_map = val_stats.get("mAP", None)
        if cur_ap50 is None:
            return

        print(
            f"[eval] epoch={epoch} | mAP={cur_map if cur_map is not None else 'NA'} | "
            f"AP50={cur_ap50:.4f}"
        )

        if cur_ap50 > self.best_ap50:
            self.best_ap50 = cur_ap50

            if self.output_dir and dist.is_main_process():
                os.makedirs(self.output_dir, exist_ok=True)

                # ---- best.pth：保存“当前评测用的模型” ----
                tosave = {
                    "model": eval_module.state_dict(),
                    "epoch": epoch,
                    "ap50": float(cur_ap50),
                }
                if cur_map is not None:
                    tosave["map"] = float(cur_map)
                torch.save(tosave, self.output_dir / "best.pth")
                print(
                    f"[best] new best (by AP50) => AP50={cur_ap50:.4f}"
                    f"{'' if cur_map is None else f' | mAP={cur_map:.4f}'} -> best.pth"
                )

                # ---- 若存在 EMA，单独存一份 best_ema.pth（通常与 best 相同）----
                if self.ema is not None and hasattr(self.ema, "module"):
                    ema_module = self.ema.module
                    tosave_ema = {
                        "model": ema_module.state_dict(),
                        "epoch": epoch,
                        "ap50": float(cur_ap50),
                    }
                    if cur_map is not None:
                        tosave_ema["map"] = float(cur_map)
                    torch.save(tosave_ema, self.output_dir / "best_ema.pth")
                    print(f"[best] (EMA) AP50={cur_ap50:.4f} -> best_ema.pth")

    def val(self, ):
        self.eval()

        base_ds = get_coco_api_from_dataset(self.val_dataloader.dataset)
        module = self.ema.module if self.ema else self.model

        test_stats, coco_evaluator = evaluate(
            module,
            self.criterion,
            self.postprocessor,
            self.val_dataloader,
            base_ds,
            self.device,
            self.output_dir,
        )

        if self.output_dir and dist.is_main_process():
            dist.save_on_master(coco_evaluator.coco_eval["bbox"].eval, self.output_dir / "eval.pth")

        return test_stats, coco_evaluator
