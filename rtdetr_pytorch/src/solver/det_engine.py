"""
Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
https://github.com/facebookresearch/detr/blob/main/engine.py

by lyuwenyu
"""

import math
import os
import sys
import pathlib
from typing import Iterable, Tuple, Optional, Dict, Any

import torch
import torch.amp
import torch.nn.functional as F

from src.data import CocoEvaluator
from src.misc import (MetricLogger, SmoothedValue, reduce_dict)


# -------------------- KD 工具函数 --------------------
def _resolve_kd_handles(kwargs: Dict[str, Any]) -> Tuple[Optional[torch.nn.Module], Optional[dict]]:
    """
    从 kwargs / kwargs['cfg'] 中解析 teacher / kd_cfg（均可选）。
    优先级：kwargs['teacher'] / kwargs['kd_cfg'] ；否则尝试 kwargs['cfg'].teacher / .kd_cfg
    """
    teacher = kwargs.get('teacher', None)
    kd_cfg = kwargs.get('kd_cfg', None)
    cfg = kwargs.get('cfg', None)
    if (teacher is None or kd_cfg is None) and cfg is not None:
        if teacher is None:
            teacher = getattr(cfg, 'teacher', None)
        if kd_cfg is None:
            kd_cfg = getattr(cfg, 'kd_cfg', None)
    return teacher, kd_cfg


def _kd_cls_per_query(student_logits: torch.Tensor, teacher_logits: torch.Tensor, temp: float = 4.0) -> torch.Tensor:
    """
    逐 query 的 KL(student || teacher)，带温度缩放；默认去掉背景列（最后一列）。
    输入:  [B, Q, C]
    输出:  [B, Q]
    """
    s = student_logits / temp
    t = teacher_logits / temp
    # 去背景列（假定最后一列为背景；若你的实现不同，请相应修改）
    s = s[..., :-1]
    t = t[..., :-1]
    logp_s = F.log_softmax(s, dim=-1)
    p_t = F.softmax(t, dim=-1)
    kl = F.kl_div(logp_s, p_t, reduction='none').sum(dim=-1)  # [B,Q]
    return kl * (temp * temp)


def _l1_per_query(s_boxes: torch.Tensor, t_boxes: torch.Tensor) -> torch.Tensor:
    """
    盒子 L1，逐 query：
    输入:  [B,Q,4] × [B,Q,4] -> 输出 [B,Q,1]
    """
    return (s_boxes - t_boxes).abs().mean(dim=-1, keepdim=True)


def _maybe_giou_per_query(s_boxes: torch.Tensor, t_boxes: torch.Tensor) -> Optional[torch.Tensor]:
    """
    可选 GIoU 蒸馏（如果你项目里已有 generalized_box_iou / kd_boxes_giou，可在这里替换）。
    兜底返回 None（只做 L1 蒸馏也足够稳定）。
    """
    return None


# -------------------- 训练与评测 --------------------
def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, max_norm: float = 0, **kwargs):
    model.train()
    criterion.train()
    metric_logger = MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = kwargs.get('print_freq', 10)

    ema = kwargs.get('ema', None)
    scaler = kwargs.get('scaler', None)

    # 解析 KD 句柄（teacher/kd_cfg）
    teacher, kd_cfg = _resolve_kd_handles(kwargs)

    for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        if scaler is not None:
            # 前向（AMP）
            with torch.autocast(device_type=str(device), cache_enabled=True):
                outputs = model(samples, targets)

            # 主损失（禁用 AMP 做更稳定的 CE / IoU 计算）
            with torch.autocast(device_type=str(device), enabled=False):
                loss_dict = criterion(outputs, targets)

                # === KD：仅当配置开启且提供了 teacher ===
                if kd_cfg and isinstance(kd_cfg, dict) and kd_cfg.get("use_kd", False) and (teacher is not None):
                    # warmup + ramp
                    warmup = int(kd_cfg.get("warmup_epochs", 5))
                    ramp = int(kd_cfg.get("ramp_epochs", 10))
                    if epoch >= warmup:
                        ramp_ratio = (epoch - warmup) / max(1, ramp)
                        ramp_ratio = float(max(0.0, min(1.0, ramp_ratio)))
                    else:
                        ramp_ratio = 0.0

                    if ramp_ratio > 0.0:
                        T = float(kd_cfg.get("temp", 4.0))
                        thr = float(kd_cfg.get("score_thr", 0.20))
                        w_cls = float(kd_cfg.get("w_cls", 0.7))
                        w_box_l1 = float(kd_cfg.get("w_box_l1", 1.0))
                        w_box_gio = float(kd_cfg.get("w_box_giou", 0.0))

                        with torch.no_grad():
                            teacher.eval()
                            t_out = teacher(samples)  # 老师只前向（不需要 targets）

                        # 取 Student / Teacher 的最终输出
                        s_logits = outputs["pred_logits"]    # [B,Q,C]
                        s_boxes = outputs["pred_boxes"]       # [B,Q,4]
                        t_logits = t_out["pred_logits"].detach()
                        t_boxes = t_out["pred_boxes"].detach()

                        # 置信度掩码：只蒸馏老师高置信 query
                        with torch.no_grad():
                            t_prob = F.softmax(t_logits, dim=-1)[..., :-1]  # 去背景列
                            t_max = t_prob.max(dim=-1).values               # [B,Q]
                            mask = (t_max > thr).float().unsqueeze(-1)      # [B,Q,1]

                        # 分类 KD（逐 query KL）
                        kd_cls_q = _kd_cls_per_query(s_logits, t_logits, temp=T)  # [B,Q]
                        kd_cls = (kd_cls_q * mask.squeeze(-1)).sum() / (mask.sum() + 1e-6)

                        # 盒子 KD（L1）
                        kd_l1_q = _l1_per_query(s_boxes, t_boxes)  # [B,Q,1]
                        kd_l1 = (kd_l1_q * mask).sum() / (mask.sum() + 1e-6)

                        # 可选 GIoU KD
                        kd_giou = _maybe_giou_per_query(s_boxes, t_boxes)
                        if kd_giou is not None and w_box_gio > 0:
                            kd_giou = (kd_giou * mask).sum() / (mask.sum() + 1e-6)
                            loss_kd = (w_cls * kd_cls + w_box_l1 * kd_l1 + w_box_gio * kd_giou) * ramp_ratio
                        else:
                            kd_giou = None
                            loss_kd = (w_cls * kd_cls + w_box_l1 * kd_l1) * ramp_ratio

                        # 并入损失字典与总损
                        loss_dict["loss_kd"] = loss_kd

            # 反传（含 KD）
            loss = sum(loss_dict.values())
            scaler.scale(loss).backward()

            if max_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        else:
            # FP32 路径
            outputs = model(samples, targets)
            loss_dict = criterion(outputs, targets)

            # === KD：FP32 路径同样合入 ===
            if kd_cfg and isinstance(kd_cfg, dict) and kd_cfg.get("use_kd", False) and (teacher is not None):
                warmup = int(kd_cfg.get("warmup_epochs", 5))
                ramp = int(kd_cfg.get("ramp_epochs", 10))
                if epoch >= warmup:
                    ramp_ratio = (epoch - warmup) / max(1, ramp)
                    ramp_ratio = float(max(0.0, min(1.0, ramp_ratio)))
                else:
                    ramp_ratio = 0.0

                if ramp_ratio > 0.0:
                    T = float(kd_cfg.get("temp", 4.0))
                    thr = float(kd_cfg.get("score_thr", 0.20))
                    w_cls = float(kd_cfg.get("w_cls", 0.7))
                    w_box_l1 = float(kd_cfg.get("w_box_l1", 1.0))
                    w_box_gio = float(kd_cfg.get("w_box_giou", 0.0))

                    with torch.no_grad():
                        teacher.eval()
                        t_out = teacher(samples)

                    s_logits = outputs["pred_logits"]
                    s_boxes = outputs["pred_boxes"]
                    t_logits = t_out["pred_logits"].detach()
                    t_boxes = t_out["pred_boxes"].detach()

                    with torch.no_grad():
                        t_prob = F.softmax(t_logits, dim=-1)[..., :-1]
                        t_max = t_prob.max(dim=-1).values
                        mask = (t_max > thr).float().unsqueeze(-1)

                    kd_cls_q = _kd_cls_per_query(s_logits, t_logits, temp=T)
                    kd_cls = (kd_cls_q * mask.squeeze(-1)).sum() / (mask.sum() + 1e-6)

                    kd_l1_q = _l1_per_query(s_boxes, t_boxes)
                    kd_l1 = (kd_l1_q * mask).sum() / (mask.sum() + 1e-6)

                    kd_giou = _maybe_giou_per_query(s_boxes, t_boxes)
                    if kd_giou is not None and w_box_gio > 0:
                        kd_giou = (kd_giou * mask).sum() / (mask.sum() + 1e-6)
                        loss_kd = (w_cls * kd_cls + w_box_l1 * kd_l1 + w_box_gio * kd_giou) * ramp_ratio
                    else:
                        kd_giou = None
                        loss_kd = (w_cls * kd_cls + w_box_l1 * kd_l1) * ramp_ratio

                    loss_dict["loss_kd"] = loss_kd

            # 反传（含 KD）
            loss = sum(loss_dict.values())
            optimizer.zero_grad()
            loss.backward()

            if max_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

            optimizer.step()

        # ema
        ema = kwargs.get('ema', None) if ema is None else ema
        if ema is not None:
            ema.update(model)

        # 统计与日志
        loss_dict_reduced = reduce_dict(loss_dict)
        loss_value = sum(loss_dict_reduced.values())

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        metric_logger.update(loss=loss_value, **loss_dict_reduced)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(model: torch.nn.Module, criterion: torch.nn.Module, postprocessors, data_loader, base_ds, device, output_dir):
    model.eval()
    criterion.eval()

    metric_logger = MetricLogger(delimiter="  ")
    header = 'Test:'

    # iou_types = tuple(k for k in ('segm', 'bbox') if k in postprocessors.keys())
    iou_types = postprocessors.iou_types
    coco_evaluator = CocoEvaluator(base_ds, iou_types)
    # coco_evaluator.coco_eval[iou_types[0]].params.iouThrs = [0, 0.1, 0.5, 0.75]

    panoptic_evaluator = None

    for samples, targets in metric_logger.log_every(data_loader, 10, header):
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        outputs = model(samples)

        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
        results = postprocessors(outputs, orig_target_sizes)

        res = {target['image_id'].item(): output for target, output in zip(targets, results)}
        if coco_evaluator is not None:
            coco_evaluator.update(res)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    if coco_evaluator is not None:
        coco_evaluator.synchronize_between_processes()
    if panoptic_evaluator is not None:
        panoptic_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    if coco_evaluator is not None:
        coco_evaluator.accumulate()
        coco_evaluator.summarize()

    stats = {}
    if coco_evaluator is not None:
        if 'bbox' in iou_types:
            stats['coco_eval_bbox'] = coco_evaluator.coco_eval['bbox'].stats.tolist()
        if 'segm' in iou_types:
            stats['coco_eval_masks'] = coco_evaluator.coco_eval['segm'].stats.tolist()

    return stats, coco_evaluator
