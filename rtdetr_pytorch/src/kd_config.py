# src/kd_config.py

KD_CFG = {
    # 是否启用知识蒸馏
    "enable": True,

    # 教师的 yml 配置路径（相对当前工作目录 rtdetr_pytorch）
    "teacher_cfg": "configs/rtdetr/neu_det_r50.yml",

    # 教师权重（建议用 best_ema.pth）
    "teacher_ckpt": "output/neu_det_r50/best_ema.pth",

    # 分类 KD 超参
    "cls_t": 2.0,      # 温度 tau
    "cls_alpha": 0.6,  # 分类 KD loss 系数

    # 边框 KD 超参
    "box_alpha": 0.5,  # 边框 L1 KD loss 系数
}
