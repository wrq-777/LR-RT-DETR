# nn/backbone/ghostnetv2.py
import warnings
from typing import Iterable, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import timm
from src.core.yaml_utils import register  # 与 presnet/dla 的导入风格保持一致


def _normalize_ghost_name(name: Optional[str]) -> str:
    """
    把 1.0/1.3 风格映射成 timm 的 100/130；并允许只写 'ghostnetv2' 或 'ghostnet'。
    例如：
      ghostnetv2_1.0 -> ghostnetv2_100
      ghostnet_1.3   -> ghostnet_130
      ghostnetv2     -> ghostnetv2_100
      ghostnet       -> ghostnet_100
    """
    if not name:
        return "ghostnetv2_100"
    n = name.lower().replace("-", "_")
    n = n.replace("1.0", "100").replace("1_0", "100")
    n = n.replace("1.3", "130").replace("1_3", "130")
    if n in {"ghostnetv2", "ghostnetv2_"}:
        n = "ghostnetv2_100"
    if n in {"ghostnet", "ghostnet_"}:
        n = "ghostnet_100"
    return n


@register
class GhostNetV2Backbone(nn.Module):
    """
    GhostNetV2/V1 backbone 封装（timm），以 features_only 形式输出三层特征：
      S3(≈1/8), S4(≈1/16), S5(≈1/32)
    供 RT-DETR 的 HybridEncoder 使用。

    参数
    ----
    model_name: str
        timm 的模型名；推荐 'ghostnetv2_100'（或 'ghostnet_100'）。
        也兼容 'ghostnetv2_1.0'/'ghostnet_1.0' 写法（自动归一化）。
    pretrained: bool
        是否载入 ImageNet 预训练权重。
    out_indices: Sequence[int]
        选择哪些 stage 作为输出（默认 (2,3,4) ≈ strides {8,16,32}）。
    in_chans: int
        输入通道数，默认为 3。
    variant / return_idx: 兼容字段（若 YAML 里这么写也能正常工作）。

    属性
    ----
    out_channels: list[int]
        三层输出的通道数 [C3, C4, C5]。
    strides: list[int]
        三层输出的步幅 [8, 16, 32]（通常）。
    """

    def __init__(
        self,
        model_name: str = "ghostnetv2_100",
        pretrained: bool = True,
        out_indices: Sequence[int] = (2, 3, 4),
        in_chans: int = 3,
        # 兼容 YAML 里可能出现的别名
        variant: Optional[str] = None,
        return_idx: Optional[Iterable[int]] = None,
        **kwargs,
    ):
        super().__init__()

        # 兼容：若传了 variant / return_idx，则覆盖主参数
        if variant and not model_name:
            model_name = variant
        if return_idx is not None:
            out_indices = tuple(return_idx)
        else:
            out_indices = tuple(out_indices)

        want = _normalize_ghost_name(model_name)

        # 枚举本机可用的 ghostnet 模型
        avail_v2 = set(timm.list_models("ghostnetv2*"))
        avail_v1 = set(timm.list_models("ghostnet*")) - avail_v2  # 剔除 v2，只留 v1
        chosen = None

        # 先尝试用户期望
        if want in avail_v2 or want in avail_v1:
            chosen = want
        # 再尝试常见候选
        if chosen is None:
            for cand in ("ghostnetv2_100", "ghostnetv2_130", "ghostnet_100", "ghostnet_130"):
                if cand in avail_v2 or cand in avail_v1:
                    chosen = cand
                    break

        if chosen is None:
            raise RuntimeError(
                "No GhostNet(V2/V1) models found in timm.\n"
                f"Requested: {model_name}\n"
                f"Available v2: {sorted(avail_v2)}\n"
                f"Available v1: {sorted(avail_v1)}\n"
                "Tip: pip install -U timm"
            )

        if chosen != want:
            warnings.warn(f"[GhostNetV2Backbone] '{model_name}' not found; fallback to '{chosen}'.", RuntimeWarning)

        # 创建 timm 模型（features_only + 指定 out_indices）
        self.body = timm.create_model(
            chosen,
            pretrained=pretrained,
            features_only=True,
            out_indices=out_indices,
            in_chans=in_chans,
        )

        # 记录通道与步幅信息
        # timm 的 FeatureInfo 支持 .channels() / .reduction()
        info = self.body.feature_info
        try:
            self.out_channels = list(info.channels())
            self.strides = list(info.reduction())
        except Exception:
            # 极老的 timm 版本兜底（不太可能用到）
            self.out_channels = [fi["num_chs"] for fi in info]
            self.strides = [fi["reduction"] for fi in info]

        # 期望三个尺度
        if len(self.out_channels) != 3:
            warnings.warn(
                f"[GhostNetV2Backbone] Expect 3 feature maps, got {len(self.out_channels)} "
                f"with out_indices={out_indices}.", RuntimeWarning
            )

    def forward(self, x: torch.Tensor) -> Sequence[torch.Tensor]:
        """
        返回三层特征 [S3, S4, S5]，形状大致为：
          [B, C3, H/8,  W/8 ],
          [B, C4, H/16, W/16],
          [B, C5, H/32, W/32]
        """
        feats = self.body(x)
        return feats

    @torch.no_grad()
    def probe(self, size: Tuple[int, int] = (640, 640), device: str = "cpu"):
        """
        生成一张零张量测试图，返回：
          (shapes, out_channels, strides)
        便于把 in_channels/feat_strides 精确写回 YAML。
        """
        x = torch.zeros(1, 3, size[0], size[1], device=device)
        feats = self.forward(x)
        shapes = [f.shape for f in feats]
        return shapes, self.out_channels, self.strides
