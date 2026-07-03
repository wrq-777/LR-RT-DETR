import torch
import torch.nn as nn
import torch.nn.functional as F
from src.core import register

__all__ = ['GhostVoVEMCAD']


# ================================================================
# Part 1: LRT-DETR 核心组件 (GSConv & VoVGSCSP)
# ================================================================

class GSConv(nn.Module):
    """
    GSConv: 混合标准卷积与深度可分离卷积 + Channel Shuffle
    比 GhostConv 梯度流更好，专门用于 Neck 层
    """

    def __init__(self, c1, c2, k=1, s=1, g=1, act=True):
        super().__init__()
        c_ = c2 // 2
        self.cv1 = nn.Conv2d(c1, c_, k, s, k // 2, bias=False)
        self.cv2 = nn.Conv2d(c_, c_, 5, 1, 5 // 2, groups=c_, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU() if act else nn.Identity()

    def forward(self, x):
        x1 = self.cv1(x)
        x2 = self.cv2(x1)
        out = torch.cat((x1, x2), 1)
        # Channel Shuffle
        b, c, h, w = out.shape
        out = out.reshape(b, 2, c // 2, h, w).permute(0, 2, 1, 3, 4).reshape(b, c, h, w)
        return self.act(self.bn(out))


class VoVGSCSP(nn.Module):
    """
    VoVGSCSP: 基于 GSConv 的高效 CSP 模块
    用来替换 EMCAD 原版笨重的 MSCB 模块
    """

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = nn.Conv2d(c1, c_, 1, 1, bias=False)
        self.cv2 = nn.Conv2d(c1, c_, 1, 1, bias=False)
        self.m = nn.Sequential(*(GSConv(c_, c_, k=3, s=1) for _ in range(n)))
        self.conv_out = nn.Conv2d(c_ * 2 + c_, c2, 1, 1, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU()

    def forward(self, x):
        x1 = self.cv1(x)
        x2 = self.cv2(x)
        x3 = self.m(x1)
        out = torch.cat((x1, x2, x3), dim=1)
        return self.act(self.bn(self.conv_out(out)))


# ================================================================
# Part 2: EMCAD 核心组件 (注意力门控 LGAG & 上采样 EUCB)
# ================================================================

class LGAG(nn.Module):
    """
    Large-Kernel Grouped Attention Gate
    EMCAD 的精髓：用深层特征(g)去'门控'浅层特征(x)，抑制背景噪声
    """

    def __init__(self, F_g, F_l, kernel_size=3):
        super(LGAG, self).__init__()
        F_int = F_g // 2
        # 使用轻量级卷积
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size, 1, kernel_size // 2, bias=False),
            nn.BatchNorm2d(F_int)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size, 1, kernel_size // 2, bias=False),
            nn.BatchNorm2d(F_int)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, 1, 1, 0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi


class EUCB(nn.Module):
    """Efficient Up-Convolution Block"""

    def __init__(self, c1, c2):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        # 上采样后接一个 GSConv 平滑特征
        self.conv = GSConv(c1, c2, k=3)

    def forward(self, x):
        return self.conv(self.up(x))


# ================================================================
# Part 3: 主 Neck 模块 (Ghost-VoV-EMCAD)
# ================================================================

@register
class GhostVoVEMCAD(nn.Module):
    """
    Ghost-VoV-EMCAD Neck
    输入: Backbone Features [P3, P4, P5]
    输出: Fused Features [P3, P4, P5] (Hidden Dim)
    """

    def __init__(self, in_channels=[160, 320, 320], hidden_dim=256, expansion=0.5):
        super().__init__()

        # 1. Input Projection: 将不同维度的 P3, P4, P5 统一映射到 hidden_dim
        self.input_proj = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(c, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim)
            ) for c in in_channels
        ])

        # 2. 核心融合层 (VoVGSCSP)
        # P5 (High Level)
        self.block_p5 = VoVGSCSP(hidden_dim, hidden_dim, n=1, e=expansion)

        # P5 -> P4
        self.up_p5_p4 = EUCB(hidden_dim, hidden_dim)
        self.att_p4 = LGAG(hidden_dim, hidden_dim)  # 门控
        self.block_p4 = VoVGSCSP(hidden_dim, hidden_dim, n=1, e=expansion)

        # P4 -> P3
        self.up_p4_p3 = EUCB(hidden_dim, hidden_dim)
        self.att_p3 = LGAG(hidden_dim, hidden_dim)  # 门控
        self.block_p3 = VoVGSCSP(hidden_dim, hidden_dim, n=1, e=expansion)

    def forward(self, feats):
        # 1. 投影对齐
        # feats: [P3, P4, P5]
        p3, p4, p5 = [proj(f) for f, proj in zip(feats, self.input_proj)]

        # 2. 自顶向下融合 (Top-Down Pathway with Attention Gates)

        # 处理 P5
        d5 = self.block_p5(p5)

        # 处理 P4: P5上采样 -> 门控P4 -> 融合 -> VoVGSCSP
        d4_up = self.up_p5_p4(d5)
        # 注意力门控: 用深层特征(d4_up)告诉浅层(p4)哪里是背景哪里是缺陷
        p4_gated = self.att_p4(g=d4_up, x=p4)
        d4 = d4_up + p4_gated
        d4 = self.block_p4(d4)

        # 处理 P3
        d3_up = self.up_p4_p3(d4)
        p3_gated = self.att_p3(g=d3_up, x=p3)
        d3 = d3_up + p3_gated
        d3 = self.block_p3(d3)

        # 输出 RT-DETR 需要的格式
        return [d3, d4, d5]