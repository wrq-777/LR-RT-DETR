import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# -----------------------------------------------------------------
# 1. 基础辅助函数
# -----------------------------------------------------------------
def _init_weights(module, scheme=''):
    if isinstance(module, nn.Conv2d) or isinstance(module, nn.Conv3d):
        if scheme == 'normal':
            nn.init.normal_(module.weight, std=.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif scheme == 'trunc_normal':
            nn.init.trunc_normal_(module.weight, std=.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        else:
            fan_out = module.kernel_size[0] * module.kernel_size[1] * module.out_channels
            fan_out //= module.groups
            nn.init.normal_(module.weight, 0, math.sqrt(2.0 / fan_out))
            if module.bias is not None:
                nn.init.zeros_(module.bias)
    elif isinstance(module, nn.BatchNorm2d) or isinstance(module, nn.BatchNorm3d):
        nn.init.constant_(module.weight, 1)
        nn.init.constant_(module.bias, 0)


def act_layer(act, inplace=False, neg_slope=0.2):
    act = act.lower()
    if act == 'relu':
        return nn.ReLU(inplace)
    elif act == 'relu6':
        return nn.ReLU6(inplace)
    elif act == 'gelu':
        return nn.GELU()
    elif act == 'silu':
        return nn.SiLU(inplace)
    else:
        return nn.ReLU(inplace)


# -----------------------------------------------------------------
# 2. LGAG (Large-kernel Grouped Attention Gate) - 用于特征融合
# -----------------------------------------------------------------
class LGAG(nn.Module):
    def __init__(self, F_g, F_l, F_int, kernel_size=3, groups=1, activation='relu'):
        super(LGAG, self).__init__()

        if kernel_size == 1:
            groups = 1

        # 处理门控信号 g (High-level)
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=kernel_size, stride=1, padding=kernel_size // 2, groups=groups,
                      bias=True),
            nn.BatchNorm2d(F_int)
        )
        # 处理输入特征 x (Low-level)
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=kernel_size, stride=1, padding=kernel_size // 2, groups=groups,
                      bias=True),
            nn.BatchNorm2d(F_int)
        )
        # 生成注意力系数
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.activation = act_layer(activation, inplace=True)

        self._init_weights()

    def _init_weights(self):
        self.apply(lambda m: _init_weights(m, scheme='normal'))

    def forward(self, g, x):
        if g.size()[2:] != x.size()[2:]:
            g = F.interpolate(g, size=x.size()[2:], mode='bilinear', align_corners=False)

        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.activation(g1 + x1)
        psi = self.psi(psi)

        return x * psi


# -----------------------------------------------------------------
# 3. MSCAM (Multi-scale Convolutional Attention Module) - 用于特征提取
# -----------------------------------------------------------------

class CAB(nn.Module):
    """ Channel Attention Block (通道注意力) """

    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        mip = max(8, in_channels // reduction)

        self.conv1 = nn.Conv2d(in_channels, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(mip, in_channels, kernel_size=1, stride=1, padding=0)
        self.bn2 = nn.BatchNorm2d(in_channels)

    def forward(self, x):
        identity = x
        n, c, h, w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)

        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)

        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        a_h = self.conv2(x_h).sigmoid()
        a_w = self.conv2(x_w).sigmoid()

        return identity * a_h * a_w


class SAB(nn.Module):
    """ Spatial Attention Block (空间注意力) """

    def __init__(self, kernel_size=7):
        super().__init__()
        # 论文建议 7x7，为了轻量化也可以 3x3，这里用 7x7 保证感受野
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        scale = self.sigmoid(self.conv1(x_cat))
        return x * scale


class MSCB(nn.Module):
    """ Multi-Scale Convolution Block (多尺度卷积块) """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        # 这里的 expansion 设为 1 以保持通道数稳定，轻量化
        mid_channels = in_channels

        # 1. 1x1 变换
        self.expand_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, 1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True)
        )

        # 2. 多尺度深度卷积 (Parallel 1x1, 3x3, 5x5)
        # 1x1 分支
        self.dw1 = nn.Sequential(
            nn.Conv2d(mid_channels, mid_channels, 1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True)
        )
        # 3x3 分支
        self.dw3 = nn.Sequential(
            nn.Conv2d(mid_channels, mid_channels, 3, padding=1, groups=mid_channels, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True)
        )
        # 5x5 分支
        self.dw5 = nn.Sequential(
            nn.Conv2d(mid_channels, mid_channels, 5, padding=2, groups=mid_channels, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True)
        )

        # 3. 投影输出
        self.project_conv = nn.Sequential(
            nn.Conv2d(mid_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        x_exp = self.expand_conv(x)

        # 融合多尺度特征
        s1 = self.dw1(x_exp)
        s3 = self.dw3(x_exp)
        s5 = self.dw5(x_exp)
        s = s1 + s3 + s5

        out = self.project_conv(s)

        # 残差连接 (如果维度允许)
        if x.shape == out.shape:
            return x + out
        return out


class MSCAMBlock(nn.Module):
    """
    完整的 MSCAM 堆叠模块: (CAB -> SAB -> MSCB) * n
    直接替换 RepC3
    """

    def __init__(self, in_channels, out_channels, n=1, **kwargs):
        super().__init__()
        self.blocks = nn.ModuleList()

        # 如果输入输出通道不同，第一个块负责维度变换
        self.blocks.append(self._make_layer(in_channels, out_channels))

        # 后续块重复 (输入输出都是 out_channels)
        for _ in range(n - 1):
            self.blocks.append(self._make_layer(out_channels, out_channels))

    def _make_layer(self, in_c, out_c):
        return nn.Sequential(
            CAB(in_c),
            SAB(),
            MSCB(in_c, out_c)
        )

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x