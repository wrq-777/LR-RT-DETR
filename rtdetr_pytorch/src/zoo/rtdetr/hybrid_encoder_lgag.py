import torch
import torch.nn as nn
import torch.nn.functional as F

from src.core import register
from .hybrid_encoder import HybridEncoder
from src.nn.emcad_ops import LGAG, MSCAMBlock


@register
class HybridEncoderLGAG(HybridEncoder):
    """
    【版本一：NEU-DET 专用】
    配置：n=3 (轻量化，速度快)
    """

    def __init__(self,
                 lgag_kernel_size=3,
                 in_channels=[40, 80, 160],
                 feat_strides=[8, 16, 32],
                 hidden_dim=192,
                 use_encoder_idx=[2],
                 num_encoder_layers=1,
                 nhead=6,
                 dim_feedforward=768,
                 dropout=0.0,
                 enc_act='gelu',
                 pe_temperature=10000,
                 expansion=1.0,
                 depth_mult=1.0,
                 act='silu',
                 eval_spatial_size=[640, 640],
                 **kwargs):

        super().__init__(
            in_channels=in_channels,
            feat_strides=feat_strides,
            hidden_dim=hidden_dim,
            use_encoder_idx=use_encoder_idx,
            num_encoder_layers=num_encoder_layers,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            enc_act=enc_act,
            pe_temperature=pe_temperature,
            expansion=expansion,
            depth_mult=depth_mult,
            act=act,
            eval_spatial_size=eval_spatial_size,
            **kwargs
        )

        # =========================================================
        # ★★★ 关键设置：n=3 (保持轻量) ★★★
        # =========================================================
        mscam_depth = 3

        # 1. 替换 FPN (自顶向下)
        for i in range(len(self.fpn_blocks)):
            self.fpn_blocks[i] = MSCAMBlock(
                in_channels=hidden_dim * 2,
                out_channels=hidden_dim,
                n=mscam_depth
            )

        # 2. 替换 PAN (自底向上)
        for i in range(len(self.pan_blocks)):
            self.pan_blocks[i] = MSCAMBlock(
                in_channels=hidden_dim * 2,
                out_channels=hidden_dim,
                n=mscam_depth
            )

        # 3. 初始化 LGAG
        self.lgag_modules = nn.ModuleList()
        num_fpn_fusion = len(self.in_channels) - 1

        for _ in range(num_fpn_fusion):
            self.lgag_modules.append(
                LGAG(
                    F_g=hidden_dim,
                    F_l=hidden_dim,
                    F_int=hidden_dim // 2,
                    kernel_size=lgag_kernel_size,
                    groups=hidden_dim // 2,
                    activation='relu'
                )
            )

    def forward(self, feats):
        assert len(feats) == len(self.in_channels)
        proj_feats = [self.input_proj[i](feat) for i, feat in enumerate(feats)]

        if self.use_encoder_idx:
            for i, enc_ind in enumerate(self.use_encoder_idx):
                h, w = proj_feats[enc_ind].shape[2:]
                src_flatten = proj_feats[enc_ind].flatten(2).permute(0, 2, 1)
                if self.num_encoder_layers > 0:
                    memory = self.encoder[i](src_flatten)
                    proj_feats[enc_ind] = memory.permute(0, 2, 1).reshape(-1, self.hidden_dim, h, w)
                else:
                    proj_feats[enc_ind] = src_flatten.permute(0, 2, 1).reshape(-1, self.hidden_dim, h, w)

        inner_outs = [proj_feats[-1]]
        for idx in range(len(self.in_channels) - 1, 0, -1):
            feat_high = inner_outs[0]
            feat_low = proj_feats[idx - 1]
            feat_high_up = F.interpolate(feat_high, scale_factor=2., mode='nearest')

            lgag_idx = len(self.in_channels) - 1 - idx
            feat_low_refined = self.lgag_modules[lgag_idx](g=feat_high_up, x=feat_low)

            concat_feat = torch.cat([feat_low_refined, feat_high_up], dim=1)
            inner_outs.insert(0, self.fpn_blocks[idx - 1](concat_feat))

        outs = [inner_outs[0]]
        for idx in range(len(self.in_channels) - 1):
            feat_low = outs[-1]
            feat_high = inner_outs[idx + 1]
            down_feat = self.downsample_convs[idx](feat_low)
            concat_feat = torch.cat([down_feat, feat_high], dim=1)
            outs.append(self.pan_blocks[idx](concat_feat))

        return outs