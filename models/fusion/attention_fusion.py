import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

# -----------------------------------------------------------------------------
# 复数工具：直接调用项目统一封装
# -----------------------------------------------------------------------------
from models.complex.tensor_utils import magnitude as mag  # 实->自身, 复->幅值
from models.complex.tensor_utils import apply_real_weight # w 实数, 自动广播到复数实/虚

# -----------------------------------------------------------------------------
# Channel‑wise Attention (CBAM‑style, 3‑D, complex‑aware)
# -----------------------------------------------------------------------------

class ChannelAttention(nn.Module):
    """Channel attention that works for real *or* complex tensors.

    若输入为复数，注意力计算仅基于幅值；生成的权重仍为实数并同时施加于
    实部/虚部。输出维度 `(B,C,1,1,1)`.
    """

    def __init__(self, channels: int, reduction_ratio: int = 16):
        super().__init__()
        hidden = max(8, channels // reduction_ratio)

        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.max_pool = nn.AdaptiveMaxPool3d(1)

        self.mlp = nn.Sequential(
            nn.Conv3d(channels, hidden, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv3d(hidden, channels, 1, bias=False),
        )
        self.act = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_mag = mag(x)                                      # (B,C,1,1,1)
        attn = self.mlp(self.avg_pool(x_mag)) + self.mlp(self.max_pool(x_mag))
        return self.act(attn)

# -----------------------------------------------------------------------------
# Spatial Attention (3‑D, complex‑aware)
# -----------------------------------------------------------------------------

class SpatialAttention(nn.Module):
    """Spatial attention for 3‑D features, complex compatible."""

    def __init__(self, kernel_size: int = 7):
        super().__init__()
        assert kernel_size % 2 == 1, "kernel_size must be odd"
        padding = kernel_size // 2
        self.conv = nn.Conv3d(2, 1, kernel_size, padding=padding, bias=False)
        self.act = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_mag = mag(x)
        avg = x_mag.mean(dim=1, keepdim=True)
        mx, _ = x_mag.max(dim=1, keepdim=True)
        return self.act(self.conv(torch.cat([avg, mx], dim=1)))  # (B,1,D,H,W)

# -----------------------------------------------------------------------------
# Dual‑attention Fusion
# -----------------------------------------------------------------------------

class AttentionFusion(nn.Module):
    """Fuse CH‑branch & spatial‑branch features with CBAM‑like dual attention.

    输入：
        ch_feat  – (B, C_ch, D, H, W)
        sp_feat  – (B, C_sp, D, H, W)
    输出：
        fused    – (B, C_ch+C_sp, D, H, W)
    支持实数与复数特征；当输入异形尺寸时自动三线性插值对齐。
    """

    def __init__(self, ch_channels: int, spatial_channels: int):
        super().__init__()
        self.total_c = ch_channels + spatial_channels
        self.chan_att = ChannelAttention(self.total_c)
        self.spa_att = SpatialAttention()
        self.fuse_conv = nn.Sequential(
            nn.Conv3d(self.total_c, self.total_c, 3, padding=1, bias=False),
            nn.InstanceNorm3d(self.total_c),
            nn.ReLU(inplace=True),
        )

    # ------------------------------------------------------------------
    def forward(self, ch_feat: torch.Tensor, sp_feat: torch.Tensor) -> torch.Tensor:
        # --- spatial align --------------------------------------------------
        if ch_feat.shape[2:] != sp_feat.shape[2:]:
            sp_feat = F.interpolate(sp_feat, size=ch_feat.shape[2:], mode="trilinear", align_corners=True)

        x = torch.cat([ch_feat, sp_feat], dim=1)                # (B,total,*)

        # --- Channel attention --------------------------------------------
        w_c = self.chan_att(x)                                  # (B,C,1,1,1)
        x = apply_real_weight(x, w_c)

        # --- Spatial attention --------------------------------------------
        w_s = self.spa_att(x)                                   # (B,1,D,H,W)
        x = apply_real_weight(x, w_s)

        # --- Residual + conv fusion ---------------------------------------
        x = x + torch.cat([ch_feat, sp_feat], dim=1)
        x = self.fuse_conv(x)
        return x
