import torch
import torch.nn as nn
import torch.nn.functional as F

from models.complex.utils import magnitude as mag  # 复数→幅值
from models.complex.utils import apply_real_weight  # 实数权重施加到实/虚
from models.complex.MLP import ChannelMLP

# ----------------------------------------------------------------------------
# Channel‑wise Attention (lazy, complex‑aware)
# ----------------------------------------------------------------------------

class ChannelAttention(nn.Module):
    """通道注意力：内部调用统一 ChannelMLP（自带 lazy build）。"""

    def __init__(self, reduction_ratio: int = 16):
        super().__init__()
        self.mlp = ChannelMLP(reduction_ratio)  # lazy 构建
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.max_pool = nn.AdaptiveMaxPool3d(1)
        self.act = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_mag = mag(x)
        attn = self.mlp(self.avg_pool(x_mag)) + self.mlp(self.max_pool(x_mag))
        return self.act(attn)  # (B,C,1,1,1)

# ----------------------------------------------------------------------------
# Spatial Attention (固定 2→1 卷积)
# ----------------------------------------------------------------------------

class SpatialAttention(nn.Module):
    """空间注意力，无需通道自适应。"""

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

# ----------------------------------------------------------------------------
# Dual‑attention Fusion (lazy channel‑adaptive)
# ----------------------------------------------------------------------------

class AttentionFusion(nn.Module):
    """CBAM‑风格频域+空间域特征融合，自适应通道数。"""

    def __init__(self):
        super().__init__()
        self.total_c: int | None = None  # 首次 forward 后确定
        self.chan_att = ChannelAttention()
        self.spa_att = SpatialAttention()
        self.fuse_conv: nn.Sequential | None = None  # lazy 构建

    # ------------------------------------------------------------------
    def _build_fuse_conv(self, channels: int):
        self.fuse_conv = nn.Sequential(
            nn.Conv3d(channels, channels, 3, padding=1, bias=False),
            nn.InstanceNorm3d(channels),
            nn.ReLU(inplace=True),
        )

    # ------------------------------------------------------------------
    def forward(self, ch_feat: torch.Tensor, sp_feat: torch.Tensor) -> torch.Tensor:
        # 尺度对齐
        if ch_feat.shape[2:] != sp_feat.shape[2:]:
            sp_feat = F.interpolate(sp_feat, size=ch_feat.shape[2:], mode="trilinear", align_corners=True)

        x = torch.cat([ch_feat, sp_feat], dim=1)  # (B,C_total,*)

        # 首次前向：根据实际通道构建融合卷积
        if self.total_c is None:
            self.total_c = x.shape[1]
            self._build_fuse_conv(self.total_c)
        assert self.fuse_conv is not None  # mypy guard

        # Channel attention
        w_c = self.chan_att(x)
        x = apply_real_weight(x, w_c)

        # Spatial attention
        w_s = self.spa_att(x)
        x = apply_real_weight(x, w_s)

        # 残余连接 + 卷积融合
        x = x + torch.cat([ch_feat, sp_feat], dim=1)
        x = self.fuse_conv(x)
        return x
