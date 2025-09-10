import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

# ---------------- Depth-wise separable 3-D conv ----------------
class DepthwiseSeparableConv3d(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, k: int = 3, stride: int = 1, pad: Optional[int] = None ):
        super().__init__()
        pad = k // 2 if pad is None else pad
        self.dw = nn.Conv3d(in_ch, in_ch, k, stride, pad, groups=in_ch, bias=False)
        self.pw = nn.Conv3d(in_ch, out_ch, 1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pw(self.dw(x))


# ---------------- Lightweight Spatial Feature Extractor --------
class SpatialFeatureExtractor(nn.Module):
    """
    in  → Conv3d → GN → ReLU
        → DWConv → GN → ReLU
        → DWConv → GN → ReLU → out
    + residual (identity or 1×1 Conv)
    """

    def __init__(self, in_channels: int = 1, mid_channels: int = 16, out_channels: int = 16):
        super().__init__()

        # Helper to choose reasonable GN groups
        def _gn(c: int) -> nn.GroupNorm:
            return nn.GroupNorm(num_groups=max(4, c // 4), num_channels=c, affine=True)

        self.encoder = nn.Sequential(
            nn.Conv3d(in_channels, mid_channels, 3, padding=1, bias=False),
            _gn(mid_channels), nn.ReLU(inplace=True),

            DepthwiseSeparableConv3d(mid_channels, mid_channels * 2),
            _gn(mid_channels * 2), nn.ReLU(inplace=True),

            DepthwiseSeparableConv3d(mid_channels * 2, out_channels),
            _gn(out_channels), nn.ReLU(inplace=True),
        )

        # Residual path
        self.shortcut = (
            nn.Conv3d(in_channels, out_channels, 1, bias=False)
            if in_channels != out_channels else nn.Identity()
        )

    # ------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.encoder(x)
        res  = self.shortcut(x)

        # spatial shape guard (rare, only if stride ≠1 added later)
        if res.shape[2:] != feat.shape[2:]:
            res = F.interpolate(res, size=feat.shape[2:], mode="trilinear", align_corners=True)

        return feat + res
