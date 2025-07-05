
from __future__ import annotations

from pathlib import Path
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from .ch_branch import CHBranch
from .spatial_branch import SpatialBranch
from .fusion.attention_fusion import AttentionFusion
from .fusion.multiscale_fusion import MultiscaleFusion
from .spatial_branch.edge_enhancement import EdgeEnhancement

__all__ = ["VesselSegmenter"]


class VesselSegmenter(nn.Module):
    """Three‑tier TA‑CHNet with fully adaptive channel dimensions."""

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        ch_params: Dict | None = None,
        tier_params: Dict | None = None,
    ) -> None:
        super().__init__()

        # ------------------------------------------------------------------
        # Branches
        # ------------------------------------------------------------------
        if ch_params is None:
            ch_params = {
                "max_n": 3,
                "max_k": 4,
                "max_l": 5,
                "cylindrical_dims": (32, 36, 32),
            }
        if tier_params is None:
            tier_params = {
                0: {"max_n": 2, "max_k": 3, "max_l": 4},
                1: {"max_n": 3, "max_k": 4, "max_l": 5},
                2: {"max_n": 4, "max_k": 5, "max_l": 6},
            }
        self.tier_params = tier_params
        self.current_tier: int | None = None
        self.edge_enhance = EdgeEnhancement(out_channels=8)

        self.ch_branch = CHBranch(**ch_params)
        self.spatial_branch = SpatialBranch(in_channels, 16, 16)  # 内部自己决定输出通道

        # ------------------------------------------------------------------
        # Fusion & head – lazy adaptive
        # ------------------------------------------------------------------
        self.attention_fusion = AttentionFusion()       # lazy inside
        self.multiscale_fusion = MultiscaleFusion()     # lazy inside

        self.seg_head_first: nn.Conv3d | None = None    # lazy build on first forward
        self.seg_head_tail = nn.Sequential(
            nn.InstanceNorm3d(32),
            nn.ReLU(inplace=True),
            nn.Conv3d(32, out_channels, 1),
            nn.Sigmoid() if out_channels == 1 else nn.Softmax(dim=1),
        )

        # cache for tier features
        self.tier_features: Dict[int, torch.Tensor] = {}

    # ------------------------------------------------------------------
    # Utility builders
    # ------------------------------------------------------------------
    def _build_seg_head(self, in_c: int, ref: torch.Tensor):
        """Create first 3×3 conv to map `in_c → 32`, align device/dtype."""
        self.seg_head_first = nn.Conv3d(in_c, 32, 3, padding=1, bias=False)
        self.seg_head_first.to(ref.device, dtype=ref.dtype)

    # ------------------------------------------------------------------
    # Tier control helpers
    # ------------------------------------------------------------------
    def set_tier(self, tier: int):
        self.current_tier = tier
        self.ch_branch.set_tier(tier)
        self.tier_features.clear()

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor, tier: int | None = None) -> torch.Tensor:
        if tier is not None:
            self.set_tier(tier)
        if self.current_tier is None:
            raise ValueError("Tier must be set before forward pass")

        # Tier‑specific scaling factor
        r_scale = self.tier_params.get(self.current_tier, {}).get("r_scale", 1.0)
        edge_feat = self.edge_enhance(x)
        
        # Branch forward
        ch_features = self.ch_branch(x, r_scale=r_scale)
        spatial_features = self.spatial_branch(x)

        # Fuse
        fused = self.attention_fusion(ch_features, spatial_features)
        self.tier_features[self.current_tier] = fused

        # Multi‑tier fusion if >1 tier collected
        final = (
            self.multiscale_fusion(self.tier_features)
            if len(self.tier_features) > 1
            else fused
        )

        # Lazy build segmentation head
        if self.seg_head_first is None:
            self._build_seg_head(final.shape[1], final)
        # ensure seg_head_first on right device/dtype even after .to()
        if next(self.seg_head_first.parameters()).device != final.device or \
           next(self.seg_head_first.parameters()).dtype  != final.dtype:
            self.seg_head_first.to(final.device, dtype=final.dtype)

        logits = self.seg_head_tail(self.seg_head_first(final))
        return logits
