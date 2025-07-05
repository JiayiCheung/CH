"""Lazy‑adaptive Multiscale Fusion (Tier‑aware).

Removes the hard‑coded `in_channels` argument: attention sub‑network is created
on the first forward pass when the true channel count is known.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional,Tuple

__all__ = ["MultiscaleFusion"]


class MultiscaleFusion(nn.Module):
    """Fuse feature maps from up to three tier levels with learned attention."""

    def __init__(self):
        super().__init__()
        self.attention: Optional[nn.Sequential]  = None  # lazy build

    # ------------------------------------------------------------------
    def _build_attention(self, C: int, ref: torch.Tensor):
        """Create attention sub‑network once channels `C` are known."""
        self.attention = nn.Sequential(
            nn.Conv3d(C, max(4, C // 2), 3, padding=1),  # hidden dim >=4
            nn.InstanceNorm3d(max(4, C // 2)),
            nn.ReLU(inplace=True),
            nn.Conv3d(max(4, C // 2), 3, 1),             # weight for 3 tiers
            nn.Softmax(dim=1),
        )
        self.attention.to(ref.device, dtype=ref.dtype)

    # ------------------------------------------------------------------
    def forward(
        self,
        tier_features: Dict[int, torch.Tensor],
        target_shape: Optional[Tuple[int, int, int]] = None,
    ) -> torch.Tensor:
        if not tier_features:
            raise ValueError("No tier features provided")

        available = sorted(tier_features.keys())
        # target shape: follow tier‑0 else first available tier
        if target_shape is None:
            tgt_feat = tier_features.get(0, tier_features[available[0]])
            target_shape = tgt_feat.shape[2:]

        # align spatial dims
        aligned: Dict[int, torch.Tensor] = {}
        for t in available:
            f = tier_features[t]
            if f.shape[2:] != target_shape:
                aligned[t] = F.interpolate(f, size=target_shape, mode="trilinear", align_corners=True)
            else:
                aligned[t] = f

        # if only one tier present return directly
        if len(aligned) == 1:
            return next(iter(aligned.values()))

        # stack into [B, 3, C, D, H, W] – zero padding for missing tiers
        B, C = next(iter(aligned.values())).shape[:2]
        device, dtype = next(iter(aligned.values())).device, next(iter(aligned.values())).dtype
        all_feats = []
        for t in range(3):
            all_feats.append(aligned.get(
                t,
                torch.zeros((B, C, *target_shape), device=device, dtype=dtype),
            ))
        stacked = torch.stack(all_feats, dim=1)  # [B,3,C,D,H,W]

        # build attention net if needed
        if self.attention is None:
            self._build_attention(C, stacked)
        # guard in case parent .to() after build
        if next(self.attention.parameters()).device != stacked.device:
            self.attention.to(stacked.device, dtype=stacked.dtype)

        # reshape to feed attention conv
        B, T, C, D, H, W = stacked.shape
        att_in = stacked.permute(0, 2, 1, 3, 4, 5).reshape(B, C, T * D, H, W)
        weights = self.attention(att_in)                 # [B,3,T*D,H,W]
        weights = weights.reshape(B, 3, T, D, H, W).permute(0, 2, 1, 3, 4, 5)

        fused = torch.sum(stacked * weights, dim=1)      # [B,C,D,H,W]
        return fused
