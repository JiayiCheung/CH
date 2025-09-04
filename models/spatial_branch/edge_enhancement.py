
from __future__ import annotations

import math
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["EdgeEnhancement"]


class EdgeEnhancement(nn.Module):
    """3‑D Sobel / Laplacian edge‑enhancement block (vectorised)."""

    def __init__(self, out_channels: int = 4):
        super().__init__()
        self._kernels: List[torch.Tensor] = []  # filled by _make_kernels()
        self._make_kernels()
        self.combine: Optional[nn.Conv3d]  = None
        self.norm:    Optional[nn.GroupNorm] = None
        self.act      = nn.ReLU(inplace=True)
        self.out_ch   = out_channels

    # ------------------------------------------------------------
    def _make_kernels(self):
        """Create 4 fixed 3×3×3 kernels via NumPy (same as original impl)."""
        import numpy as np
        sobel_kernels = []
        # X‑direction Sobel
        kx = np.zeros((3,3,3), dtype=np.float32)
        kx[:,:,0] = np.array([[-1,-2,-1],[-2,-4,-2],[-1,-2,-1]], dtype=np.float32)
        kx[:,:,2] = -kx[:,:,0]
        sobel_kernels.append(kx)
        # Y‑direction Sobel
        ky = np.zeros((3,3,3), dtype=np.float32)
        ky[:,0,:] = kx[:,:,0]
        ky[:,2,:] = -kx[:,:,0]
        sobel_kernels.append(ky)
        # Z‑direction Sobel
        kz = np.zeros((3,3,3), dtype=np.float32)
        kz[0,:,:] = kx[:,:,0]
        kz[2,:,:] = -kx[:,:,0]
        sobel_kernels.append(kz)
        # 3‑D Laplacian
        lap = np.full((3,3,3), -1, dtype=np.float32)
        lap[1,1,1] = 26.
        sobel_kernels.append(lap)
        # Normalize & stack
        normed = [k / np.abs(k).sum() for k in sobel_kernels]
        kernel_stack = torch.from_numpy(np.stack(normed)).unsqueeze(1)  # [K,1,3,3,3]
        self.kernels = nn.Parameter(kernel_stack.clone(), requires_grad=True)


    # ------------------------------------------------------------
    def _build_head(self, C_in: int, ref: torch.Tensor):
        K = self.kernels.shape[0]
        self.combine = nn.Conv3d(K * C_in, self.out_ch, 1, bias=False)
        groups = max(4, self.out_ch // 4)
        self.norm   = nn.GroupNorm(groups, self.out_ch, affine=True)
        for m in (self.combine, self.norm):
            m.to(ref.device, dtype=ref.dtype)

    # ------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, D, H, W = x.shape
        K = self.kernels.shape[0]

        # depth‑wise grouped conv ==> produce [B, K*C, D,H,W]
        # Expand kernels to [K*C,1,3,3,3] where groups=C
        weight = self.kernels.repeat_interleave(C, dim=0)
        edge = F.conv3d(
            x,  # 保持原始形状 [B, 1, D, H, W]
            weight,  # [K, 1, 3, 3, 3]
            bias=None, padding=1,
            groups=1,  # 标准卷积
        )

        # lazy build combine + GN
        if self.combine is None:
            self._build_head(C, edge)
        # guard after parent .to()
        if next(self.combine.parameters()).device != edge.device:
            for m in (self.combine, self.norm):
                m.to(edge.device, dtype=edge.dtype)

        out = self.combine(edge)
        out = self.norm(out)
        return self.act(out)
