import math
from functools import lru_cache
from typing import Tuple

import torch
import torch.nn.functional as F

from models.complex.utils import apply_to_complex

# complex‑aware grid_sample (real/imag split automatically handled)
complex_grid_sample = apply_to_complex(F.grid_sample)


class CylindricalMapping:
    """Differentiable Cartesian ↔ Cylindrical mapping for 3‑D volumes.

    Notes
    -----
    * Supports **real or complex** tensors of shape (B,C,D,H,W).
    * All coordinates are normalised to **[-1,1]** for `grid_sample`.
    * Grid is cached per spatial shape on **first call** (use LRU).
    """

    def __init__(self,
                 r_samples: int = 32,
                 theta_samples: int = 36,
                 z_samples: int = 32):
        self.r_samples = r_samples
        self.theta_samples = theta_samples
        self.z_samples = z_samples

    # ------------------------------------------------------------------
    # grid builders (cached)
    # ------------------------------------------------------------------

    @lru_cache(maxsize=8)
    def _cartesian_grid(self, D: int, H: int, W: int, device: torch.device) -> torch.Tensor:
        """Return (1,D,H,W,3) grid that samples cylindrical → Cartesian."""
        z = torch.linspace(-1, 1, D, device=device)
        y = torch.linspace(-1, 1, H, device=device)
        x = torch.linspace(-1, 1, W, device=device)
        zz, yy, xx = torch.meshgrid(z, y, x, indexing='ij')

        r = torch.sqrt(xx ** 2 + yy ** 2)
        theta = torch.atan2(yy, xx) % (2 * math.pi)

        r_norm = r                    # [0,1]
        theta_norm = theta / (2 * math.pi)
        z_norm = (zz + 1) / 2         # [-1,1] → [0,1]

        grid = torch.stack([
            2 * z_norm - 1,
            2 * theta_norm - 1,
            2 * r_norm - 1,
        ], dim=-1)                   # (D,H,W,3)
        return grid.unsqueeze(0)      # (1,D,H,W,3)

    @lru_cache(maxsize=8)
    def _cylindrical_grid(self, D: int, H: int, W: int, device: torch.device) -> torch.Tensor:
        """Return (1,r,θ,z,3) grid that samples Cartesian → cylindrical."""
        r = torch.linspace(0, 1, self.r_samples, device=device)
        theta = torch.linspace(0, 2 * math.pi, self.theta_samples, device=device)
        z = torch.linspace(-1, 1, self.z_samples, device=device)
        rr, tt, zz = torch.meshgrid(r, theta, z, indexing='ij')
        xx = rr * torch.cos(tt)
        yy = rr * torch.sin(tt)
        grid = torch.stack([zz, yy, xx], dim=-1)  # (r,θ,z,3)
        return grid.unsqueeze(0)                  # (1,r,θ,z,3)

    # ------------------------------------------------------------------
    # forward mappings
    # ------------------------------------------------------------------

    def cartesian_to_cylindrical(self, volume: torch.Tensor) -> torch.Tensor:
        """Map (B,C,D,H,W) → (B,C,r,θ,z)."""
        B, C, D, H, W = volume.shape
        grid = self._cylindrical_grid(D, H, W, volume.device)
        grid = grid.expand(B, -1, -1, -1, -1)      # (B,r,θ,z,3)

        # grid_sample expects format (B,C,D,H,W) with grid (B,D,H,W,3)
        # Our grid is (B,r,θ,z,3) → permute to match (D_out,H_out,W_out)
        vol_perm = volume                             # (B,C,D,H,W)
        cyl = complex_grid_sample(vol_perm, grid, mode='bilinear', align_corners=True)
        # output is (B,C,r,θ,z)

        # handle r=0 singularity by averaging over θ
        cyl[:, :, 0] = cyl[:, :, 0].mean(dim=2, keepdim=True)
        return cyl

    def cylindrical_to_cartesian(self, cyl: torch.Tensor, out_shape: Tuple[int, int, int]) -> torch.Tensor:
        """Map (B,C,r,θ,z) → (B,C,D,H,W) with given spatial size."""
        B, C = cyl.shape[:2]
        D, H, W = out_shape
        grid = self._cartesian_grid(D, H, W, cyl.device)
        grid = grid.expand(B, -1, -1, -1, -1)       # (B,D,H,W,3)

        # reshape cyl => (B,C,z,θ,r) to align with grid_sample's D,H,W order
        cyl_perm = cyl.permute(0, 1, 4, 3, 2)       # (B,C,z,θ,r)
        cart = complex_grid_sample(cyl_perm, grid, mode='bilinear', align_corners=True)
        return cart
