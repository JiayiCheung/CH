
from __future__ import annotations

import torch
import torch.nn as nn

__all__ = ["ChannelMLP"]

class ChannelMLP(nn.Module):
    """1×1×1 Conv → ReLU → 1×1×1 Conv block.

    Parameters
    ----------
    reduction_ratio : int, optional
        C//ratio hidden dim, minimum 8.
    bias : bool
        Whether to use bias in conv layers.
    """

    def __init__(self, reduction_ratio: int = 16, bias: bool = False):
        super().__init__()
        self.reduction_ratio = reduction_ratio
        self.bias = bias
        self.net: nn.Sequential | None = None  # lazy

    # ------------------------------------------------------------------
    def _build(self, channels: int, ref: torch.Tensor):
        hidden = max(8, channels // self.reduction_ratio)
        self.net = nn.Sequential(
            nn.Conv3d(channels, hidden, kernel_size=1, bias=self.bias),
            nn.ReLU(inplace=True),
            nn.Conv3d(hidden, channels, kernel_size=1, bias=self.bias),
        )
        self.net.to(ref.device, dtype=ref.dtype)

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B,C,*,*,*) any spatial
        if self.net is None:
            self._build(x.shape[1],x)
        return self.net(x)
