import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


from models.complex.utils import complex_to_real, real_to_complex


class CHAttention(nn.Module):
    """Channel‑wise attention on Cylindrical Harmonic (CH) coefficients.

    Each complex CH coefficient (real + imag) is treated as a 2‑channel real
    feature map.  We first extract lightweight features with a 1×1 projection,
    then generate an attention field with depth‑wise separable 3‑D convolutions.
    The attention field modulates both real & imag parts and uses a residual
    connection to preserve the original signal.
    """

    def __init__(self, max_n: int, max_k: int, max_l: int):
        super().__init__()
        # ---------------- hyper parameters ---------------- #
        self.max_n = max_n  # angular order
        self.max_k = max_k  # radial order
        self.max_l = max_l  # axial order

        # CH coefficient spatial dims → (N, K, L)
        self.ch_dims: Tuple[int, int, int] = (
            2 * max_n + 1,
            max_k,
            2 * max_l + 1,
        )

        # ----------------  layers  ---------------- #
        # 1×1 projection (2  →  8 channels) + IN + ReLU
        self.projection = nn.Sequential(
            nn.Conv3d(2, 8, kernel_size=1, bias=False),
            nn.InstanceNorm3d(8, affine=True),
            nn.ReLU(inplace=True),
        )

        # depth‑wise separable convs to generate per‑voxel attention (8→16→2)
        self.attention = nn.Sequential(
            nn.Conv3d(8, 8, kernel_size=3, padding=1, groups=8, bias=False),
            nn.Conv3d(8, 16, kernel_size=1, bias=False),
            nn.InstanceNorm3d(16, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv3d(16, 16, kernel_size=3, padding=1, groups=16, bias=False),
            nn.Conv3d(16, 2, kernel_size=1),
            nn.Sigmoid(),  # (0,1) range attention for real & imag
        )

        # frequency‑aware prior
        self.frequency_encoding = self._create_frequency_encoding()

    # ---------------------------------------------------------------------
    # utilities
    # ---------------------------------------------------------------------
    
    def _create_frequency_encoding(self) -> torch.nn.Parameter:
	    """Pre‑compute a (2, N, K, L) tensor that injects hand‑crafted priors.

		The encoding emphasises low‑frequency components and directional cues
		that correlate with vascular structures, while damping very high‑freq
		noise. The values are heuristically set; you can tune them as needed.
		"""
	    encoding = torch.ones((2, *self.ch_dims), dtype=torch.float32)
	    
	    for n_idx, n in enumerate(range(-self.max_n, self.max_n + 1)):
		    for k_idx, k in enumerate(range(1, self.max_k + 1)):
			    for l_idx, l in enumerate(range(-self.max_l, self.max_l + 1)):
				    # boost broad low‑frequency context
				    if abs(n) <= 1 and k <= 2 and abs(l) <= 2:
					    encoding[:, n_idx, k_idx, l_idx] = 1.2
				    # emphasise dominant vascular orientations (n = ±1)
				    if abs(n) == 1:
					    encoding[:, n_idx, k_idx, l_idx] = 1.3
				    # suppress extreme high‑freq noise
				    if abs(n) > 3 or k > self.max_k - 1 or abs(l) > self.max_l - 1:
					    encoding[:, n_idx, k_idx, l_idx] = 0.8
	    
	    # 返回为可训练参数，与原版一致
	    return nn.Parameter(encoding.unsqueeze(0), requires_grad=True)  # shape (1, 2, N, K, L)

    # ---------------------------------------------------------------------
    # forward
    # ---------------------------------------------------------------------

    def forward(self, ch_coeffs: torch.Tensor) -> torch.Tensor:
        """Apply attention to complex CH coefficients.

        Parameters
        ----------
        ch_coeffs : torch.cfloat  tensor of shape [B, C, N, K, L]
            Complex‑valued CH spectrum.

        Returns
        -------
        torch.cfloat tensor of shape [B, C, N, K, L]
            Spectrum after attention modulation (with residual).
        """
        if not torch.is_complex(ch_coeffs):
            raise TypeError("CHAttention expects complex‑valued `ch_coeffs`.")

        B, C, N, K, L = ch_coeffs.shape
        device = ch_coeffs.device

        # -------- vectorise over channels to remove Python loop -------- #
        real = ch_coeffs.real.view(B * C, 1, N, K, L)
        imag = ch_coeffs.imag.view(B * C, 1, N, K, L)
        coeffs_input = torch.cat([real, imag], dim=1)  # (B*C, 2, N, K, L)

        encoded = coeffs_input * self.frequency_encoding.to(device)

        feats = self.projection(encoded)               # (B*C, 8, …)
        attn = self.attention(feats)                  # (B*C, 2, N, K, L)

        real_att, imag_att = attn[:, 0:1], attn[:, 1:2]
        enhanced_real = real * real_att + real        # residual
        enhanced_imag = imag * imag_att + imag
        enhanced = torch.complex(enhanced_real, enhanced_imag)

        # reshape back to [B, C, …]
        enhanced = enhanced.view(B, C, N, K, L)
        return enhanced

    # ------------------------------------------------------------------
    # tier‑specific heuristics (kept from original implementation)
    # ------------------------------------------------------------------

    def tier_specific_enhancement(self, ch_coeffs: torch.Tensor, tier: int):
        """Apply additional frequency masking depending on TA‑CHNet tier.

        tier = 0 → organ‑level (boost global, damp high‑freq)
        tier = 1 → default (no extra op)
        tier = 2 → detail‑level (boost high‑freq)
        """
        enhanced = self.forward(ch_coeffs)

        if tier == 0:  # organ level – damp high‑freq
            for n_idx, n in enumerate(range(-self.max_n, self.max_n + 1)):
                if abs(n) > 2:
                    enhanced[:, :, n_idx] *= 0.8
            for k_idx, k in enumerate(range(1, self.max_k + 1)):
                if k > 3:
                    enhanced[:, :, :, k_idx] *= 0.8
        elif tier == 2:  # detail level – boost high‑freq
            for n_idx, n in enumerate(range(-self.max_n, self.max_n + 1)):
                if abs(n) >= 2:
                    enhanced[:, :, n_idx] *= 1.2
            for k_idx, k in enumerate(range(1, self.max_k + 1)):
                if k >= 3:
                    enhanced[:, :, :, k_idx] *= 1.2

        return enhanced
