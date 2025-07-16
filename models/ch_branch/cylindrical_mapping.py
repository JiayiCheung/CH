import math
from functools import lru_cache
from typing import Tuple

import torch
import torch.nn.functional as F


from models.complex.functional import grid_sample as complex_grid_sample
from models.complex.utils import apply_to_complex, magnitude


class CylindricalMapping:
	"""
    Differentiable Cartesian ↔ Cylindrical mapping for 3‑D volumes.
    Notes
    -----
    * Supports **real or complex** tensors of shape (B,C,D,H,W).
    * All coordinates are normalised to **[-1,1]** for `grid_sample`.
    * Grid is cached per spatial shape on **first call** (use LRU).
    * Uses complex toolkit for automatic real/imaginary handling.
    """
	
	def __init__(self,
	             r_samples: int = 32,
	             theta_samples: int = 36,
	             z_samples: int = 32):
		self.r_samples = r_samples
		self.theta_samples = theta_samples
		self.z_samples = z_samples
		
		
		self._robust_center_complex = apply_to_complex(self._robust_center_real)
	
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
		
		r_norm = r  # [0,1]
		theta_norm = theta / (2 * math.pi)
		z_norm = (zz + 1) / 2  # [-1,1] → [0,1]
		
		grid = torch.stack([
			2 * z_norm - 1,
			2 * theta_norm - 1,
			2 * r_norm - 1,
		], dim=-1)  # (D,H,W,3)
		return grid.unsqueeze(0)  # (1,D,H,W,3)
	
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
		return grid.unsqueeze(0)  # (1,r,θ,z,3)
	
	# ------------------------------------------------------------------
	# r=0 singularity handling - 大幅精简版
	# ------------------------------------------------------------------
	
	def _robust_center_real(self, tensor: torch.Tensor, dim: int) -> torch.Tensor:
		"""
        计算实数张量的robust中心值
        注意：这个函数只处理实数，复数由apply_to_complex装饰器自动处理
        """
		# 使用截断均值：排除极端值后取平均
		sorted_values, _ = torch.sort(tensor, dim=dim)
		
		# 计算截断范围（去除最大和最小的25%）
		total_size = tensor.size(dim)
		start_idx = max(1, total_size // 4)
		end_idx = min(total_size - 1, 3 * total_size // 4)
		
		if start_idx >= end_idx:
			# 如果数据太少，直接用平均值
			return tensor.mean(dim=dim, keepdim=True)
		
		# 截断均值
		truncated = sorted_values.narrow(dim, start_idx, end_idx - start_idx)
		robust_mean = truncated.mean(dim=dim, keepdim=True)
		
		return robust_mean
	
	def _handle_r0_singularity_vessel_focused(self, cyl: torch.Tensor) -> torch.Tensor:
		"""
        精简版r=0奇异性处理 - 使用complex工具包

        关键简化：
        1. 不再手动处理复数的实部和虚部
        2. 使用apply_to_complex装饰器自动处理复数
        3. 代码大幅简化，更不容易出错

        参数:
            cyl: 柱坐标体积 [B, C, r, θ, z] (实数或复数)

        返回:
            修正后的柱坐标体积
        """
		B, C, R, T, Z = cyl.shape
		
		# 检查是否需要处理r=0
		if R == 0:
			return cyl
		
		# 对每个z切片单独处理r=0层
		# 关键：保持血管沿z轴的变化！
		for z_idx in range(Z):
			# 提取r=0层的当前z切片 [B, C, T]
			r0_z_slice = cyl[:, :, 0, :, z_idx]
			
			# ✨ 神奇之处：一行代码处理实数和复数！
			# apply_to_complex装饰器自动处理复数的实部和虚部
			center_value = self._robust_center_complex(r0_z_slice, dim=2)
			
			# 将计算得到的中心值广播到所有theta角度
			cyl[:, :, 0, :, z_idx] = center_value
		
		return cyl
	
	# ------------------------------------------------------------------
	# forward mappings
	# ------------------------------------------------------------------
	
	def cartesian_to_cylindrical(self, volume: torch.Tensor) -> torch.Tensor:
		"""Map (B,C,D,H,W) → (B,C,r,θ,z) with FIXED r=0 handling."""
		B, C, D, H, W = volume.shape
		grid = self._cylindrical_grid(D, H, W, volume.device)
		grid = grid.expand(B, -1, -1, -1, -1)  # (B,r,θ,z,3)
		
		# 使用complex工具包的grid_sample - 自动处理复数
		vol_perm = volume  # (B,C,D,H,W)
		cyl = complex_grid_sample(vol_perm, grid, mode='bilinear', align_corners=True)
		# output is (B,C,r,θ,z)
		
		# ✅ 精简的r=0处理 - 自动支持复数
		cyl = self._handle_r0_singularity_vessel_focused(cyl)
		
		return cyl
	
	def cylindrical_to_cartesian(self, cyl: torch.Tensor, out_shape: Tuple[int, int, int]) -> torch.Tensor:
		"""Map (B,C,r,θ,z) → (B,C,D,H,W) with given spatial size."""
		B, C = cyl.shape[:2]
		D, H, W = out_shape
		grid = self._cartesian_grid(D, H, W, cyl.device)
		grid = grid.expand(B, -1, -1, -1, -1)  # (B,D,H,W,3)
		
		# reshape cyl => (B,C,z,θ,r) to align with grid_sample's D,H,W order
		cyl_perm = cyl.permute(0, 1, 4, 3, 2)  # (B,C,z,θ,r)
		
		# 使用complex工具包的grid_sample - 自动处理复数
		cart = complex_grid_sample(cyl_perm, grid, mode='bilinear', align_corners=True)
		return cart