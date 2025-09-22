from __future__ import annotations

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["EdgeEnhancement"]


class EdgeEnhancement(nn.Module):
	"""3‑D Sobel / Laplacian edge‑enhancement block (vectorised)."""
	
	def __init__(self, in_channels: int = 1, out_channels: int = 4, device=None, dtype=None):
		super().__init__()
		self.in_channels = in_channels
		self.out_channels = out_channels
		
		# 创建卷积核
		self._make_kernels()
		
		# 直接初始化组件
		K = self.kernels.shape[0]  # 通常是4
		self.combine = nn.Conv3d(K * in_channels, out_channels, 1, bias=False)
		
		# 创建归一化层
		groups = max(4, out_channels // 4)
		self.norm = nn.GroupNorm(groups, out_channels, affine=True)
		
		# 激活函数
		self.act = nn.ReLU(inplace=True)
		
		# 如果提供了设备和数据类型，则移动模块
		if device is not None or dtype is not None:
			self.to(device=device, dtype=dtype)
	
	# ------------------------------------------------------------
	def _make_kernels(self):
		"""Create 4 fixed 3×3×3 kernels via NumPy (same as original impl)."""
		sobel_kernels = []
		# X‑direction Sobel
		kx = np.zeros((3, 3, 3), dtype=np.float32)
		kx[:, :, 0] = np.array([[-1, -2, -1], [-2, -4, -2], [-1, -2, -1]], dtype=np.float32)
		kx[:, :, 2] = -kx[:, :, 0]
		sobel_kernels.append(kx)
		# Y‑direction Sobel
		ky = np.zeros((3, 3, 3), dtype=np.float32)
		ky[:, 0, :] = kx[:, :, 0]
		ky[:, 2, :] = -kx[:, :, 0]
		sobel_kernels.append(ky)
		# Z‑direction Sobel
		kz = np.zeros((3, 3, 3), dtype=np.float32)
		kz[0, :, :] = kx[:, :, 0]
		kz[2, :, :] = -kx[:, :, 0]
		sobel_kernels.append(kz)
		# 3‑D Laplacian
		lap = np.full((3, 3, 3), -1, dtype=np.float32)
		lap[1, 1, 1] = 26.
		sobel_kernels.append(lap)
		# Normalize & stack
		normed = [k / np.abs(k).sum() for k in sobel_kernels]
		kernel_stack = torch.from_numpy(np.stack(normed)).unsqueeze(1)  # [K,1,3,3,3]
		self.kernels = nn.Parameter(kernel_stack.clone(), requires_grad=True)
	
	# ------------------------------------------------------------
	def forward(self, x: torch.Tensor) -> torch.Tensor:
		B, C, D, H, W = x.shape
		K = self.kernels.shape[0]
		
		# 确保输入通道数匹配
		assert C == self.in_channels, f"输入通道数 {C} 与初始化通道数 {self.in_channels} 不匹配"
		
		# 扩展卷积核以应用于所有通道
		weight = self.kernels.repeat_interleave(C, dim=0)
		
		# 应用卷积计算边缘特征
		edge = F.conv3d(
			x,
			weight,
			bias=None,
			padding=1,
			groups=1,  # 标准卷积
		)
		
		# 应用组合卷积和归一化
		out = self.combine(edge)
		out = self.norm(out)
		return self.act(out)