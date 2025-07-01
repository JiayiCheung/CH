import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np


class EdgeEnhancement(nn.Module):
	"""预定义边缘增强模块"""
	
	def __init__(self, in_channels=1, out_channels=8):
		"""
		初始化边缘增强模块

		参数:
			in_channels: 输入通道数
			out_channels: 输出通道数
		"""
		super().__init__()
		
		# 预定义的3D边缘检测滤波器
		self.edge_kernels = self._create_edge_kernels()
		
		# 合并边缘特征的1×1×1卷积
		self.combine = nn.Conv3d(len(self.edge_kernels) * in_channels, out_channels, kernel_size=1)
		
		# 归一化和激活
		self.norm = nn.InstanceNorm3d(out_channels)
		self.activation = nn.ReLU(inplace=True)
	
	def _create_edge_kernels(self):
		"""
		创建预定义的3D边缘检测核

		返回:
			边缘检测核列表
		"""
		# Sobel核 (6个方向)
		sobel_kernels = []
		
		# X方向Sobel核
		sobel_x = np.zeros((3, 3, 3))
		sobel_x[:, :, 0] = np.array([[-1, -2, -1], [-2, -4, -2], [-1, -2, -1]])
		sobel_x[:, :, 2] = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]])
		sobel_kernels.append(sobel_x)
		
		# Y方向Sobel核
		sobel_y = np.zeros((3, 3, 3))
		sobel_y[:, 0, :] = np.array([[-1, -2, -1], [-2, -4, -2], [-1, -2, -1]])
		sobel_y[:, 2, :] = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]])
		sobel_kernels.append(sobel_y)
		
		# Z方向Sobel核
		sobel_z = np.zeros((3, 3, 3))
		sobel_z[0, :, :] = np.array([[-1, -2, -1], [-2, -4, -2], [-1, -2, -1]])
		sobel_z[2, :, :] = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]])
		sobel_kernels.append(sobel_z)
		
		# 拉普拉斯核
		laplacian = np.zeros((3, 3, 3))
		laplacian[1, 1, 1] = 26
		laplacian[1, 1, 0] = laplacian[1, 1, 2] = laplacian[1, 0, 1] = laplacian[1, 2, 1] = laplacian[0, 1, 1] = \
		laplacian[2, 1, 1] = -4
		laplacian[0, 0, 0] = laplacian[0, 0, 2] = laplacian[0, 2, 0] = laplacian[0, 2, 2] = laplacian[2, 0, 0] = \
		laplacian[2, 0, 2] = laplacian[2, 2, 0] = laplacian[2, 2, 2] = -1
		sobel_kernels.append(laplacian)
		
		# 将NumPy数组转换为PyTorch张量并规范化
		kernels = []
		for kernel in sobel_kernels:
			# 规范化以保持能量
			kernel = kernel / np.abs(kernel).sum()
			
			# 转换为PyTorch张量并增加批次和通道维度
			kernel_tensor = torch.from_numpy(kernel).float().unsqueeze(0).unsqueeze(0)
			kernels.append(nn.Parameter(kernel_tensor, requires_grad=False))
		
		return nn.ParameterList(kernels)
	
	def forward(self, x):
		"""
		应用边缘增强

		参数:
			x: 输入体积 [B, C, D, H, W]

		返回:
			边缘增强特征 [B, out_channels, D, H, W]
		"""
		B, C = x.shape[:2]
		
		# 应用边缘检测核
		edge_features = []
		
		for c in range(C):
			channel_features = []
			
			for kernel in self.edge_kernels:
				# 将核移动到与输入相同的设备
				kernel = kernel.to(x.device)
				
				# 应用3D卷积
				edge = F.conv3d(
					x[:, c:c + 1],
					kernel.expand(1, 1, 3, 3, 3),  # 扩展为(out_channels, in_channels/groups, D, H, W)
					padding=1
				)
				
				channel_features.append(edge)
			
			# 合并当前通道的所有边缘特征
			edge_features.extend(channel_features)
		
		# 合并所有特征
		edge_volume = torch.cat(edge_features, dim=1)
		
		# 应用1×1×1卷积、归一化和激活
		output = self.combine(edge_volume)
		output = self.norm(output)
		output = self.activation(output)
		
		return output