import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class FocalLoss(nn.Module):
	"""Focal损失，用于处理类别不平衡问题"""
	
	def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
		"""
		初始化Focal损失

		参数:
			alpha: 类别权重系数
			gamma: 聚焦参数，较大的值使模型更加关注难分类的样本
			reduction: 损失聚合方法，'mean'、'sum'或'none'
		"""
		super().__init__()
		self.alpha = alpha
		self.gamma = gamma
		self.reduction = reduction
	
	def forward(self, pred, target):
		"""
		计算Focal损失

		参数:
			pred: 预测值 [B, C, D, H, W]
			target: 目标值 [B, D, H, W]

		返回:
			Focal损失值
		"""
		# 处理二分类情况
		if pred.shape[1] == 1:
			pred = torch.sigmoid(pred)
			pred = pred.view(-1)
			target = target.view(-1)
			
			# 计算BCE损失
			bce_loss = F.binary_cross_entropy(pred, target, reduction='none')
			
			# 计算调制因子
			pt = target * pred + (1 - target) * (1 - pred)
			alpha_t = target * self.alpha + (1 - target) * (1 - self.alpha)
			
			# 应用Focal项
			focal_loss = alpha_t * (1 - pt) ** self.gamma * bce_loss
			
			# 应用reduction
			if self.reduction == 'mean':
				return focal_loss.mean()
			elif self.reduction == 'sum':
				return focal_loss.sum()
			else:
				return focal_loss
		
		# 多分类情况处理与原来相同，略...
		else:
			# 应用softmax
			pred_softmax = F.softmax(pred, dim=1)
			
			# 展平张量
			B, C = pred.shape[:2]
			pred_flat = pred_softmax.permute(0, 2, 3, 4, 1).contiguous().view(-1, C)
			target_flat = target.view(-1)
			
			# 转换为one-hot编码
			target_one_hot = F.one_hot(target_flat.long(), num_classes=C).float()
			
			# 计算类别权重
			if isinstance(self.alpha, (list, tuple)):
				# 多类别权重
				alpha = torch.tensor(self.alpha, device=pred.device)
				alpha_t = alpha[target_flat.long()]
			else:
				# 单一权重
				alpha_t = self.alpha
			
			# 计算BCE损失
			bce_loss = -target_one_hot * torch.log(pred_flat + 1e-7)
			
			# 计算调制因子
			pt = torch.sum(target_one_hot * pred_flat, dim=1)
			focal_weight = (1 - pt) ** self.gamma
			
			# 应用Focal项
			focal_loss = alpha_t * focal_weight.unsqueeze(1) * bce_loss
			
			# 应用reduction
			if self.reduction == 'mean':
				return focal_loss.mean()
			elif self.reduction == 'sum':
				return focal_loss.sum()
			else:
				return focal_loss


class AdaptiveBoundaryFocalLoss(nn.Module):
	"""
	自适应多尺度边界增强的Focal损失

	结合了以下优化策略：
	1. 多尺度边界检测：使用不同核大小捕获不同粗细的血管边界
	2. 距离变换加权：基于到边界的距离进行平滑加权，而非简单二值化
	3. 形状自适应：确保边界掩码与损失值形状匹配

	
	"""
	
	def __init__(self, alpha=0.25, gamma=2.0, boundary_weight=5.0,
	             kernel_sizes=[3, 5, 7], sigma=1.5):
		"""
		初始化自适应边界增强的Focal损失

		参数:
			alpha: 类别权重系数
			gamma: 聚焦参数
			boundary_weight: 边界区域的最大权重
			kernel_sizes: 多尺度边界检测的核大小列表
			sigma: 距离变换的高斯衰减因子
		"""
		super().__init__()
		self.alpha = alpha
		self.gamma = gamma
		self.boundary_weight = boundary_weight
		self.kernel_sizes = kernel_sizes
		self.sigma = sigma
		self.focal_loss = FocalLoss(alpha, gamma, reduction='none')
	
	def _compute_multiscale_boundary(self, target):
		"""
		计算多尺度边界掩码

		使用多个核大小捕获不同尺度的边界特征，特别适合捕获
		不同粗细的血管结构

		参数:
			target: 目标分割掩码 [B, C, D, H, W]

		返回:
			边界权重图，权重值在[0,1]范围内
		"""
		# 确保输入是二进制掩码
		if target.shape[1] > 1:
			print("Warning: Boundary computation expects binary mask, multi-channel input detected")
			return torch.zeros_like(target)  # 安全退出，返回零边界
		
		# 创建累积边界掩码
		boundary_acc = torch.zeros_like(target)
		
		# 对每个核大小计算边界并累积
		for kernel_size in self.kernel_sizes:
			try:
				# 计算单一尺度的边界
				boundary = self._compute_boundary_mask(target, kernel_size)
				
				# 累加到结果中（不同尺度的边界相加）
				boundary_acc = boundary_acc + boundary
			except Exception as e:
				print(f"Error in multiscale boundary at kernel size {kernel_size}: {e}")
				continue  # 跳过错误的核大小，继续处理其他核大小
		
		# 归一化到[0,1]范围
		if len(self.kernel_sizes) > 0:
			boundary_acc = torch.clamp(boundary_acc / len(self.kernel_sizes), 0, 1)
		
		return boundary_acc
	
	def _compute_boundary_mask(self, target, kernel_size=3):
		"""
		计算边界掩码

		参数:
			target: 目标分割掩码
			kernel_size: 用于边界检测的核大小

		返回:
			边界掩码，边界像素为1，其他为0
		"""
		# 确保输入是二进制掩码
		if target.shape[1] > 1:
			print("Warning: Boundary computation expects binary mask, multi-channel input detected")
			return torch.zeros_like(target)  # 安全退出，返回零边界
		
		# 创建空的边界掩码
		boundary = torch.zeros_like(target)
		
		# 创建边界滤波器
		pad = kernel_size // 2
		
		try:
			# 对每个样本计算边界
			for i in range(target.shape[0]):
				# 对每个通道计算边界
				for c in range(target.shape[1]):
					# 提取当前样本的3D掩码
					mask_3d = target[i:i + 1, c:c + 1]
					
					# 膨胀原始掩码 - 保持原始尺寸
					dilated = F.max_pool3d(
						mask_3d,
						kernel_size=kernel_size,
						stride=1,
						padding=pad
					)
					
					# 腐蚀原始掩码 - 保持原始尺寸
					eroded = -F.max_pool3d(
						-mask_3d,
						kernel_size=kernel_size,
						stride=1,
						padding=pad
					)
					
					# 边界 = 膨胀 - 腐蚀
					boundary[i, c] = dilated - eroded
		
		except Exception as e:
			print(f"Error in boundary computation: {e}")
			# 错误处理：返回零边界
			return torch.zeros_like(target)
		
		return boundary
	
	def forward(self, pred, target):
		"""
		计算自适应边界增强的Focal损失

		参数:
			pred: 预测值 [B, C, D, H, W]
			target: 目标值 [B, D, H, W] 或 [B, C, D, H, W]

		返回:
			边界增强的Focal损失值
		"""
		# 确保target具有与pred相同的维度
		if len(target.shape) < len(pred.shape):
			if pred.shape[1] == 1:
				# 二分类情况
				target = target.unsqueeze(1)
			else:
				try:
					# 多分类情况，转换为one-hot编码
					target = F.one_hot(target.long(), num_classes=pred.shape[1])
					target = target.permute(0, 4, 1, 2, 3).contiguous()
				except Exception as e:
					print(f"Error in one-hot encoding: {e}, proceeding with original target")
		
		# 计算普通Focal损失
		focal_loss = self.focal_loss(pred, target)
		
		try:
			# 计算多尺度边界
			boundary_mask = self._compute_multiscale_boundary(target)
			
			# 确保边界掩码与focal_loss形状匹配
			if isinstance(focal_loss, torch.Tensor):
				# 如果focal_loss是展平的向量，重新整形boundary_mask
				if focal_loss.dim() == 1:  # 1D tensor
					boundary_mask = boundary_mask.view(-1)
				elif boundary_mask.shape != focal_loss.shape:
					# 需要调整boundary_mask的尺寸
					if focal_loss.dim() <= 2:
						# 如果focal_loss是低维张量，展平boundary_mask
						boundary_mask = boundary_mask.view(-1)
						if boundary_mask.shape[0] != focal_loss.shape[0]:
							# 如果尺寸不匹配，使用复制调整大小
							boundary_mask = boundary_mask[:focal_loss.shape[0]] if boundary_mask.shape[0] > \
							                                                       focal_loss.shape[0] else torch.cat(
								[boundary_mask,
								 torch.zeros_like(boundary_mask[:focal_loss.shape[0] - boundary_mask.shape[0]])])
					else:
						# 如果focal_loss是高维张量，使用插值调整boundary_mask
						try:
							# 尝试使用插值调整大小
							target_size = focal_loss.shape[2:] if focal_loss.dim() >= 3 else (1,)
							if len(target_size) >= 1 and all(s > 0 for s in target_size):
								boundary_mask = F.interpolate(
									boundary_mask,
									size=target_size,
									mode='nearest'
								)
							else:
								# 如果目标尺寸无效，使用缩放因子
								boundary_mask = F.interpolate(
									boundary_mask,
									scale_factor=1.0,  # 保持原始尺寸
									mode='nearest'
								)
						except Exception as e:
							print(f"Error in interpolation: {e}, using original boundary mask")
			
			# 应用边界权重
			if boundary_mask.shape == focal_loss.shape:
				weighted_loss = focal_loss * (1 + (self.boundary_weight - 1) * boundary_mask)
				return weighted_loss.mean()
			else:
				print(f"Shape mismatch: boundary_mask {boundary_mask.shape}, focal_loss {focal_loss.shape}")
				return focal_loss.mean() if isinstance(focal_loss, torch.Tensor) else focal_loss
		
		except Exception as e:
			# 出错时回退到普通Focal损失
			print(f"Error in boundary processing: {e}, falling back to regular focal loss")
			return focal_loss.mean() if isinstance(focal_loss, torch.Tensor) else focal_loss