import torch
import torch.nn as nn
import torch.nn.functional as F


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
		
		# 多分类情况
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


class BoundaryFocalLoss(nn.Module):
	"""边界增强的Focal损失，对边界区域给予更高的权重"""
	
	def __init__(self, alpha=0.25, gamma=2.0, boundary_weight=5.0):
		"""
		初始化边界增强的Focal损失

		参数:
			alpha: 类别权重系数
			gamma: 聚焦参数
			boundary_weight: 边界区域的权重
		"""
		super().__init__()
		self.alpha = alpha
		self.gamma = gamma
		self.boundary_weight = boundary_weight
		self.focal_loss = FocalLoss(alpha, gamma, reduction='none')
	
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
			raise ValueError("Boundary computation requires binary mask")
		
		# 创建空的边界掩码
		boundary = torch.zeros_like(target)
		
		# 创建边界滤波器
		pad = kernel_size // 2
		
		# 对每个样本计算边界
		for i in range(target.shape[0]):
			# 对每个通道计算边界
			for c in range(target.shape[1]):
				# 膨胀原始掩码
				dilated = F.max_pool3d(
					target[i:i + 1, c:c + 1],
					kernel_size=kernel_size,
					stride=1,
					padding=pad
				)
				
				# 腐蚀原始掩码
				eroded = -F.max_pool3d(
					-target[i:i + 1, c:c + 1],
					kernel_size=kernel_size,
					stride=1,
					padding=pad
				)
				
				# 边界 = 膨胀 - 腐蚀
				boundary[i, c] = dilated - eroded
		
		return boundary
	
	def forward(self, pred, target):
		"""
		计算边界增强的Focal损失

		参数:
			pred: 预测值 [B, C, D, H, W]
			target: 目标值 [B, C, D, H, W] 或 [B, D, H, W]

		返回:
			边界增强的Focal损失值
		"""
		# 确保target具有与pred相同的维度
		if len(target.shape) < len(pred.shape):
			if pred.shape[1] == 1:
				# 二分类情况
				target = target.unsqueeze(1)
			else:
				# 多分类情况，转换为one-hot编码
				target = F.one_hot(target.long(), num_classes=pred.shape[1])
				target = target.permute(0, 4, 1, 2, 3).contiguous()
		
		# 计算普通Focal损失
		focal_loss = self.focal_loss(pred, target)
		
		# 计算边界掩码
		boundary_mask = self._compute_boundary_mask(target)
		
		# 应用边界权重
		weighted_loss = focal_loss * (1 + (self.boundary_weight - 1) * boundary_mask)
		
		# 返回平均损失
		return weighted_loss.mean()