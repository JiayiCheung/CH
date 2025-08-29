import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
	"""
	Dice损失函数
	"""
	
	def __init__(self, smooth=1.0):
		super(DiceLoss, self).__init__()
		self.smooth = smooth
	
	def forward(self, inputs, targets):
		# 展平输入和目标
		inputs_flat = inputs.view(-1)
		targets_flat = targets.view(-1)
		
		# 计算交集
		intersection = (inputs_flat * targets_flat).sum()
		
		# 计算Dice系数
		dice = (2. * intersection + self.smooth) / (inputs_flat.sum() + targets_flat.sum() + self.smooth)
		
		return 1 - dice


class WeightedDiceLoss(nn.Module):
	"""
	加权Dice损失函数
	根据类别频率进行加权
	"""
	
	def __init__(self, smooth=1.0):
		super(WeightedDiceLoss, self).__init__()
		self.smooth = smooth
	
	def forward(self, inputs, targets):
		# 计算类别权重
		weights = self._calculate_weights(targets)
		
		# 展平输入、目标和权重
		inputs_flat = inputs.view(-1)
		targets_flat = targets.view(-1)
		weights_flat = weights.view(-1)
		
		# 应用权重
		weighted_inputs = inputs_flat * weights_flat
		weighted_targets = targets_flat * weights_flat
		
		# 计算加权交集
		intersection = (weighted_inputs * weighted_targets).sum()
		
		# 计算加权Dice系数
		dice = (2. * intersection + self.smooth) / (
				weighted_inputs.sum() + weighted_targets.sum() + self.smooth)
		
		return 1 - dice
	
	def _calculate_weights(self, targets):
		"""
		计算类别权重
		反比于类别频率
		"""
		# 计算类别频率
		num_pixels = targets.numel()
		num_pos = targets.sum()
		num_neg = num_pixels - num_pos
		
		# 避免除零
		if num_pos == 0 or num_neg == 0:
			return torch.ones_like(targets)
		
		# 计算权重 - 前景权重高于背景
		pos_weight = num_pixels / (2 * num_pos)
		neg_weight = num_pixels / (2 * num_neg)
		
		# 创建权重掩码
		weights = torch.ones_like(targets)
		weights[targets > 0] = pos_weight
		weights[targets == 0] = neg_weight
		
		return weights


class FocalDiceLoss(nn.Module):
	"""
	Focal Dice损失
	增加对难分类样本的关注
	"""
	
	def __init__(self, gamma=1.0, smooth=1.0):
		super(FocalDiceLoss, self).__init__()
		self.gamma = gamma
		self.smooth = smooth
	
	def forward(self, inputs, targets):
		# 展平输入和目标
		inputs_flat = inputs.view(-1)
		targets_flat = targets.view(-1)
		
		# 计算Dice系数
		intersection = (inputs_flat * targets_flat).sum()
		dice = (2. * intersection + self.smooth) / (
				inputs_flat.sum() + targets_flat.sum() + self.smooth)
		
		# 应用Focal调制
		focal_factor = (1 - dice) ** self.gamma
		
		return focal_factor * (1 - dice)


class CombinedLoss(nn.Module):
	"""
	组合损失函数
	结合Dice损失和交叉熵损失
	"""
	
	def __init__(self, alpha=0.5, gamma=2.0, class_weights=None):
		super(CombinedLoss, self).__init__()
		self.alpha = alpha
		self.gamma = gamma
		self.class_weights = class_weights
		self.dice_loss = WeightedDiceLoss() if class_weights is not None else DiceLoss()
	
	def forward(self, inputs, targets):
		# Dice损失
		dice = self.dice_loss(inputs, targets)
		
		# Focal修改的交叉熵
		bce = F.binary_cross_entropy(inputs, targets, reduction='none')
		pt = torch.exp(-bce)  # pt = p if y=1, 1-p if y=0
		focal_factor = (1 - pt) ** self.gamma
		focal_ce = focal_factor * bce
		
		# 应用类别权重(如果提供)
		if self.class_weights is not None:
			weights = self._get_weights(targets)
			focal_ce = focal_ce * weights
		
		# 组合损失
		loss = self.alpha * dice + (1 - self.alpha) * focal_ce.mean()
		return loss
	
	def _get_weights(self, targets):
		"""
		获取每个样本的权重
		"""
		if self.class_weights is None:
			return torch.ones_like(targets)
		
		weights = torch.ones_like(targets)
		weights[targets > 0] = self.class_weights[1]
		weights[targets == 0] = self.class_weights[0]
		return weights


def calculate_class_weights(labels):
	"""
	计算类别权重
	反比于类别频率

	参数:
		labels: 分割标签

	返回:
		weights: 类别权重
	"""
	# 计算类别频率
	class_counts = torch.bincount(labels.flatten().long())
	total_samples = class_counts.sum()
	
	# 计算权重
	weights = total_samples / (len(class_counts) * class_counts)
	return weights