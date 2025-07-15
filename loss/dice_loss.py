import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
	"""Dice损失函数，适用于不平衡分割任务"""
	
	def __init__(self, smooth=1.0, squared=False):
		"""
		初始化Dice损失

		参数:
			smooth: 平滑项，防止分母为0
			squared: 是否使用平方版本
		"""
		super().__init__()
		self.smooth = smooth
		self.squared = squared
	
	def forward(self, pred, target):
		"""
		计算Dice损失

		参数:
			pred: 预测值 [B, C, D, H, W]
			target: 目标值 [B, C, D, H, W] 或 [B, D, H, W]

		返回:
			Dice损失值
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
		
		# 展平预测和目标
		pred_flat = pred.reshape(pred.shape[0], pred.shape[1], -1)
		target_flat = target.reshape(target.shape[0], target.shape[1], -1)
		
		# 可选: 应用平方
		if self.squared:
			pred_flat = pred_flat ** 2
			target_flat = target_flat ** 2
		
		# 计算交集和并集
		intersection = torch.sum(pred_flat * target_flat, dim=2)
		union = torch.sum(pred_flat, dim=2) + torch.sum(target_flat, dim=2)
		
		# 计算Dice系数
		dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
		
		# 每个通道的平均Dice损失
		dice_loss = 1.0 - dice.mean(dim=1)
		
		# 批次平均
		return dice_loss.mean()


class GeneralizedDiceLoss(nn.Module):
	"""广义Dice损失，对不平衡类别提供更好的性能"""
	
	def __init__(self, smooth=1.0):
		"""
		初始化广义Dice损失

		参数:
			smooth: 平滑项，防止分母为0
		"""
		super().__init__()
		self.smooth = smooth
	
	def forward(self, pred, target):
		"""
		计算广义Dice损失

		参数:
			pred: 预测值 [B, C, D, H, W]
			target: 目标值 [B, C, D, H, W] 或 [B, D, H, W]

		返回:
			广义Dice损失值
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
		
		# 展平预测和目标
		pred_flat = pred.reshape(pred.shape[0], pred.shape[1], -1)
		target_flat = target.reshape(target.shape[0], target.shape[1], -1)
		
		# 计算类别权重 (基于类别频率的倒数)
		weight = 1.0 / (torch.sum(target_flat, dim=2) ** 2 + self.smooth)
		
		# 计算加权交集和并集
		intersection = torch.sum(pred_flat * target_flat, dim=2)
		weighted_intersection = torch.sum(weight * intersection, dim=1)
		
		union = torch.sum(pred_flat, dim=2) + torch.sum(target_flat, dim=2)
		weighted_union = torch.sum(weight * union, dim=1)
		
		# 计算广义Dice系数
		dice = (2.0 * weighted_intersection + self.smooth) / (weighted_union + self.smooth)
		
		# Dice损失
		dice_loss = 1.0 - dice
		
		# 批次平均
		return dice_loss.mean()