import torch
import torch.nn as nn

from .dice_loss import DiceLoss, GeneralizedDiceLoss
from .focal_loss import FocalLoss, AdaptiveBoundaryFocalLoss  # 修改这里导入新类


class CombinedLoss(nn.Module):
	"""组合损失，结合多个损失函数"""
	
	def __init__(self, losses=None, weights=None):
		"""
		初始化组合损失
 
		参数:
		   losses: 损失函数列表
		   weights: 每个损失函数的权重
		"""
		super().__init__()
		
		# 如果没有提供损失函数，使用默认组合
		if losses is None:
			self.losses = [
				DiceLoss(squared=True),
				FocalLoss(alpha=0.25, gamma=2.0)
			]
			self.weights = [0.7, 0.3]
		else:
			self.losses = losses
			self.weights = weights if weights is not None else [1.0 / len(losses)] * len(losses)
	
	def forward(self, pred, target):
		"""
		计算组合损失
 
		参数:
		   pred: 预测值
		   target: 目标值
 
		返回:
		   组合损失值
		"""
		# 计算每个损失函数的加权损失
		total_loss = 0.0
		loss_values = {}
		
		for i, (loss_fn, weight) in enumerate(zip(self.losses, self.weights)):
			loss_name = loss_fn.__class__.__name__
			loss_value = loss_fn(pred, target)
			weighted_loss = weight * loss_value
			
			total_loss += weighted_loss
			loss_values[loss_name] = loss_value.item()
		
		# 存储各损失函数的值，便于日志记录
		self.loss_values = loss_values
		
		return total_loss


