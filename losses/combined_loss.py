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
			self.weights = [0.5, 0.5]
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


class VesselSegmentationLoss(nn.Module):
	"""专为血管分割设计的损失函数"""
	
	def __init__(self, num_classes=2, vessel_weight=10.0, tumor_weight=15.0, use_boundary=True):
		"""
		初始化血管分割损失
 
		参数:
		   num_classes: 分类数量 (1 = 二分类, >1 = 多分类)
		   vessel_weight: 血管类别的权重
		   tumor_weight: 肿瘤类别的权重 (如果num_classes > 2)
		   use_boundary: 是否使用边界增强损失
		"""
		super().__init__()
		
		self.num_classes = num_classes
		self.vessel_weight = vessel_weight
		self.tumor_weight = tumor_weight
		self.use_boundary = use_boundary
		
		# 设置损失函数组合
		if num_classes == 1:  # 二分类
			dice_loss = DiceLoss(squared=True)
			focal_loss = FocalLoss(alpha=0.25, gamma=2.0)
			
			if use_boundary:
				# 使用新的自适应边界损失
				boundary_loss = AdaptiveBoundaryFocalLoss(
					alpha=0.25,
					gamma=2.0,
					boundary_weight=3.0,
					kernel_sizes=[3, 5]  # 使用多尺度边界检测
				)
				self.combined_loss = CombinedLoss(
					[dice_loss, focal_loss, boundary_loss],
					[0.4, 0.3, 0.3]
				)
			else:
				self.combined_loss = CombinedLoss(
					[dice_loss, focal_loss],
					[0.5, 0.5]
				)
		else:  # 多分类
			# 使用广义Dice损失处理多类别
			dice_loss = GeneralizedDiceLoss()
			
			# 对于多类别，alpha需要是类别权重列表
			alphas = [0.1]  # 背景权重
			alphas.append(vessel_weight / (vessel_weight + tumor_weight + 0.1))  # 血管权重
			
			if num_classes > 2:
				alphas.append(tumor_weight / (vessel_weight + tumor_weight + 0.1))  # 肿瘤权重
			
			# 填充其他类别权重 (如果有)
			while len(alphas) < num_classes:
				alphas.append(0.1)
			
			focal_loss = FocalLoss(alpha=alphas, gamma=2.0)
			
			if use_boundary:
				# 多类别情况下也使用自适应边界损失
				boundary_loss = AdaptiveBoundaryFocalLoss(
					alpha=0.25,
					gamma=2.0,
					boundary_weight=3.0,
					kernel_sizes=[3, 5]
				)
				self.combined_loss = CombinedLoss(
					[dice_loss, focal_loss, boundary_loss],
					[0.4, 0.3, 0.3]
				)
			else:
				self.combined_loss = CombinedLoss(
					[dice_loss, focal_loss],
					[0.5, 0.5]
				)
	
	def forward(self, pred, target):
		"""
		计算血管分割损失
 
		参数:
		   pred: 预测值 [B, C, D, H, W]
		   target: 目标值 [B, D, H, W]
 
		返回:
		   损失值
		"""
		return self.combined_loss(pred, target)
	
	def get_loss_values(self):
		"""获取各损失函数的值"""
		return self.combined_loss.loss_values