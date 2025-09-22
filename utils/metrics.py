import torch
import numpy as np
from scipy.ndimage import distance_transform_edt


def calculate_dice(pred, target, smooth=1e-5):
	"""
	计算Dice系数

	参数:
		pred: 预测分割掩码
		target: 目标分割掩码
		smooth: 平滑项，避免除零

	返回:
		dice: Dice系数
	"""
	# 展平预测和目标
	pred_flat = pred.view(-1)
	target_flat = target.view(-1)
	
	# 计算交集
	intersection = (pred_flat * target_flat).sum()
	
	# 计算Dice系数
	dice = (2. * intersection + smooth) / (pred_flat.sum() + target_flat.sum() + smooth)
	
	return dice


def calculate_iou(pred, target, smooth=1e-5):
	"""
	计算IoU(交并比)

	参数:
		pred: 预测分割掩码
		target: 目标分割掩码
		smooth: 平滑项，避免除零

	返回:
		iou: IoU系数
	"""
	# 展平预测和目标
	pred_flat = pred.view(-1)
	target_flat = target.view(-1)
	
	# 计算交集和并集
	intersection = (pred_flat * target_flat).sum()
	union = pred_flat.sum() + target_flat.sum() - intersection
	
	# 计算IoU
	iou = (intersection + smooth) / (union + smooth)
	
	return iou


def calculate_precision(pred, target):
	"""
	计算精确率

	参数:
		pred: 预测分割掩码
		target: 目标分割掩码

	返回:
		precision: 精确率
	"""
	# 展平预测和目标
	pred_flat = pred.view(-1)
	target_flat = target.view(-1)
	
	# 计算真阳性和假阳性
	true_positives = (pred_flat * target_flat).sum()
	false_positives = pred_flat.sum() - true_positives
	
	# 计算精确率
	precision = true_positives / (true_positives + false_positives + 1e-5)
	
	return precision


def calculate_recall(pred, target):
	"""
	计算召回率

	参数:
		pred: 预测分割掩码
		target: 目标分割掩码

	返回:
		recall: 召回率
	"""
	# 展平预测和目标
	pred_flat = pred.view(-1)
	target_flat = target.view(-1)
	
	# 计算真阳性和假阴性
	true_positives = (pred_flat * target_flat).sum()
	false_negatives = target_flat.sum() - true_positives
	
	# 计算召回率
	recall = true_positives / (true_positives + false_negatives + 1e-5)
	
	return recall


def calculate_hausdorff_distance(pred, target, percentile=95):
	"""
	计算Hausdorff距离

	参数:
		pred: 预测分割掩码(numpy数组或Torch tensor)，形状需为 (D, H, W)
		target: 目标分割掩码(numpy数组或Torch tensor)，形状需为 (D, H, W)
		percentile: 距离百分位数(避免离群点影响)

	返回:
		hd: Hausdorff距离
	"""
	# --- Step 1: 统一格式 ---
	if isinstance(pred, torch.Tensor):
		pred = pred.squeeze().cpu().numpy()  # 确保为 (D, H, W)
	if isinstance(target, torch.Tensor):
		target = target.squeeze().cpu().numpy()  # 确保为 (D, H, W)
	
	# --- Step 2: 二值化 ---
	pred = (pred > 0.5).astype(np.bool_)
	target = (target > 0.5).astype(np.bool_)
	
	# --- Step 3: 排除全0情况 ---
	if pred.sum() == 0 or target.sum() == 0:
		return float('inf')
	
	# --- Step 4: 距离变换 ---
	pred_to_target = distance_transform_edt(~target)[pred]
	target_to_pred = distance_transform_edt(~pred)[target]
	
	# --- Step 5: 取百分位距离 ---
	hd = max(
		np.percentile(pred_to_target, percentile),
		np.percentile(target_to_pred, percentile)
	)
	
	return hd


def calculate_assd(pred, target):
	"""
	计算平均对称表面距离(ASSD)

	参数:
		pred: 预测分割掩码(numpy数组)
		target: 目标分割掩码(numpy数组)

	返回:
		assd: 平均对称表面距离
	"""
	# 转换为NumPy数组
	if isinstance(pred, torch.Tensor):
		pred = pred.cpu().numpy()
	if isinstance(target, torch.Tensor):
		target = target.cpu().numpy()
	
	# 确保二值掩码
	pred = (pred > 0.5).astype(np.bool_)
	target = (target > 0.5).astype(np.bool_)
	
	# 如果预测或目标全为0或全为1，返回无穷大
	if pred.sum() == 0 or target.sum() == 0:
		return float('inf')
	
	# 计算预测到目标的距离
	pred_to_target = distance_transform_edt(~target)
	pred_to_target = pred_to_target[pred]
	
	# 计算目标到预测的距离
	target_to_pred = distance_transform_edt(~pred)
	target_to_pred = target_to_pred[target]
	
	# 计算平均对称表面距离
	assd = (np.mean(pred_to_target) + np.mean(target_to_pred)) / 2
	
	return assd


