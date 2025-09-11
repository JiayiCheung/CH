# val.py
import os
import csv
import time
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from typing import Tuple, List, Union, Optional, Callable
from functools import lru_cache


@lru_cache(maxsize=2)
def compute_gaussian(tile_size: Tuple[int, ...], sigma_scale: float = 1. / 8,
                     value_scaling_factor: float = 1, dtype=torch.float32,
                     device=torch.device('cuda', 0)) -> torch.Tensor:
	"""
	计算用于滑动窗口推理的高斯重要性图（纯PyTorch实现）。

	参数:
		tile_size: 窗口/补丁的大小
		sigma_scale: 高斯sigma相对于窗口大小的比例
		value_scaling_factor: 高斯值的缩放因子
		dtype: 张量数据类型
		device: 张量设备

	返回:
		高斯重要性图张量
	"""
	# 创建坐标网格
	dim = len(tile_size)
	coords = []
	for i, size in enumerate(tile_size):
		# 创建从-1到1的坐标
		t = torch.linspace(-1, 1, size, device=device)
		# 为其他维度添加维度
		for j in range(dim):
			if j != i:
				t = t.unsqueeze(j)
		coords.append(t)
	
	# 计算高斯分布
	gaussian = torch.ones(tile_size, device=device)
	for i, (coord, size) in enumerate(zip(coords, tile_size)):
		sigma = sigma_scale * size
		gaussian = gaussian * torch.exp(-(coord ** 2) / (2 * sigma ** 2))
	
	# 归一化并应用缩放因子
	gaussian = gaussian / torch.max(gaussian) * value_scaling_factor
	gaussian = gaussian.to(dtype)
	
	# 确保没有零值，避免NaN
	min_non_zero = torch.min(gaussian[gaussian > 0])
	gaussian[gaussian == 0] = min_non_zero
	
	return gaussian


def compute_steps_for_sliding_window(image_size: Tuple[int, ...], tile_size: Tuple[int, ...],
                                     tile_step_size: float) -> List[List[int]]:
	"""
	计算滑动窗口推理的步长。

	参数:
		image_size: 输入图像的大小
		tile_size: 滑动窗口的大小
		tile_step_size: 步长与窗口大小的比例

	返回:
		每个维度的步长位置列表
	"""
	assert [i >= j for i, j in zip(image_size, tile_size)], "图像大小必须大于或等于补丁大小"
	assert 0 < tile_step_size <= 1, '步长必须大于0且小于或等于1'
	
	target_step_sizes_in_voxels = [i * tile_step_size for i in tile_size]
	num_steps = [int(np.ceil((i - k) / j)) + 1 for i, j, k in zip(image_size, target_step_sizes_in_voxels, tile_size)]
	
	steps = []
	for dim in range(len(tile_size)):
		max_step_value = image_size[dim] - tile_size[dim]
		if num_steps[dim] > 1:
			actual_step_size = max_step_value / (num_steps[dim] - 1)
		else:
			actual_step_size = 99999999999  # 不重要，因为在0处只有一个步长
		steps_here = [int(np.round(actual_step_size * i)) for i in range(num_steps[dim])]
		steps.append(steps_here)
	
	return steps


def sliding_window_inference(model, image, window_size, overlap=0.5):
	"""
	在整个体积上进行滑动窗口推理。

	参数:
		model: 分割模型
		image: 输入图像张量 [B, C, D, H, W]
		window_size: 滑动窗口的大小
		overlap: 窗口间的重叠量(0-1)

	返回:
		完整的预测张量
	"""
	device = next(model.parameters()).device
	image_shape = image.shape[2:]  # D, H, W
	batch_size = image.shape[0]
	
	# 初始化输出张量和计数掩码用于平均
	output_shape = [batch_size, 1] + list(image_shape)
	output = torch.zeros(output_shape, dtype=torch.float32, device=device)
	count_map = torch.zeros(output_shape, dtype=torch.float32, device=device)
	
	# 计算高斯重要性图
	gaussian_map = compute_gaussian(window_size, dtype=torch.float32, device=device)
	
	# 计算滑动窗口步长
	steps = compute_steps_for_sliding_window(image_shape, window_size, 1 - overlap)
	
	# 获取所有步长的组合
	all_slices = []
	for d in steps[0]:
		d_slice = slice(d, d + window_size[0])
		for h in steps[1]:
			h_slice = slice(h, h + window_size[1])
			for w in steps[2]:
				w_slice = slice(w, w + window_size[2])
				all_slices.append((d_slice, h_slice, w_slice))
	
	# 对每个窗口进行推理
	model.eval()
	with torch.no_grad():
		for d_slice, h_slice, w_slice in tqdm(all_slices, desc="滑动窗口推理", disable=True):
			# 提取窗口
			window = image[:, :, d_slice, h_slice, w_slice]
			
			# 在窗口上运行模型
			window_pred = model(window)
			
			# 应用sigmoid
			window_pred = torch.sigmoid(window_pred)
			
			# 应用高斯加权
			window_pred = window_pred * gaussian_map
			
			# 添加到输出和计数图
			output[:, :, d_slice, h_slice, w_slice] += window_pred
			count_map[:, :, d_slice, h_slice, w_slice] += gaussian_map
	
	# 通过除以计数来平均预测
	output = output / (count_map + 1e-8)
	
	return output


def calculate_dice(pred, target, threshold=0.5):
	"""
	计算Dice系数。

	参数:
		pred: 预测概率张量 (在0-1之间)
		target: 二值真实标签张量
		threshold: 二值化阈值

	返回:
		Dice系数(浮点数)
	"""
	# 在这里应用阈值
	pred = (pred > threshold).float()
	
	# 展平张量
	pred_flat = pred.view(-1)
	target_flat = target.view(-1)
	
	# 计算交集和并集
	intersection = torch.sum(pred_flat * target_flat)
	union = torch.sum(pred_flat) + torch.sum(target_flat)
	
	# 计算Dice
	dice = (2.0 * intersection + 1e-8) / (union + 1e-8)
	
	return dice.item()


def log_metric(logger, tag, value, step):
	"""
	兼容不同类型的logger记录指标

	参数:
		logger: 日志记录器（TensorBoardLogger或SummaryWriter）
		tag: 指标标签
		value: 指标值
		step: 当前步数
	"""
	if logger is None:
		return
	
	# 检查logger类型并调用相应的方法
	if hasattr(logger, 'log_scalar'):
		# TensorBoardLogger
		logger.log_scalar(tag, value, step)
	elif hasattr(logger, 'add_scalar'):
		# SummaryWriter
		logger.add_scalar(tag, value, step)
	else:
		print(f"警告: 无法识别的logger类型，无法记录 {tag}={value}")


def validate(model, val_loader, local_rank, logger, step, log_dir, window_size=None):
	"""
	在验证集上运行验证。

	参数:
		model: 分割模型(DDP包装)
		val_loader: 验证数据加载器
		local_rank: 当前GPU ID
		logger: TensorBoard记录器
		step: 当前全局步数
		log_dir: 保存日志的目录
		window_size: 滑动窗口的大小，如果为None则自动根据图像大小确定

	返回:
		平均Dice系数
	"""
	# 只在rank 0上执行验证
	if local_rank != 0:
		return None
	
	model.eval()
	start_time = time.time()
	
	# 创建验证日志目录
	validation_dir = os.path.join(log_dir, 'validation')
	os.makedirs(validation_dir, exist_ok=True)
	
	# 定义CSV文件路径
	summary_csv_path = os.path.join(validation_dir, 'validation_summary.csv')
	cases_csv_path = os.path.join(validation_dir, 'validation_cases.csv')
	
	# 检查并创建summary CSV文件（如果不存在）
	summary_file_exists = os.path.exists(summary_csv_path)
	if not summary_file_exists:
		with open(summary_csv_path, 'w', newline='') as csvfile:
			csv_writer = csv.writer(csvfile)
			csv_writer.writerow(['step', 'avg_dice', 'num_cases', 'time_sec'])
	
	# 检查并创建cases CSV文件（如果不存在）
	cases_file_exists = os.path.exists(cases_csv_path)
	if not cases_file_exists:
		with open(cases_csv_path, 'w', newline='') as csvfile:
			csv_writer = csv.writer(csvfile)
			csv_writer.writerow(['step', 'case_id', 'dice'])
	
	# 收集所有案例的Dice分数
	all_dice_scores = []
	
	# 处理每个验证案例
	for batch in tqdm(val_loader, desc="验证中"):
		images, labels, case_ids = batch
		
		# 将数据移至设备
		images = images.cuda(local_rank, non_blocking=True)
		labels = labels.cuda(local_rank, non_blocking=True)
		
		# 确保batch size为1
		assert images.shape[0] == 1, "验证时batch size必须为1"
		
		# 获取图像形状
		image_shape = images.shape[2:]  # D, H, W
		
		# 如果没有指定window_size，则根据图像大小自动确定
		if window_size is None:
			window_size = [min(96, d) for d in image_shape]
		
		# 运行滑动窗口推理
		predictions = sliding_window_inference(model, images, window_size, overlap=0.5)
		
		# 计算此案例的Dice (阈值化在calculate_dice内部完成)
		dice_score = calculate_dice(predictions, labels)
		all_dice_scores.append(dice_score)
		
		# 记录到cases CSV
		with open(cases_csv_path, 'a', newline='') as csvfile:
			csv_writer = csv.writer(csvfile)
			csv_writer.writerow([step, case_ids[0], dice_score])
	
	# 计算平均Dice和验证时间
	avg_dice = np.mean(all_dice_scores) if all_dice_scores else 0.0
	validation_time = time.time() - start_time
	num_cases = len(all_dice_scores)
	
	# 记录到summary CSV
	with open(summary_csv_path, 'a', newline='') as csvfile:
		csv_writer = csv.writer(csvfile)
		csv_writer.writerow([step, avg_dice, num_cases, validation_time])
	
	# 记录到TensorBoard
	if logger is not None:
		if hasattr(logger, 'log_scalar'):
			# TensorBoardLogger
			logger.log_scalar('val/dice', avg_dice, step)
			logger.log_scalar('val/time', validation_time, step)
		elif hasattr(logger, 'add_scalar'):
			# SummaryWriter
			logger.add_scalar('val/dice', avg_dice, step)
			logger.add_scalar('val/time', validation_time, step)
	
	print(f"在步骤{step}完成验证。平均Dice: {avg_dice:.4f}，共{num_cases}个病例，耗时: {validation_time:.2f}秒")
	
	# 返回平均Dice分数
	return avg_dice


	

