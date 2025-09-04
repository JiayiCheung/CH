# visualization.py
import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from typing import Union, Tuple, Optional


def save_patch_visualization(
		inputs: torch.Tensor,
		predictions: torch.Tensor,
		labels: torch.Tensor,
		epoch: int,
		batch_idx: int,
		output_dir: str = 'visualizations',
		sample_idx: int = 0,
		include_error_map: bool = True
) -> None:
	"""
	保存3D补丁的可视化结果，包括输入、预测和标签的中心切片。

	Args:
		inputs: 输入补丁，形状为[B,C,D,H,W]或[B,D,H,W]
		predictions: 预测分割，形状为[B,C,D,H,W]或[B,D,H,W]
		labels: 真实标签，形状为[B,C,D,H,W]或[B,D,H,W]
		epoch: 当前训练轮数
		batch_idx: 批次索引
		output_dir: 输出目录
		sample_idx: 要可视化的批次中样本索引
		include_error_map: 是否包含错误分析图
	"""
	# 确保输出目录存在
	os.makedirs(output_dir, exist_ok=True)
	
	# 提取待可视化样本的数据
	input_patch = _extract_sample(inputs, sample_idx)
	pred_patch = _extract_sample(predictions, sample_idx)
	label_patch = _extract_sample(labels, sample_idx)
	
	# 获取中心切片
	D = input_patch.shape[0]
	mid_slice = D // 2
	
	# 保存基本的三联图
	_save_triplet_visualization(
		input_slice=input_patch[mid_slice],
		pred_slice=pred_patch[mid_slice],
		label_slice=label_patch[mid_slice],
		filename=os.path.join(output_dir, f"epoch_{epoch}_batch_{batch_idx}.png")
	)
	
	# 如果需要，保存错误分析图
	if include_error_map:
		_save_error_analysis(
			pred_slice=pred_patch[mid_slice],
			label_slice=label_patch[mid_slice],
			filename=os.path.join(output_dir, f"epoch_{epoch}_batch_{batch_idx}_error.png")
		)


def _extract_sample(tensor: torch.Tensor, sample_idx: int = 0) -> np.ndarray:
	"""
	从批次张量中提取单个样本并转换为NumPy数组。

	Args:
		tensor: 输入张量，形状为[B,C,D,H,W]或[B,D,H,W]
		sample_idx: 要提取的样本索引

	Returns:
		提取的样本，形状为[D,H,W]
	"""
	if tensor.dim() == 5:  # [B,C,D,H,W]
		sample = tensor[sample_idx, 0].cpu().numpy()
	else:  # [B,D,H,W]
		sample = tensor[sample_idx].cpu().numpy()
	
	return sample


def _save_triplet_visualization(
		input_slice: np.ndarray,
		pred_slice: np.ndarray,
		label_slice: np.ndarray,
		filename: str
) -> None:
	"""
	保存输入、预测和标签的三联图。

	Args:
		input_slice: 输入切片
		pred_slice: 预测切片
		label_slice: 标签切片
		filename: 输出文件名
	"""
	plt.figure(figsize=(12, 4))
	
	# 输入切片
	plt.subplot(131)
	plt.imshow(input_slice, cmap='gray')
	plt.title('Input')
	plt.axis('off')
	
	# 预测切片
	plt.subplot(132)
	plt.imshow(pred_slice, cmap='hot')
	plt.title('Prediction')
	plt.axis('off')
	
	# 标签切片
	plt.subplot(133)
	plt.imshow(label_slice, cmap='hot')
	plt.title('Ground Truth')
	plt.axis('off')
	
	plt.tight_layout()
	plt.savefig(filename, dpi=150, bbox_inches='tight')
	plt.close()


def _save_error_analysis(
		pred_slice: np.ndarray,
		label_slice: np.ndarray,
		filename: str
) -> None:
	"""
	保存预测和标签的错误分析图。

	Args:
		pred_slice: 预测切片
		label_slice: 标签切片
		filename: 输出文件名
	"""
	plt.figure(figsize=(6, 6))
	
	# 创建RGB错误图：绿色=TP，红色=FP，蓝色=FN
	rgb = np.zeros((pred_slice.shape[0], pred_slice.shape[1], 3))
	rgb[:, :, 0] = pred_slice * (1 - label_slice)  # 红色：假阳性(FP)
	rgb[:, :, 1] = pred_slice * label_slice  # 绿色：真阳性(TP)
	rgb[:, :, 2] = (1 - pred_slice) * label_slice  # 蓝色：假阴性(FN)
	
	plt.imshow(rgb)
	plt.title('TP(绿)/FP(红)/FN(蓝)')
	plt.axis('off')
	
	plt.tight_layout()
	plt.savefig(filename, dpi=150, bbox_inches='tight')
	plt.close()