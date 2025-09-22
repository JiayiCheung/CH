#/models/dataload/ch_dataload.py

import os
import torch
from torch.utils.data import DataLoader
from .ch_dataset import VesselDataset
from torch.utils.data.distributed import DistributedSampler


def get_dataloaders(config):
	"""
	创建训练和验证数据加载器

	参数:
		config: 配置字典，包含以下键:
			- data_dir: 处理后数据目录
			- props_dir: 属性文件目录
			- split_file: 数据集划分文件
			- patch_size: 补丁大小
			- batch_size: 批次大小
			- num_workers: 工作进程数
			- oversample_foreground_percent: 前景过采样比例

	返回:
		train_loader, val_loader
	"""
	# 获取配置参数
	data_dir = config['data'].get('data_dir', 'preprocess/processed/data')
	props_dir = config['data'].get('props_dir', 'preprocess/processed/properties')
	split_file = config['data'].get('split_file', 'preprocess/splits.json')
	patch_size = config['data'].get('patch_size', (64, 128, 128))
	batch_size = config['data'].get('batch_size', 2)
	num_workers = config['data'].get('num_workers', 4)
	oversample_foreground_percent = config['data'].get('oversample_foreground_percent', 0.9)
	
	# 创建训练数据集
	train_dataset = VesselDataset(
		data_dir=data_dir,
		props_dir=props_dir,
		split_file=split_file,
		split='train',
		patch_size=patch_size,
		oversample_foreground_percent=oversample_foreground_percent,
		mode='train'
	)
	
	# 创建验证数据集
	val_dataset = VesselDataset(
		data_dir=data_dir,
		props_dir=props_dir,
		split_file=split_file,
		split='val',
		patch_size=patch_size,
		oversample_foreground_percent=0.0,  # 验证时不需要过采样
		mode='val'
	)
	
	train_sampler = DistributedSampler(train_dataset)
	# 创建数据加载器
	train_loader = DataLoader(
		train_dataset,
		batch_size=batch_size,
		sampler=train_sampler,
		num_workers=num_workers,
		pin_memory=True,
		drop_last=True
	)
	
	val_sampler = DistributedSampler(val_dataset, shuffle=False)
	val_loader = DataLoader(
		val_dataset,
		batch_size=1,  # 验证通常使用批次大小为1
		sampler=val_sampler,
		num_workers=max(1, num_workers // 2),  # 验证时使用较少的工作进程
		pin_memory=True
	)
	
	print(f"创建了训练数据加载器: {len(train_dataset)} 样本, 批次大小 {batch_size}")
	print(f"创建了验证数据加载器: {len(val_dataset)} 样本, 批次大小 1")
	
	return train_loader, val_loader