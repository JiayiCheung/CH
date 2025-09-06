# models/dataload/ch_dataload.py

import os
import json
from torch.utils.data import DataLoader
from data.dataset import HepaticVesselDataset
from data.transforms import get_training_transforms, get_validation_transforms


def load_ch_data(config):
	"""
	加载训练和验证数据，使用基于nnUNet的处理策略

	参数:
		config: 配置字典

	返回:
		train_loader, val_loader: 训练和验证数据加载器
	"""
	# 预处理数据目录
	preprocessed_dir = "/home/bingxing2/home/scx7as9/run/CSUqx/CH/data/preprocessed"
	data_dir = os.path.join(preprocessed_dir, "data")
	properties_dir = os.path.join(preprocessed_dir, "properties")
	splits_file = os.path.join(preprocessed_dir, "splits.json")
	
	# 检查目录是否存在
	if not os.path.exists(data_dir) or not os.path.exists(properties_dir):
		raise ValueError(f"预处理数据目录不存在: {preprocessed_dir}\n请先运行预处理脚本!")
	
	# 读取划分文件
	if not os.path.exists(splits_file):
		print(f"警告: 划分文件不存在 {splits_file}, 将使用所有数据作为训练集")
		splits_file = None
	
	# 补丁大小
	patch_size = config['data'].get('patch_size', 64)
	if isinstance(patch_size, int):
		patch_size = (patch_size, patch_size, patch_size)
	
	# 前景过采样比例
	oversample_foreground_percent = 0.33  # 默认值，适合血管分割任务
	
	# 每个体积的采样数量
	samples_per_volume = config['data'].get('samples_per_volume', 30)
	
	# 创建训练集
	train_dataset = HepaticVesselDataset(
		data_dir=data_dir,
		properties_dir=properties_dir,
		patch_size=patch_size,
		mode='train',
		splits_file=splits_file,
		oversample_foreground_percent=oversample_foreground_percent,
		num_samples_per_volume=samples_per_volume,
		transform=get_training_transforms(config['aug'])
	)
	
	# 创建验证集
	val_dataset = HepaticVesselDataset(
		data_dir=data_dir,
		properties_dir=properties_dir,
		patch_size=patch_size,
		mode='val',
		splits_file=splits_file,
		oversample_foreground_percent=0.0,  # 验证集不需要过采样
		num_samples_per_volume=max(1, samples_per_volume // 5),  # 验证集样本数量减少
		transform=get_validation_transforms()
	)
	
	# 创建数据加载器
	train_loader = DataLoader(
		train_dataset,
		batch_size=config['train']['batch_size'],
		shuffle=True,
		num_workers=8,
		pin_memory=True
	)
	
	val_loader = DataLoader(
		val_dataset,
		batch_size=config['train']['batch_size'],
		shuffle=False,
		num_workers=4,
		pin_memory=True
	)
	
	print(f"训练集: {len(train_dataset)} 个样本")
	print(f"验证集: {len(val_dataset)} 个样本")
	
	return train_loader, val_loader