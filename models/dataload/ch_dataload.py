import os
import numpy as np
from torch.utils.data import DataLoader
from models.dataload.ch_dataset import CHDataset
from data.transforms import get_training_transforms, get_validation_transforms


# 直接添加到train.py文件中
def load_ch_data(config):
	"""加载训练和验证数据，使用CHDataset实现惰性加载"""
	# 使用nnU-Net数据集路径
	images_dir = config['data']['data_dir']
	labels_dir = config['data']['label_dir']
	
	print(f"加载图像目录: {images_dir}")
	print(f"加载标签目录: {labels_dir}")
	
	# 获取预处理目录
	preprocessed_dir = config['data'].get('preprocessed_dir')
	if not preprocessed_dir or not os.path.exists(preprocessed_dir):
		raise ValueError(f"必须在配置文件中提供有效的preprocessed_dir，包含预计算的Frangi响应")
	
	print(f"使用预计算的Frangi响应: {preprocessed_dir}")
	
	# 读取所有体积文件的路径
	volume_files = sorted([os.path.join(images_dir, f) for f in os.listdir(images_dir) if f.endswith('.nii.gz')])
	label_files = sorted([os.path.join(labels_dir, f) for f in os.listdir(labels_dir) if f.endswith('.nii.gz')])
	
	print(f"找到 {len(volume_files)} 个体积文件和 {len(label_files)} 个标签文件")
	
	# 划分训练集和验证集
	num_samples = len(volume_files)
	np.random.seed(config['data']['random_seed'])
	indices = np.random.permutation(num_samples)
	split_idx = int(num_samples * config['data']['train_val_split'])
	
	train_indices = indices[:split_idx]
	val_indices = indices[split_idx:]
	
	train_volume_paths = [volume_files[i] for i in train_indices]
	train_label_paths = [label_files[i] for i in train_indices]
	val_volume_paths = [volume_files[i] for i in val_indices]
	val_label_paths = [label_files[i] for i in val_indices]
	
	print(f"训练集: {len(train_volume_paths)} 个样本, 验证集: {len(val_volume_paths)} 个样本")
	
	# 创建惰性加载数据集
	train_dataset = CHDataset(
		volume_paths=train_volume_paths,
		label_paths=train_label_paths,
		patch_size=config['data']['patch_size'],
		samples_per_volume=config['data']['samples_per_volume'],
		transform=get_training_transforms(config['aug']),
		preprocessed_dir=preprocessed_dir
	)
	
	val_dataset = CHDataset(
		volume_paths=val_volume_paths,
		label_paths=val_label_paths,
		patch_size=config['data']['patch_size'],
		samples_per_volume=config['data']['samples_per_volume'],
		transform=get_validation_transforms(),
		preprocessed_dir=preprocessed_dir
	)
	
	# 创建数据加载器
	train_loader = DataLoader(
		train_dataset,
		batch_size=config['train']['batch_size'],
		shuffle=True,
		num_workers=1,  # 减少worker数量以节省内存
		pin_memory=True
	)
	
	val_loader = DataLoader(
		val_dataset,
		batch_size=config['train']['batch_size'],
		shuffle=False,
		num_workers=1,
		pin_memory=True
	)
	
	return train_loader, val_loader