import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import time
import math
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse
import logging
import json
from datetime import datetime

# 导入自定义模块
from config import DATA_CONFIG, MODEL_CONFIG, TRAIN_CONFIG, LOSS_CONFIG, AUGMENTATION_CONFIG
from models.model import VesselSegModel
from data.dataset import VesselSegDataset
from data.transforms import get_training_transforms, get_validation_transforms
from utils.losses import DiceLoss, WeightedDiceLoss, CombinedLoss, calculate_class_weights
from utils.metrics import calculate_dice, calculate_all_metrics


def setup_logging(log_dir):
	"""设置日志记录"""
	if not os.path.exists(log_dir):
		os.makedirs(log_dir)
	
	timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
	log_file = os.path.join(log_dir, f'training_{timestamp}.log')
	
	# 配置日志格式
	logging.basicConfig(
		level=logging.INFO,
		format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
		handlers=[
			logging.FileHandler(log_file),
			logging.StreamHandler()
		]
	)
	
	return logging.getLogger(__name__), timestamp


def parse_args():
	"""解析命令行参数"""
	parser = argparse.ArgumentParser(description='Train Vessel Segmentation Model')
	
	# 数据参数
	parser.add_argument('--data_dir', type=str, required=True, help='数据目录路径')
	parser.add_argument('--output_dir', type=str, default='outputs', help='输出目录路径')
	
	# 配置参数
	parser.add_argument('--config', type=str, default='', help='自定义配置文件路径(JSON格式)')
	
	return parser.parse_args()


def load_custom_config(config_path):
	"""加载自定义配置"""
	if not config_path or not os.path.exists(config_path):
		return {}
	
	with open(config_path, 'r') as f:
		return json.load(f)


def update_config(base_config, custom_config):
	"""更新配置"""
	for key, value in custom_config.items():
		if isinstance(value, dict) and key in base_config and isinstance(base_config[key], dict):
			update_config(base_config[key], value)
		else:
			base_config[key] = value


def load_data(data_dir):
	"""
	加载训练和验证数据

	参数:
		data_dir: 数据目录路径

	返回:
		train_volumes, train_labels, val_volumes, val_labels
	"""
	# 这里应该实现实际的数据加载逻辑
	# 以下是示例代码，实际应用中需要替换为真实数据加载
	
	# 创建随机示例数据
	num_train = 5
	num_val = 2
	vol_shape = (128, 128, 128)
	
	# 创建随机训练数据
	train_volumes = [np.random.rand(*vol_shape).astype(np.float32) for _ in range(num_train)]
	train_labels = [np.random.randint(0, 2, vol_shape).astype(np.float32) for _ in range(num_train)]
	
	# 创建随机验证数据
	val_volumes = [np.random.rand(*vol_shape).astype(np.float32) for _ in range(num_val)]
	val_labels = [np.random.randint(0, 2, vol_shape).astype(np.float32) for _ in range(num_val)]
	
	return train_volumes, train_labels, val_volumes, val_labels


def warmup_cosine_schedule(epoch, warmup_epochs, total_epochs):
	"""
	余弦退火学习率调度器，带预热阶段

	参数:
		epoch: 当前轮数
		warmup_epochs: 预热轮数
		total_epochs: 总轮数

	返回:
		lr_factor: 学习率因子
	"""
	if epoch < warmup_epochs:
		return epoch / warmup_epochs
	else:
		return 0.5 * (1 + math.cos(math.pi * (epoch - warmup_epochs) / (total_epochs - warmup_epochs)))


def train(args, configs, logger, timestamp):
	"""
	训练主函数

	参数:
		args: 命令行参数
		configs: 配置字典
		logger: 日志记录器
		timestamp: 时间戳
	"""
	data_config = configs['data']
	model_config = configs['model']
	train_config = configs['train']
	loss_config = configs['loss']
	aug_config = configs['aug']
	
	# 创建输出目录
	output_dir = args.output_dir
	if not os.path.exists(output_dir):
		os.makedirs(output_dir)
	
	# 创建保存目录
	save_dir = os.path.join(output_dir, f'model_{timestamp}')
	if not os.path.exists(save_dir):
		os.makedirs(save_dir)
	
	# 保存配置
	config_save_path = os.path.join(save_dir, 'config.json')
	with open(config_save_path, 'w') as f:
		json.dump(configs, f, indent=4)
	
	# 加载数据
	logger.info("加载数据...")
	train_volumes, train_labels, val_volumes, val_labels = load_data(args.data_dir)
	
	# 创建数据集
	train_dataset = VesselSegDataset(
		volumes=train_volumes,
		labels=train_labels,
		patch_size=data_config['patch_size'],
		samples_per_volume=data_config['samples_per_volume'],
		transform=get_training_transforms(aug_config)
	)
	
	val_dataset = VesselSegDataset(
		volumes=val_volumes,
		labels=val_labels,
		patch_size=data_config['patch_size'],
		samples_per_volume=data_config['samples_per_volume'] * 2,  # 验证时使用更多样本
		transform=get_validation_transforms()
	)
	
	# 创建数据加载器
	train_loader = DataLoader(
		train_dataset,
		batch_size=train_config['batch_size'],
		shuffle=True,
		num_workers=4,
		pin_memory=True
	)
	
	val_loader = DataLoader(
		val_dataset,
		batch_size=train_config['batch_size'],
		shuffle=False,
		num_workers=4,
		pin_memory=True
	)
	
	# 初始化模型
	logger.info("初始化模型...")
	model = VesselSegModel(
		input_channels=model_config['input_channels'],
		output_classes=model_config['output_classes'],
		feature_channels=model_config['feature_channels'],
		max_harmonic_degree=model_config['max_harmonic_degree']
	)
	
	# 移到GPU
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	model = model.to(device)
	
	# 计算类别权重
	logger.info("计算类别权重...")
	# 收集所有标签
	all_labels = torch.cat([
		torch.from_numpy(label).flatten()
		for label in train_labels
	])
	class_weights = calculate_class_weights(all_labels)
	logger.info(f"类别权重: {class_weights}")
	
	# 损失函数
	criterion = CombinedLoss(
		alpha=loss_config['alpha'],
		gamma=loss_config['gamma'],
		class_weights=class_weights.to(device)
	)
	
	# 优化器 - 使用AdamW
	optimizer = optim.AdamW(
		model.parameters(),
		lr=train_config['lr'],
		weight_decay=train_config['weight_decay']
	)
	
	# 学习率调度器 - 余弦退火
	lr_scheduler_fn = lambda epoch: warmup_cosine_schedule(
		epoch, train_config['warmup_epochs'], train_config['epochs']
	)
	lr_scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_scheduler_fn)
	
	# 训练跟踪
	best_val_dice = 0.0
	train_losses = []
	val_dices = []
	
	# 训练循环
	logger.info(f"开始训练，总轮数: {train_config['epochs']}")
	for epoch in range(train_config['epochs']):
		# 训练阶段
		model.train()
		epoch_loss = 0
		
		progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{train_config['epochs']}")
		for batch_idx, (inputs, targets) in enumerate(progress_bar):
			inputs, targets = inputs.to(device), targets.to(device)
			
			# 前向传播
			outputs = model(inputs)
			loss = criterion(outputs, targets)
			
			# 反向传播和优化
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()
			
			# 更新进度条
			epoch_loss += loss.item()
			progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})
		
		# 更新学习率
		lr_scheduler.step()
		current_lr = optimizer.param_groups[0]['lr']
		
		# 每个epoch的平均损失
		avg_loss = epoch_loss / len(train_loader)
		train_losses.append(avg_loss)
		logger.info(f"Epoch {epoch + 1}/{train_config['epochs']} | 平均损失: {avg_loss:.4f} | 学习率: {current_lr:.6f}")
		
		# 验证阶段
		if (epoch + 1) % train_config['validate_every'] == 0:
			val_dice = validate(model, val_loader, device, logger)
			val_dices.append(val_dice)
			
			# 保存最佳模型
			if val_dice > best_val_dice:
				best_val_dice = val_dice
				torch.save(
					model.state_dict(),
					os.path.join(save_dir, "best_model.pth")
				)
				logger.info(f"保存最佳模型，Dice: {best_val_dice:.4f}")
		
		# 定期保存模型
		if (epoch + 1) % train_config['save_every'] == 0:
			torch.save(
				model.state_dict(),
				os.path.join(save_dir, f"model_epoch_{epoch + 1}.pth")
			)
			logger.info(f"保存检查点: epoch_{epoch + 1}")
	
	# 保存最终模型
	torch.save(
		model.state_dict(),
		os.path.join(save_dir, "final_model.pth")
	)
	logger.info("训练完成，保存最终模型")
	
	# 绘制训练曲线
	plot_training_curves(train_losses, val_dices, train_config, save_dir)
	
	return model


def validate(model, val_loader, device, logger):
	"""
	验证函数

	参数:
		model: 模型
		val_loader: 验证数据加载器
		device: 设备
		logger: 日志记录器

	返回:
		avg_dice: 平均Dice系数
	"""
	model.eval()
	dice_scores = []
	
	logger.info("开始验证...")
	with torch.no_grad():
		for inputs, targets in tqdm(val_loader, desc="Validating"):
			inputs, targets = inputs.to(device), targets.to(device)
			
			# 前向传播
			outputs = model(inputs)
			
			# 二值化预测
			predictions = (outputs > 0.5).float()
			
			# 计算Dice系数
			dice = calculate_dice(predictions, targets)
			dice_scores.append(dice.item())
	
	# 计算平均Dice系数
	avg_dice = sum(dice_scores) / len(dice_scores)
	logger.info(f"验证Dice系数: {avg_dice:.4f}")
	
	return avg_dice


def plot_training_curves(train_losses, val_dices, train_config, save_dir):
	"""
	绘制训练曲线

	参数:
		train_losses: 训练损失
		val_dices: 验证Dice系数
		train_config: 训练配置
		save_dir: 保存目录
	"""
	# 创建图形
	plt.figure(figsize=(12, 5))
	
	# 绘制训练损失
	plt.subplot(1, 2, 1)
	plt.plot(train_losses, 'b-', label='Training Loss')
	plt.title('Training Loss')
	plt.xlabel('Epoch')
	plt.ylabel('Loss')
	plt.legend()
	
	# 绘制验证Dice系数
	plt.subplot(1, 2, 2)
	val_epochs = range(train_config['validate_every'] - 1, len(train_losses), train_config['validate_every'])
	plt.plot(val_epochs, val_dices, 'r-', label='Validation Dice')
	plt.title('Validation Dice')
	plt.xlabel('Epoch')
	plt.ylabel('Dice')
	plt.legend()
	
	# 保存图形
	plt.savefig(os.path.join(save_dir, 'training_curves.png'))
	plt.close()


if __name__ == "__main__":
	# 解析命令行参数
	args = parse_args()
	
	# 加载配置
	configs = {
		'data': DATA_CONFIG,
		'model': MODEL_CONFIG,
		'train': TRAIN_CONFIG,
		'loss': LOSS_CONFIG,
		'aug': AUGMENTATION_CONFIG
	}
	
	# 加载自定义配置
	custom_config = load_custom_config(args.config)
	if custom_config:
		for key, value in custom_config.items():
			if key in configs:
				update_config(configs[key], value)
	
	# 设置日志
	logger, timestamp = setup_logging(TRAIN_CONFIG['log_dir'])
	
	# 记录训练配置
	logger.info("训练配置:")
	for config_name, config_dict in configs.items():
		logger.info(f"  {config_name}:")
		for key, value in config_dict.items():
			logger.info(f"    {key}: {value}")
	
	# 开始训练
	train(args, configs, logger, timestamp)