#train.py


import os
import gc
import yaml
import argparse
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import Adam

# 导入自定义模块
from models.dataload.ch_dataload import get_dataloaders
from models.vessel_segmenter import VesselSegmenter
from utils.losses import CombinedLoss
from utils.tensorboard_logger import TensorBoardLogger


def setup_distributed(local_rank):
	"""
	设置分布式训练环境

	参数:
		local_rank: 本地进程序号
	"""
	# 设置当前设备
	torch.cuda.set_device(local_rank)
	
	# 初始化分布式进程组
	dist.init_process_group(backend='nccl')
	
	# 返回当前进程的rank和总进程数
	rank = dist.get_rank()
	world_size = dist.get_world_size()
	
	print(f"Initialized process {rank}/{world_size} on device {local_rank}")
	return rank, world_size


def load_config(config_path):
	"""
	加载配置文件

	参数:
		config_path: 配置文件路径
	"""
	with open(config_path, 'r') as file:
		config = yaml.safe_load(file)
	return config


def create_model(config, local_rank):
	"""
	创建并初始化模型

	参数:
		config: 配置字典
		local_rank: 本地进程序号
	"""
	# 创建模型实例
	model = VesselSegmenter(
		in_channels=config['model']['in_channels'],
		out_channels=config['model']['out_channels']
	)
	
	# 移动模型到对应GPU
	model = model.cuda(local_rank)
	
	# 使用DDP包装模型
	model = DDP(model, device_ids=[local_rank], output_device=local_rank)
	
	return model


def create_optimizer(model, config):
	"""
	创建优化器

	参数:
		model: 神经网络模型
		config: 配置字典
	"""
	return Adam(
		model.parameters(),
		lr=float(config['training']['learning_rate']),
		weight_decay=float(config['training']['weight_decay'])
	)


def save_checkpoint(model, optimizer, epoch, best_dice, checkpoint_dir, filename):
	"""
	保存模型检查点

	参数:
		model: 模型
		optimizer: 优化器
		epoch: 当前轮次
		best_dice: 最佳Dice系数
		checkpoint_dir: 保存目录
		filename: 文件名
	"""
	# 确保目录存在
	os.makedirs(checkpoint_dir, exist_ok=True)
	
	# 保存检查点
	checkpoint = {
		'epoch': epoch,
		'model_state_dict': model.module.state_dict(),  # 使用.module访问DDP封装的模型
		'optimizer_state_dict': optimizer.state_dict(),
		'best_dice': best_dice
	}
	
	torch.save(checkpoint, os.path.join(checkpoint_dir, filename))
	print(f"Checkpoint saved: {filename}")


def dice_coefficient(pred, target):
	"""
	计算Dice系数

	参数:
		pred: 预测结果 [B, C, D, H, W]
		target: 真实标签 [B, C, D, H, W]
	"""
	smooth = 1e-5
	pred = pred.float()
	target = target.float()
	
	# 将预测转换为二值
	pred = (pred > 0.5).float()
	
	# 计算交集
	intersection = (pred * target).sum()
	
	# 计算并集
	union = pred.sum() + target.sum()
	
	# 计算Dice系数
	dice = (2. * intersection + smooth) / (union + smooth)
	
	return dice.item()


def validate(model, val_loader, local_rank, logger, step):
	"""
	验证模型性能

	参数:
		model: 模型
		val_loader: 验证数据加载器
		local_rank: 本地进程序号
		logger: TensorBoard日志记录器
		step: 当前全局步数
	"""
	model.eval()
	dice_sum = 0.0
	count = 0
	
	with torch.no_grad():
		for images, labels in val_loader:
			# 移动数据到GPU
			images = images.cuda(local_rank)
			labels = labels.cuda(local_rank)
			
			# 前向传播
			outputs = model(images)
			
			# 计算Dice系数
			dice = dice_coefficient(outputs, labels)
			dice_sum += dice
			count += 1
	
	# 计算平均Dice系数
	avg_dice = dice_sum / count if count > 0 else 0
	
	# 记录到TensorBoard
	logger.log_scalar('val/dice', avg_dice, step)
	
	# 恢复训练模式
	model.train()
	
	return avg_dice


def train(config, local_rank):
	"""
	训练主函数

	参数:
		config: 配置字典
		local_rank: 本地进程序号
	"""
	# 设置分布式环境
	rank, world_size = setup_distributed(local_rank)
	
	
	
	# 创建日志目录
	log_dir = os.path.join(config['logging']['log_dir'], f'rank{rank}')
	os.makedirs(log_dir, exist_ok=True)
	
	# 创建检查点目录
	checkpoint_dir = config['logging']['checkpoint_dir']
	os.makedirs(checkpoint_dir, exist_ok=True)
	
	# 初始化TensorBoard日志记录器
	logger = TensorBoardLogger(log_dir)
	
	# 加载数据
	train_loader, val_loader = get_dataloaders(config)

	
	# 创建模型和优化器
	model = create_model(config, local_rank)
	optimizer = create_optimizer(model, config)
	
	# 创建损失函数
	criterion = CombinedLoss(
		alpha=config['loss']['alpha'],
		gamma=config['loss']['gamma']
	).cuda(local_rank)
	
	# 训练参数
	num_epochs = config['training']['num_epochs']
	validate_every = config['training']['validate_every']
	save_every = config['training']['save_every']
	
	# 最佳Dice系数
	best_dice = 0.0
	
	# 开始训练
	model.train()
	steps_per_epoch = len(train_loader)
	
	for epoch in range(num_epochs):
		# 设置数据采样器的epoch
		train_loader.sampler.set_epoch(epoch)
		
		# 每个epoch开始时打印信息
		if rank == 0:
			print(f"Epoch {epoch + 1}/{num_epochs}")
		
		# 遍历训练数据
		for batch_idx, (images, labels) in enumerate(train_loader):
			# 计算全局步数
			step = epoch * steps_per_epoch + batch_idx
			
			# 移动数据到GPU
			images = images.cuda(local_rank)
			labels = labels.cuda(local_rank)
			
			# 前向传播
			outputs = model(images)
			
			# 计算损失
			loss = criterion(outputs, labels)
			
			# 反向传播和优化
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()
			
			# 记录到TensorBoard
			logger.log_scalar('train/loss', loss.item(), step)
			logger.log_scalar('train/lr', optimizer.param_groups[0]['lr'], step)
			
			# 打印训练信息
			if batch_idx % 10 == 0:
				print(
					f"Rank {rank}, Epoch {epoch + 1}/{num_epochs}, Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}")
		
		# 验证阶段（仅在rank 0执行）
		if rank == 0 and (epoch + 1) % validate_every == 0:
			print(f"Validating at epoch {epoch + 1}...")
			dice = validate(model, val_loader, local_rank, logger, step)
			print(f"Validation Dice: {dice:.4f}")
			
			# 保存最佳模型
			if dice > best_dice:
				best_dice = dice
				save_checkpoint(model, optimizer, epoch, best_dice, checkpoint_dir, 'best_model.pth')
		
		# 定期保存检查点（仅在rank 0执行）
		if rank == 0 and (epoch + 1) % save_every == 0:
			save_checkpoint(model, optimizer, epoch, best_dice, checkpoint_dir, f'checkpoint_epoch_{epoch + 1}.pth')
		
		# 每10个epoch清理内存
		if (epoch + 1) % 10 == 0:
			torch.cuda.empty_cache()
			gc.collect()
		
		# 同步所有进程
		dist.barrier()
	
	# 训练结束，保存最终模型（仅在rank 0执行）
	if rank == 0:
		save_checkpoint(model, optimizer, num_epochs, best_dice, checkpoint_dir, 'final_model.pth')
		print(f"Training completed. Best validation Dice: {best_dice:.4f}")


def main():
	"""主函数"""
	
	local_rank = int(os.environ["LOCAL_RANK"])
	# 解析命令行参数
	parser = argparse.ArgumentParser(description='Vessel Segmentation Training')
	parser.add_argument('--config', type=str, default='config/train_config.yaml', help='Path to config file')
	args = parser.parse_args()
	
	# 加载配置文件
	config = load_config(args.config)
	
	# 启动训练
	train(config, local_rank)


if __name__ == '__main__':
	main()