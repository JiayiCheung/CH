#!/usr/bin/env python3
"""
分布式训练脚本 - 完整实现版
支持跨节点流水线训练
"""

import os
import sys
import time
import argparse
import yaml
import logging
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.utils.data import DataLoader, DistributedSampler

# 添加项目根目录到Python路径
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from data.dataset import LiverVesselDataset
from models import create_vessel_segmenter
from loss.combined_loss import CombinedLoss
from scripts.distributed.cross_node_pipeline import create_pipeline
from scripts.distributed.node_communicator import NodeCommunicator

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_args():
	"""解析命令行参数"""
	parser = argparse.ArgumentParser(description='Distributed Liver Vessel Segmentation Training')
	
	# 数据参数
	parser.add_argument('--image_dir', type=str, required=True, help='Images directory')
	parser.add_argument('--label_dir', type=str, required=True, help='Labels directory')
	parser.add_argument('--output_dir', type=str, default='./output', help='Output directory')
	parser.add_argument('--config', type=str, default='configs/default.yaml', help='Config file')
	
	# 训练参数
	parser.add_argument('--batch_size', type=int, default=1, help='Batch size')
	parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
	parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
	parser.add_argument('--num_workers', type=int, default=4, help='Number of workers')
	parser.add_argument('--val_interval', type=int, default=5, help='Validation interval')
	parser.add_argument('--save_interval', type=int, default=10, help='Save interval')
	parser.add_argument('--log_interval', type=int, default=10, help='Log interval')
	
	# 分布式参数
	parser.add_argument('--resume', type=str, help='Resume from checkpoint')
	parser.add_argument('--amp', action='store_true', help='Use automatic mixed precision')
	
	return parser.parse_args()


def setup_distributed():
	"""设置分布式环境"""
	if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
		rank = int(os.environ['RANK'])
		world_size = int(os.environ['WORLD_SIZE'])
		local_rank = int(os.environ['LOCAL_RANK'])
	else:
		raise RuntimeError("Distributed environment not properly set up")
	
	# 初始化分布式进程组
	dist.init_process_group(
		backend='nccl',
		init_method='env://',
		world_size=world_size,
		rank=rank
	)
	
	# 设置CUDA设备
	torch.cuda.set_device(local_rank)
	
	return rank, world_size, local_rank


def cleanup_distributed():
	"""清理分布式环境"""
	if dist.is_initialized():
		dist.destroy_process_group()


def load_config(config_path):
	"""加载配置文件"""
	with open(config_path, 'r') as f:
		config = yaml.safe_load(f)
	return config


def create_dataloaders(args, config, rank, world_size):
	"""创建数据加载器"""
	# 数据集配置
	data_config = config.get('data', {})
	
	# 训练数据集
	train_dataset = LiverVesselDataset(
		image_dir=args.image_dir,
		label_dir=args.label_dir,
		max_cases=data_config.get('max_cases', None),
		random_sampling=data_config.get('random_sampling', True),
		enable_smart_sampling=True,
		config=config
	)
	
	# 验证数据集
	val_dataset = LiverVesselDataset(
		image_dir=args.image_dir,
		label_dir=args.label_dir,
		max_cases=data_config.get('max_val_cases', None),
		random_sampling=False,
		enable_smart_sampling=False,
		config=config
	)
	
	# 分布式采样器
	train_sampler = DistributedSampler(
		train_dataset,
		num_replicas=world_size,
		rank=rank,
		shuffle=True
	)
	
	val_sampler = DistributedSampler(
		val_dataset,
		num_replicas=world_size,
		rank=rank,
		shuffle=False
	)
	
	# 数据加载器
	train_loader = DataLoader(
		train_dataset,
		batch_size=args.batch_size,
		sampler=train_sampler,
		num_workers=args.num_workers,
		pin_memory=True,
		drop_last=True
	)
	
	val_loader = DataLoader(
		val_dataset,
		batch_size=args.batch_size,
		sampler=val_sampler,
		num_workers=args.num_workers,
		pin_memory=True,
		drop_last=False
	)
	
	return train_loader, val_loader, train_sampler, val_sampler


def create_pipeline_stages(config, rank, device):
	"""创建流水线阶段"""
	from scripts.distributed.stages import (
		FrontendStage, PatchSchedulingStage, CHProcessingStage,
		SpatialFusionStage, FeatureFusionStage, MultiscaleFusionStage,
		BackendStage
	)
	
	# 创建完整模型（用于提取组件）
	full_model = create_vessel_segmenter(config)
	
	stages = {}
	
	# 根据rank创建相应的阶段
	if rank == 0:  # 节点1, GPU 0 - 预处理
		stages['preprocessing'] = FrontendStage(full_model, device)
	elif rank == 1:  # 节点1, GPU 1 - 采样调度
		stages['patch_scheduling'] = PatchSchedulingStage(full_model, device)
	elif rank == 2:  # 节点1, GPU 2 - CH分支
		stages['ch_branch'] = CHProcessingStage(full_model, device)
	elif rank == 3:  # 节点1, GPU 3 - 空间分支
		stages['spatial_branch'] = SpatialFusionStage(full_model, device)
	elif rank == 4:  # 节点2, GPU 4 - 特征融合
		stages['feature_fusion'] = FeatureFusionStage(full_model, device)
	elif rank == 5:  # 节点2, GPU 5 - 多尺度融合
		stages['multiscale_fusion'] = MultiscaleFusionStage(full_model, device)
	elif rank == 6:  # 节点2, GPU 6 - 分割头
		stages['segmentation_head'] = BackendStage(full_model, device)
	
	return stages


def create_distributed_model(config, rank, device):
	"""创建分布式模型"""
	# 创建节点通信
	node_comm = NodeCommunicator()
	
	# 创建流水线阶段
	stages = create_pipeline_stages(config, rank, device)
	
	# 创建跨节点流水线
	pipeline = create_pipeline(config, node_comm)
	pipeline.stages = stages
	
	# 启动工作线程
	pipeline.start_worker()
	
	return pipeline


def create_loss_function(config):
	"""创建损失函数"""
	loss_config = config.get('loss', {})
	
	return CombinedLoss(
		num_classes=loss_config.get('num_classes', 1),
		vessel_weight=loss_config.get('vessel_weight', 10.0),
		tumor_weight=loss_config.get('tumor_weight', 15.0),
		use_boundary=loss_config.get('use_boundary', True)
	)


def train_epoch(model, dataloader, optimizer, loss_fn, device, epoch, args, scaler=None):
	"""训练一个epoch - 完整实现"""
	model.train()
	
	total_loss = 0.0
	num_batches = 0
	
	for batch_idx, batch in enumerate(dataloader):
		try:
			# 1. 数据预处理
			if isinstance(batch, (list, tuple)) and len(batch) >= 2:
				images, labels = batch[0], batch[1]
			else:
				images, labels = batch, None
			
			# 2. 移动到设备
			images = images.to(device, non_blocking=True)
			if labels is not None:
				labels = labels.to(device, non_blocking=True)
			
			# 3. 前向传播
			optimizer.zero_grad()
			
			if args.amp and scaler is not None:
				# 混合精度训练
				with torch.cuda.amp.autocast():
					outputs = model.forward(images)
					if labels is not None:
						loss = loss_fn(outputs, labels)
					else:
						loss = loss_fn(outputs, images)
				
				# 反向传播
				scaler.scale(loss).backward()
				scaler.step(optimizer)
				scaler.update()
			else:
				# 普通训练
				outputs = model.forward(images)
				
				if labels is not None:
					loss = loss_fn(outputs, labels)
				else:
					loss = loss_fn(outputs, images)
				
				# 反向传播
				loss.backward()
				
				# 梯度裁剪
				torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
				optimizer.step()
			
			# 4. 统计
			total_loss += loss.item()
			num_batches += 1
			
			# 5. 日志输出
			if batch_idx % args.log_interval == 0:
				logger.info(f'Epoch {epoch}, Batch {batch_idx}/{len(dataloader)}, '
				            f'Loss: {loss.item():.6f}')
		
		except Exception as e:
			logger.error(f"Training batch {batch_idx} failed: {e}")
			continue
	
	avg_loss = total_loss / max(num_batches, 1)
	return avg_loss


def validate_epoch(model, dataloader, loss_fn, device, epoch, args):
	"""验证一个epoch"""
	model.eval()
	
	total_loss = 0.0
	num_batches = 0
	
	with torch.no_grad():
		for batch_idx, batch in enumerate(dataloader):
			try:
				# 数据预处理
				if isinstance(batch, (list, tuple)) and len(batch) >= 2:
					images, labels = batch[0], batch[1]
				else:
					images, labels = batch, None
				
				# 移动到设备
				images = images.to(device, non_blocking=True)
				if labels is not None:
					labels = labels.to(device, non_blocking=True)
				
				# 前向传播
				outputs = model.forward(images)
				
				# 计算损失
				if labels is not None:
					loss = loss_fn(outputs, labels)
				else:
					loss = loss_fn(outputs, images)
				
				total_loss += loss.item()
				num_batches += 1
			
			except Exception as e:
				logger.error(f"Validation batch {batch_idx} failed: {e}")
				continue
	
	avg_loss = total_loss / max(num_batches, 1)
	return avg_loss


def save_checkpoint(model, optimizer, epoch, output_dir, rank):
	"""保存检查点"""
	if rank == 0:  # 只在主进程保存
		checkpoint = {
			'epoch': epoch,
			'model_state_dict': model.state_dict(),
			'optimizer_state_dict': optimizer.state_dict(),
		}
		
		checkpoint_path = Path(output_dir) / f'checkpoint_epoch_{epoch}.pt'
		checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
		
		torch.save(checkpoint, checkpoint_path)
		logger.info(f"Checkpoint saved to {checkpoint_path}")


def load_checkpoint(model, optimizer, checkpoint_path):
	"""加载检查点"""
	checkpoint = torch.load(checkpoint_path, map_location='cpu')
	
	model.load_state_dict(checkpoint['model_state_dict'])
	optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
	
	return checkpoint['epoch']


def main():
	"""主函数"""
	# 解析参数
	args = parse_args()
	
	# 设置分布式环境
	rank, world_size, local_rank = setup_distributed()
	device = torch.cuda.current_device()
	
	# 加载配置
	config = load_config(args.config)
	
	# 创建输出目录
	Path(args.output_dir).mkdir(parents=True, exist_ok=True)
	
	# 创建数据加载器
	train_loader, val_loader, train_sampler, val_sampler = create_dataloaders(
		args, config, rank, world_size
	)
	
	# 创建模型
	model = create_distributed_model(config, rank, device)
	
	# 创建优化器和损失函数
	optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
	loss_fn = create_loss_function(config)
	
	# 学习率调度器
	scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
	
	# 混合精度训练
	scaler = torch.cuda.amp.GradScaler() if args.amp else None
	
	# 加载检查点
	start_epoch = 0
	if args.resume:
		start_epoch = load_checkpoint(model, optimizer, args.resume)
		logger.info(f"Resumed from epoch {start_epoch}")
	
	# 主训练循环
	for epoch in range(start_epoch, args.epochs):
		# 设置epoch（用于DistributedSampler）
		train_sampler.set_epoch(epoch)
		
		# 训练
		train_loss = train_epoch(model, train_loader, optimizer, loss_fn,
		                         device, epoch, args, scaler)
		
		# 验证（每几个epoch一次）
		if epoch % args.val_interval == 0:
			val_loss = validate_epoch(model, val_loader, loss_fn,
			                          device, epoch, args)
			
			if rank == 0:
				logger.info(f'Epoch {epoch}: Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}')
		
		# 更新学习率
		scheduler.step()
		
		# 保存检查点
		if epoch % args.save_interval == 0:
			save_checkpoint(model, optimizer, epoch, args.output_dir, rank)
	
	# 清理
	model.stop_worker()
	cleanup_distributed()


if __name__ == '__main__':
	main()