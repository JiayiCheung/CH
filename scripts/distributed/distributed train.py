import os
import argparse
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm
import sys
from pathlib import Path

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data import LiverVesselDataset, HardSampleTracker
from models import VesselSegmenter
from losses import VesselSegmentationLoss
from utils import SegmentationMetrics, Visualizer, Logger, SamplingScheduler
from data.sampling_manager import SamplingManager


def parse_args():
	"""解析命令行参数"""
	parser = argparse.ArgumentParser(description='Train Liver Vessel Segmentation Model')
	
	# 数据参数
	parser.add_argument('--image_dir', type=str, required=True, help='Path to input volume or directory')
	parser.add_argument('--label_dir', type=str, required=True, help='Path to label directory')
	parser.add_argument('--output_dir', type=str, default='./output', help='Path to output directory')
	parser.add_argument('--config', type=str, default='configs/default.yaml', help='Path to config file')
	
	# 训练参数
	parser.add_argument('--tier', type=int, help='Training tier (0, 1, or 2)')
	parser.add_argument('--batch_size', type=int, default=1, help='Batch size')
	parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
	parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
	parser.add_argument('--num_workers', type=int, default=4, help='Number of workers')
	parser.add_argument('--resume', type=str, help='Path to model checkpoint')
	
	# 分布式训练参数
	parser.add_argument('--distributed', action='store_true', help='Enable distributed training')
	parser.add_argument('--world_size', type=int, default=1, help='Number of processes')
	parser.add_argument('--rank', type=int, default=0, help='Process rank')
	parser.add_argument('--local_rank', type=int, default=0, help='Local process rank')
	parser.add_argument('--dist_url', type=str, default='env://', help='Distributed training URL')
	
	# 其他参数
	parser.add_argument('--seed', type=int, default=42, help='Random seed')
	parser.add_argument('--val_interval', type=int, default=5, help='Validation interval')
	parser.add_argument('--amp', action='store_true', help='Use mixed precision training')
	parser.add_argument('--smart_sampling', action='store_true', help='Enable smart sampling')
	
	return parser.parse_args()


def build_optimizer(model, base_lr=1e-4, kernel_lr_scale=0.1, weight_decay=1e-5):
	"""
	构建优化器 - 为边缘增强内核设置单独的学习率

	参数:
		model: 模型
		base_lr: 基础学习率
		kernel_lr_scale: 内核学习率缩放因子
		weight_decay: 权重衰减

	返回:
		优化器
	"""
	# 将参数分为边缘内核和其他参数
	edge_kernels, others = [], []
	
	for name, param in model.named_parameters():
		if not param.requires_grad:
			continue
		# 识别边缘内核参数
		if name.endswith(".kernels"):
			edge_kernels.append(param)
		else:
			others.append(param)
	
	# 创建参数组
	param_groups = [
		{
			"params": edge_kernels,
			"lr": base_lr * kernel_lr_scale,
			"weight_decay": 0.0  # 内核不应用权重衰减
		},
		{
			"params": others,
			"lr": base_lr,
			"weight_decay": weight_decay
		}
	]
	
	print(f"Optimizer: {len(edge_kernels)} edge kernels with lr={base_lr * kernel_lr_scale}, "
	      f"{len(others)} other params with lr={base_lr}")
	
	return optim.Adam(param_groups)


def train_one_epoch(model, dataloader, criterion, optimizer, scaler, device, epoch, args, logger=None):
	"""
	训练一个epoch

	参数:
		model: 模型
		dataloader: 数据加载器
		criterion: 损失函数
		optimizer: 优化器
		scaler: 梯度缩放器
		device: 设备
		epoch: 当前epoch
		args: 命令行参数
		logger: 日志记录器

	返回:
		平均损失
	"""
	model.train()
	running_loss = 0.0
	
	# 创建进度条
	iterator = tqdm(dataloader, desc=f"Epoch {epoch}") if args.rank == 0 else dataloader
	
	for batch in iterator:
		# 获取数据
		img = batch['image'].to(device, memory_format=torch.channels_last)
		lab = batch['label'].to(device)
		tier = batch['tier']
		
		# 清零梯度
		optimizer.zero_grad(set_to_none=True)
		
		# 处理每个tier
		outs = []
		for j, t in enumerate(tier):
			model.set_tier(int(t))
			with autocast(enabled=args.amp):
				outs.append(model(img[j:j + 1]))
		
		# 合并输出
		out = torch.cat(outs)
		
		# 计算损失
		with autocast(enabled=args.amp):
			loss = criterion(out, lab)
		
		# 反向传播
		scaler.scale(loss).backward()
		scaler.step(optimizer)
		scaler.update()
		
		# 更新统计
		running_loss += loss.item()
		
		# 更新进度条
		if args.rank == 0 and isinstance(iterator, tqdm):
			iterator.set_postfix(loss=loss.item())
	
	# 计算平均损失
	avg_loss = running_loss / len(dataloader)
	
	return avg_loss


def validate(model, dataloader, device, args, logger=None):
	"""
	验证模型

	参数:
		model: 模型
		dataloader: 数据加载器
		device: 设备
		args: 命令行参数
		logger: 日志记录器

	返回:
		评估指标字典
	"""
	model.eval()
	
	# 创建评估指标计算器
	metrics_calc = SegmentationMetrics()
	
	# 创建进度条
	iterator = tqdm(dataloader, desc="Validation") if args.rank == 0 else dataloader
	
	all_metrics = []
	
	with torch.no_grad():
		for batch in iterator:
			# 获取数据
			img = batch['image'].to(device, memory_format=torch.channels_last)
			lab = batch['label'].to(device)
			tier = batch['tier']
			
			# 处理每个tier
			for j, t in enumerate(tier):
				model.set_tier(int(t))
				with autocast(enabled=args.amp):
					pred = model(img[j:j + 1])
				
				# 二值化预测
				pred_binary = (pred > 0.5).float()
				
				# 计算指标
				batch_metrics = metrics_calc.evaluate_all(
					pred_binary.cpu().numpy(),
					lab[j:j + 1].cpu().numpy()
				)
				
				all_metrics.append(batch_metrics)
	
	# 计算平均指标
	avg_metrics = {}
	if all_metrics:
		for key in all_metrics[0].keys():
			avg_metrics[key] = sum(m[key] for m in all_metrics) / len(all_metrics)
	
	# 记录日志
	if logger and args.rank == 0:
		logger.log_info(f"Validation - Dice: {avg_metrics.get('dice', 0):.4f}, IoU: {avg_metrics.get('iou', 0):.4f}")
	
	return avg_metrics


def main_worker(rank, args):
	"""
	分布式训练工作进程

	参数:
		rank: 进程编号
		args: 命令行参数
	"""
	# 初始化分布式环境
	if args.distributed:
		args.rank = rank
		os.environ['MASTER_ADDR'] = 'localhost'
		os.environ['MASTER_PORT'] = '12355'
		dist.init_process_group(
			backend='nccl',
			init_method=args.dist_url,
			world_size=args.world_size,
			rank=rank
		)
		torch.cuda.set_device(rank)
	
	# 设置设备
	device = torch.device(f'cuda:{rank}' if torch.cuda.is_available() else 'cpu')
	
	# 创建日志记录器（仅在主进程中）
	logger = Logger(Path(args.output_dir) / 'logs') if rank == 0 else None
	
	# 加载配置
	config = yaml.safe_load(open(args.config, 'r'))
	
	# 设置随机种子
	torch.manual_seed(args.seed)
	torch.cuda.manual_seed_all(args.seed)
	
	# 创建智能采样管理器
	sampling_manager = None
	if args.smart_sampling:
		sampling_manager = SamplingManager(
			config.get('smart_sampling', {}),
			logger=logger
		)
	
	# 创建采样调度器（兼容旧代码）
	sampling_scheduler = None
	if config.get('smart_sampling', {}).get('enabled', False):
		sampling_scheduler = SamplingScheduler(
			base_tier1=config.get('smart_sampling', {}).get('base_tier1', 10),
			base_tier2=config.get('smart_sampling', {}).get('base_tier2', 30),
			max_tier1=config.get('smart_sampling', {}).get('max_tier1', 20),
			max_tier2=config.get('smart_sampling', {}).get('max_tier2', 60),
			warmup_epochs=config.get('smart_sampling', {}).get('warmup_epochs', 5),
			enable_hard_mining=config.get('smart_sampling', {}).get('enable_hard_mining', True),
			enable_adaptive_density=config.get('smart_sampling', {}).get('enable_adaptive_density', True),
			enable_importance_sampling=config.get('smart_sampling', {}).get('enable_importance_sampling', True),
			logger=logger
		)
	
	# 创建难样本跟踪器
	hard_sample_tracker = HardSampleTracker(
		base_dir=Path(args.output_dir) / 'difficulty_maps',
		logger=logger
	)
	
	# 创建数据集
	train_dataset = LiverVesselDataset(
		args.image_dir,
		args.label_dir,
		tier=args.tier,
		transform=None,
		preprocess=True,
		max_cases=config.get('max_cases'),
		random_sampling=config.get('random_sampling', True),
		enable_smart_sampling=args.smart_sampling,
		sampling_scheduler=sampling_scheduler,
		hard_sample_tracker=hard_sample_tracker,
		logger=logger
	)
	
	val_dataset = LiverVesselDataset(
		args.image_dir,
		args.label_dir,
		tier=args.tier,
		transform=None,
		preprocess=True,
		max_cases=config.get('max_val_cases'),
		random_sampling=False,
		enable_smart_sampling=False,
		logger=logger
	)
	
	# 创建数据采样器（分布式训练）
	train_sampler = DistributedSampler(train_dataset) if args.distributed else None
	val_sampler = DistributedSampler(val_dataset, shuffle=False) if args.distributed else None
	
	# 创建数据加载器
	train_loader = DataLoader(
		train_dataset,
		batch_size=args.batch_size,
		sampler=train_sampler,
		shuffle=train_sampler is None,
		num_workers=args.num_workers,
		pin_memory=True,
		drop_last=True
	)
	
	val_loader = DataLoader(
		val_dataset,
		batch_size=1,
		sampler=val_sampler,
		shuffle=False,
		num_workers=args.num_workers,
		pin_memory=True
	)
	
	# 创建模型
	model = VesselSegmenter(
		in_channels=1,
		out_channels=1,
		ch_params=config.get('ch_params'),
		tier_params=config.get('tier_params')
	).to(device, memory_format=torch.channels_last)
	
	# 分布式模型包装
	if args.distributed:
		model = nn.parallel.DistributedDataParallel(
			model,
			device_ids=[rank],
			output_device=rank,
			find_unused_parameters=True
		)
	
	# 创建损失函数
	criterion = VesselSegmentationLoss(
		num_classes=1,
		vessel_weight=config.get('vessel_weight', 10.0),
		tumor_weight=config.get('tumor_weight', 15.0),
		use_boundary=config.get('use_boundary', True)
	).to(device)
	
	# 创建优化器
	optimizer = build_optimizer(
		model,
		base_lr=args.lr,
		kernel_lr_scale=config.get('optimizer', {}).get('kernel_lr_scale', 0.1),
		weight_decay=config.get('optimizer', {}).get('weight_decay', 1e-5)
	)
	
	# 创建梯度缩放器
	scaler = GradScaler(enabled=args.amp)
	
	# 恢复检查点（如果有）
	start_epoch = 0
	best_dice = 0.0
	if args.resume:
		if os.path.isfile(args.resume):
			checkpoint = torch.load(args.resume, map_location=device)
			start_epoch = checkpoint['epoch'] + 1
			best_dice = checkpoint.get('best_dice', 0.0)
			
			# 加载模型权重
			if 'model' in checkpoint:
				model.load_state_dict(checkpoint['model'])
			elif 'model_state_dict' in checkpoint:
				model.load_state_dict(checkpoint['model_state_dict'])
			
			# 加载优化器状态
			if 'optimizer' in checkpoint:
				optimizer.load_state_dict(checkpoint['optimizer'])
			
			# 加载混合精度状态
			if 'scaler' in checkpoint and args.amp:
				scaler.load_state_dict(checkpoint['scaler'])
			
			if rank == 0:
				logger.log_info(f"Loaded checkpoint '{args.resume}' (epoch {checkpoint['epoch']})")
		else:
			if rank == 0:
				logger.log_warning(f"No checkpoint found at '{args.resume}'")
	
	# 训练循环
	for epoch in range(start_epoch, args.epochs):
		# 设置epoch（分布式采样器）
		if args.distributed:
			train_sampler.set_epoch(epoch)
		
		# 更新采样策略（只在主进程上更新）
		if rank == 0 and sampling_manager and sampling_manager.should_update(epoch):
			# 获取原始模型（如果是DDP）
			local_model = model.module if args.distributed else model
			sampling_manager.update_sampling_strategy(local_model, train_dataset, epoch, device)
		
		# 更新采样调度器（兼容旧代码）
		if sampling_scheduler:
			sampling_scheduler.update(epoch)
			if hasattr(train_dataset, 'sampler') and train_dataset.sampler:
				train_dataset.sampler.set_sampling_params(
					sampling_scheduler.get_tier_sampling_params()
				)
		
		# 训练一个epoch
		train_loss = train_one_epoch(
			model, train_loader, criterion, optimizer, scaler, device, epoch, args, logger
		)
		
		# 记录训练损失
		if rank == 0:
			logger.log_info(f"Epoch {epoch} - Train Loss: {train_loss:.4f}")
			logger.log_metrics({'train/loss': train_loss}, epoch)
		
		# 验证
		if epoch % args.val_interval == 0 or epoch == args.epochs - 1:
			val_metrics = validate(model, val_loader, device, args, logger)
			
			# 记录验证指标
			if rank == 0:
				for key, value in val_metrics.items():
					logger.log_metrics({f'val/{key}': value}, epoch)
				
				# 检查是否是最佳模型
				if val_metrics.get('dice', 0) > best_dice:
					best_dice = val_metrics['dice']
					
					# 保存最佳模型
					if args.distributed:
						model_state = model.module.state_dict()
					else:
						model_state = model.state_dict()
					
					torch.save({
						'epoch': epoch,
						'model_state_dict': model_state,
						'best_dice': best_dice,
						'config': config
					}, Path(args.output_dir) / 'best_model.pt')
					
					logger.log_info(f"New best model saved with Dice: {best_dice:.4f}")
		
		# 保存定期检查点
		if rank == 0 and (epoch % 10 == 0 or epoch == args.epochs - 1):
			# 保存检查点
			if args.distributed:
				model_state = model.module.state_dict()
			else:
				model_state = model.state_dict()
			
			torch.save({
				'epoch': epoch,
				'model_state_dict': model_state,
				'optimizer': optimizer.state_dict(),
				'scaler': scaler.state_dict(),
				'best_dice': best_dice,
				'config': config
			}, Path(args.output_dir) / f'checkpoint_epoch_{epoch}.pt')
			
			logger.log_info(f"Checkpoint saved at epoch {epoch}")
	
	# 训练结束
	if rank == 0:
		logger.log_info(f"Training completed. Best Dice: {best_dice:.4f}")
		
		# 保存最终模型
		if args.distributed:
			model_state = model.module.state_dict()
		else:
			model_state = model.state_dict()
		
		torch.save({
			'epoch': args.epochs - 1,
			'model_state_dict': model_state,
			'best_dice': best_dice,
			'config': config
		}, Path(args.output_dir) / 'final_model.pt')
		
		logger.log_info("Final model saved")
	
	# 清理分布式环境
	if args.distributed:
		dist.destroy_process_group()


def main():
	"""主函数"""
	args = parse_args()
	
	# 创建输出目录
	Path(args.output_dir).mkdir(parents=True, exist_ok=True)
	
	# 分布式训练
	if args.distributed:
		args.world_size = torch.cuda.device_count()
		mp.spawn(main_worker, nprocs=args.world_size, args=(args,))
	else:
		main_worker(0, args)


if __name__ == '__main__':
	main()