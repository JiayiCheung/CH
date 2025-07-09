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
import shutil
from pathlib import Path

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from data import LiverVesselDataset, HardSampleTracker
from models import VesselSegmenter
from losses import VesselSegmentationLoss
from utils import SegmentationMetrics, Visualizer, Logger, SamplingScheduler
from data.sampling_manager import SamplingManager
from scripts.evaluation import EvaluationManager


class ModelAdapter:
	"""模型适配器，统一不同类型模型的接口"""
	
	def __init__(self, model, device, is_distributed=False):
		self.model = model
		self.device = device
		self.is_distributed = is_distributed
		self._tier_features = {}
	
	def set_tier(self, tier):
		"""设置tier"""
		real_model = self.model.module if self.is_distributed else self.model
		if hasattr(real_model, 'set_tier'):
			real_model.set_tier(tier)
	
	def forward(self, x):
		"""前向传播"""
		return self.model(x)
	
	def eval(self):
		"""设置评估模式"""
		self.model.eval()
		return self
	
	def train(self):
		"""设置训练模式"""
		self.model.train()
		return self
	
	def pause_pipeline(self):
		"""暂停流水线（兼容性方法）"""
		pass
	
	def resume_pipeline(self):
		"""恢复流水线（兼容性方法）"""
		pass
	
	def clear_tier_features(self):
		"""清除tier特征"""
		self._tier_features.clear()
		real_model = self.model.module if self.is_distributed else self.model
		if hasattr(real_model, 'tier_features'):
			real_model.tier_features.clear()
	
	@property
	def tier_features(self):
		"""获取tier特征字典"""
		real_model = self.model.module if self.is_distributed else self.model
		if hasattr(real_model, 'tier_features'):
			return real_model.tier_features
		return self._tier_features
	
	@property
	def pipeline(self):
		"""提供pipeline接口兼容性"""
		return self
	
	@property
	def stages(self):
		"""提供stages接口兼容性"""
		
		class MockBackendStage:
			def __init__(self, tier_features):
				self.tier_features = tier_features
				self.device = 'cuda:0'
		
		return [None, None, None, MockBackendStage(self.tier_features)]


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
	"""构建优化器 - 为边缘增强内核设置单独的学习率"""
	edge_kernels, others = [], []
	
	for name, param in model.named_parameters():
		if not param.requires_grad:
			continue
		if name.endswith(".kernels"):
			edge_kernels.append(param)
		else:
			others.append(param)
	
	param_groups = [
		{
			"params": edge_kernels,
			"lr": base_lr * kernel_lr_scale,
			"weight_decay": 0.0
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
	"""训练一个epoch"""
	model.train()
	running_loss = 0.0
	
	iterator = tqdm(dataloader, desc=f"Epoch {epoch}") if args.rank == 0 else dataloader
	
	for batch in iterator:
		img = batch['image'].to(device, memory_format=torch.channels_last_3d)
		lab = batch['label'].to(device)
		tier = batch['tier']
		
		optimizer.zero_grad(set_to_none=True)
		
		# 处理每个tier
		outs = []
		real_model = model.module if args.distributed else model
		
		for j, t in enumerate(tier):
			if hasattr(real_model, 'set_tier'):
				real_model.set_tier(int(t))
			
			with autocast(enabled=args.amp):
				outs.append(model(img[j:j + 1]))
		
		out = torch.cat(outs)
		
		with autocast(enabled=args.amp):
			loss = criterion(out, lab)
		
		scaler.scale(loss).backward()
		scaler.step(optimizer)
		scaler.update()
		
		running_loss += loss.item()
		
		if args.rank == 0 and isinstance(iterator, tqdm):
			iterator.set_postfix(loss=loss.item())
	
	avg_loss = running_loss / len(dataloader)
	return avg_loss


def run_simple_evaluation(model, val_loader, device, args, logger, max_samples=5):
	"""简单评估（降级方案）"""
	model.eval()
	metrics_sum = {}
	total_samples = 0
	
	with torch.no_grad():
		for i, batch in enumerate(val_loader):
			if i >= max_samples:
				break
			
			images = batch['image'].to(device, memory_format=torch.channels_last_3d)
			labels = batch['label'].to(device)
			tiers = batch['tier']
			
			real_model = model.module if args.distributed else model
			
			for j, tier in enumerate(tiers):
				if hasattr(real_model, 'set_tier'):
					real_model.set_tier(int(tier))
				
				output = model(images[j:j + 1])
				
				# 计算简单指标
				pred = (output > 0.5).float()
				target = labels[j:j + 1]
				
				# 计算Dice
				intersection = torch.sum(pred * target)
				union = torch.sum(pred) + torch.sum(target)
				dice = (2.0 * intersection) / (union + 1e-7)
				
				if 'dice' not in metrics_sum:
					metrics_sum['dice'] = 0
				metrics_sum['dice'] += dice.item()
				total_samples += 1
	
	model.train()
	
	if total_samples > 0:
		return {k: v / total_samples for k, v in metrics_sum.items()}
	return {'dice': 0.0}


def setup_evaluation_config(config, output_dir):
	"""设置完整的评估配置"""
	evaluation_config = config.get('evaluation', {})
	
	# 确保所有必要的配置项都存在
	defaults = {
		'eval_full_interval': 10,
		'eval_quick_interval': 2,
		'quick_samples': 5,
		'group_by_tier': True,
		'feature_mmap_enabled': False,  # 默认关闭以避免复杂性
		'feature_mmap_dir': os.path.join(output_dir, 'eval_tier_features'),
		'clear_cache_interval': 3,
		'max_eval_samples': 50,
		'include_advanced_metrics': False,
		'save_predictions': False
	}
	
	for key, default_value in defaults.items():
		if key not in evaluation_config:
			evaluation_config[key] = default_value
	
	return evaluation_config


def main_worker(rank, args):
	"""分布式训练工作进程"""
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
		sampling_params=sampling_scheduler.get_tier_sampling_params() if sampling_scheduler else None,
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
	).to(device, memory_format=torch.channels_last_3d)
	
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
	
	# 创建评估管理器
	evaluation_manager = None
	if rank == 0:
		evaluation_config = setup_evaluation_config(config, args.output_dir)
		try:
			evaluation_manager = EvaluationManager(evaluation_config, logger=logger)
		except Exception as e:
			if logger:
				logger.log_warning(f"Failed to create EvaluationManager: {e}. Using simple evaluation.")
			evaluation_manager = None
	
	def run_evaluation(epoch):
		"""执行评估的优化版本"""
		if evaluation_manager and evaluation_manager.should_evaluate(epoch):
			try:
				# 创建模型适配器
				adapter = ModelAdapter(model, device, args.distributed)
				
				# 执行评估
				val_metrics = evaluation_manager.evaluate(adapter, val_loader, epoch)
				
				if val_metrics:
					# 记录验证指标
					for key, value in val_metrics.items():
						logger.log_metrics({f'val/{key}': value}, epoch)
					return val_metrics
			
			except Exception as e:
				if logger:
					logger.log_warning(f"Advanced evaluation failed at epoch {epoch}: {e}. Using simple evaluation.")
		
		# 降级到简单评估
		if (epoch % args.val_interval == 0 or epoch == args.epochs - 1):
			return run_simple_evaluation(model, val_loader, device, args, logger)
		
		return None
	
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
			try:
				local_model = model.module if args.distributed else model
				sampling_manager.update_sampling_strategy(local_model, train_dataset, epoch, device)
			except Exception as e:
				if logger:
					logger.log_warning(f"Sampling strategy update failed: {e}")
		
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
		if rank == 0:
			val_metrics = run_evaluation(epoch)
			
			if val_metrics:
				current_dice = val_metrics.get('dice', 0)
				if current_dice > best_dice:
					best_dice = current_dice
					
					# 保存最佳模型
					model_state = (model.module if args.distributed else model).state_dict()
					torch.save({
						'epoch': epoch,
						'model_state_dict': model_state,
						'best_dice': best_dice,
						'config': config
					}, Path(args.output_dir) / 'best_model.pt')
					
					logger.log_info(f"New best model saved with Dice: {best_dice:.4f}")
		
		# 保存定期检查点
		if rank == 0 and (epoch % 10 == 0 or epoch == args.epochs - 1):
			model_state = (model.module if args.distributed else model).state_dict()
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
		# 清理评估资源
		if evaluation_manager:
			try:
				if hasattr(evaluation_manager, '_cleanup_evaluation_resources'):
					evaluation_manager._cleanup_evaluation_resources()
				
				# 清理MMap资源
				if (hasattr(evaluation_manager, 'mmap_manager') and
						evaluation_manager.mmap_manager and
						evaluation_manager.config.get('feature_mmap_enabled', False)):
					
					mmap_dir = Path(evaluation_manager.config['feature_mmap_dir'])
					if mmap_dir.exists():
						try:
							shutil.rmtree(mmap_dir)
							logger.log_info("Cleaned up evaluation temporary files")
						except Exception as e:
							logger.log_warning(f"Error cleaning up temp files: {e}")
			
			except Exception as e:
				logger.log_warning(f"Error during evaluation cleanup: {e}")
		
		logger.log_info(f"Training completed. Best Dice: {best_dice:.4f}")
		
		# 保存最终模型
		model_state = (model.module if args.distributed else model).state_dict()
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