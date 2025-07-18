#component_factory.py

import torch
import torch.optim as optim
from pathlib import Path

# 导入模型相关组件
from models import VesselSegmenter
from loss import CombinedLoss
from data.sampling_manager import SamplingManager
from utils import Logger


class ComponentFactory:
	"""组件工厂类，负责统一创建各种组件"""
	
	@staticmethod
	def create_model(config=None):
		"""
		创建VesselSegmenter模型

		参数:
			config: 配置字典

		返回:
			VesselSegmenter实例
		"""
		if config is None:
			config = {}
		
		model_config = config.get('model', {})
		
		model = VesselSegmenter(
			in_channels=model_config.get('in_channels', 1),
			out_channels=model_config.get('out_channels', 1),
			ch_params=model_config.get('ch_params'),
			tier_params=model_config.get('tier_params')
		)
		
		return model
	
	@staticmethod
	def create_loss_function(config=None):
		"""
		创建损失函数

		参数:
			config: 配置字典

		返回:
			VesselSegmentationLoss实例
		"""
		if config is None:
			config = {}
		
		loss_config = config.get('loss', {})
		
		criterion = CombinedLoss(
		)
		
		return criterion
	
	@staticmethod
	def create_optimizer(model, config=None, lr=1e-4):
		"""
		创建优化器，支持分组学习率

		参数:
			model: 模型实例
			config: 配置字典
			lr: 基础学习率

		返回:
			优化器实例
		"""
		if config is None:
			config = {}
		
		optimizer_config = config.get('optimizer', {})
		
		# 分组参数：边缘增强核使用较小学习率
		edge_kernels = []
		other_params = []
		
		for name, param in model.named_parameters():
			if not param.requires_grad:
				continue
			
			if 'kernels' in name.lower() or 'edge' in name.lower():
				edge_kernels.append(param)
			else:
				other_params.append(param)
		
		# 创建参数组
		param_groups = []
		
		if edge_kernels:
			param_groups.append({
				'params': edge_kernels,
				'lr': lr * optimizer_config.get('kernel_lr_scale', 0.1),
				'weight_decay': 0.0
			})
		
		if other_params:
			param_groups.append({
				'params': other_params,
				'lr': lr,
				'weight_decay': optimizer_config.get('weight_decay', 1e-5)
			})
		
		# 选择优化器类型
		optimizer_type = optimizer_config.get('type', 'adamw').lower()
		
		if optimizer_type == 'adamw':
			optimizer = optim.AdamW(param_groups)
		elif optimizer_type == 'adam':
			optimizer = optim.Adam(param_groups)
		elif optimizer_type == 'sgd':
			momentum = optimizer_config.get('momentum', 0.9)
			optimizer = optim.SGD(param_groups, momentum=momentum)
		else:
			raise ValueError(f"Unknown optimizer type: {optimizer_type}")
		
		return optimizer
	
	@staticmethod
	def create_sampling_manager(config=None, logger=None):
		"""
		创建采样管理器

		参数:
			config: 配置字典
			logger: 日志记录器

		返回:
			SamplingManager实例或None
		"""
		if config is None:
			config = {}
		
		sampling_config = config.get('smart_sampling', {})
		
		if not sampling_config.get('enabled', False):
			return None
		
		sampling_manager = SamplingManager(
			config=sampling_config,
			logger=logger
		)
		
		return sampling_manager
	
	@staticmethod
	def create_logger(output_dir, log_level='INFO'):
		"""
		创建日志记录器

		参数:
			output_dir: 输出目录
			log_level: 日志级别

		返回:
			Logger实例
		"""
		log_dir = Path(output_dir) / 'logs'
		log_dir.mkdir(parents=True, exist_ok=True)
		
		logger = Logger(log_dir, level=log_level)
		return logger
	
	@staticmethod
	def create_lr_scheduler(optimizer, config=None, total_epochs=100):
		"""
		创建学习率调度器

		参数:
			optimizer: 优化器
			config: 配置字典
			total_epochs: 总训练轮数

		返回:
			学习率调度器实例
		"""
		if config is None:
			config = {}
		
		scheduler_config = config.get('lr_scheduler', {})
		scheduler_type = scheduler_config.get('type', 'cosine').lower()
		
		if scheduler_type == 'cosine':
			from torch.optim.lr_scheduler import CosineAnnealingLR
			scheduler = CosineAnnealingLR(
				optimizer,
				T_max=total_epochs,
				eta_min=scheduler_config.get('min_lr', 1e-6)
			)
		elif scheduler_type == 'step':
			from torch.optim.lr_scheduler import StepLR
			scheduler = StepLR(
				optimizer,
				step_size=scheduler_config.get('step_size', 30),
				gamma=scheduler_config.get('gamma', 0.1)
			)
		elif scheduler_type == 'plateau':
			from torch.optim.lr_scheduler import ReduceLROnPlateau
			scheduler = ReduceLROnPlateau(
				optimizer,
				mode='max',  # 监控验证指标，越大越好
				factor=scheduler_config.get('factor', 0.5),
				patience=scheduler_config.get('patience', 10),
				verbose=True
			)
		else:
			raise ValueError(f"Unknown scheduler type: {scheduler_type}")
		
		return scheduler
	
	@staticmethod
	def create_device_manager(local_rank=None):
		"""
		创建设备管理器

		参数:
			local_rank: 本地GPU编号

		返回:
			设备对象和相关信息
		"""
		if torch.cuda.is_available():
			if local_rank is not None:
				device = torch.device(f'cuda:{local_rank}')
				torch.cuda.set_device(local_rank)
			else:
				device = torch.device('cuda')
			
			device_info = {
				'device': device,
				'device_type': 'cuda',
				'device_count': torch.cuda.device_count(),
				'device_name': torch.cuda.get_device_name(device),
				'memory_total': torch.cuda.get_device_properties(device).total_memory,
				'amp_supported': True
			}
		else:
			device = torch.device('cpu')
			device_info = {
				'device': device,
				'device_type': 'cpu',
				'device_count': 1,
				'device_name': 'CPU',
				'memory_total': None,
				'amp_supported': False
			}
		
		return device_info
	
	@staticmethod
	def create_checkpoint_manager(output_dir):
		"""
		创建检查点管理器

		参数:
			output_dir: 输出目录

		返回:
			检查点管理器字典
		"""
		checkpoint_dir = Path(output_dir) / 'checkpoints'
		checkpoint_dir.mkdir(parents=True, exist_ok=True)
		
		manager = {
			'checkpoint_dir': checkpoint_dir,
			'best_model_path': checkpoint_dir / 'best_model.pt',
			'latest_model_path': checkpoint_dir / 'latest_model.pt'
		}
		
		return manager
	
	@staticmethod
	def save_checkpoint(checkpoint_manager, model, optimizer, scaler, epoch,
	                    best_dice, config, is_best=False):
		"""
		保存检查点

		参数:
			checkpoint_manager: 检查点管理器
			model: 模型
			optimizer: 优化器
			scaler: 梯度缩放器
			epoch: 当前轮数
			best_dice: 最佳Dice分数
			config: 配置字典
			is_best: 是否为最佳模型
		"""
		checkpoint = {
			'epoch': epoch,
			'model_state_dict': model.state_dict(),
			'optimizer': optimizer.state_dict(),
			'scaler': scaler.state_dict(),
			'best_dice': best_dice,
			'config': config
		}
		
		# 保存最新检查点
		torch.save(checkpoint, checkpoint_manager['latest_model_path'])
		
		# 如果是最佳模型，也保存为best
		if is_best:
			torch.save(checkpoint, checkpoint_manager['best_model_path'])
		
		# 保存定期检查点
		if epoch % 10 == 0:
			epoch_checkpoint_path = checkpoint_manager['checkpoint_dir'] / f'epoch_{epoch}.pt'
			torch.save(checkpoint, epoch_checkpoint_path)
	
	@staticmethod
	def load_checkpoint(checkpoint_path, model, optimizer=None, scaler=None, device=None):
		"""
		加载检查点

		参数:
			checkpoint_path: 检查点路径
			model: 模型
			optimizer: 优化器（可选）
			scaler: 梯度缩放器（可选）
			device: 设备（可选）

		返回:
			加载的信息字典
		"""
		if device is None:
			device = torch.device('cpu')
		
		checkpoint = torch.load(checkpoint_path, map_location=device)
		
		# 加载模型权重
		model.load_state_dict(checkpoint['model_state_dict'])
		
		# 加载优化器状态
		if optimizer is not None and 'optimizer' in checkpoint:
			optimizer.load_state_dict(checkpoint['optimizer'])
		
		# 加载缩放器状态
		if scaler is not None and 'scaler' in checkpoint:
			scaler.load_state_dict(checkpoint['scaler'])
		
		return {
			'epoch': checkpoint.get('epoch', 0),
			'best_dice': checkpoint.get('best_dice', 0.0),
			'config': checkpoint.get('config', {})
		}