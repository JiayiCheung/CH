#!/usr/bin/env python3
"""
集成化分布式训练系统 - 肝脏血管分割
统一管理所有distributed目录下的功能组件
"""

import os
import sys
import time
import argparse
import yaml
import logging
import signal
# 在现有导入中添加
from tqdm import tqdm
import traceback
import copy
from pathlib import Path
from typing import Dict, Optional, Any, List, Tuple
from dataclasses import dataclass

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.utils.data import DataLoader, DistributedSampler

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# 核心组件导入
from data import LiverVesselDataset
from models import create_vessel_segmenter
from loss.combined_loss import CombinedLoss

# 分布式组件导入
from scripts.distributed.cross_node_pipeline import create_pipeline
from scripts.distributed.node_communicator import NodeCommunicator
from scripts.distributed.stages import create_pipeline_stages
from scripts.distributed.evaluation import EvaluationManager

# 可靠性组件导入 - 使用正确的类名
from scripts.distributed.reliability.checkpoint_manager import DistributedCheckpointManager
from scripts.distributed.reliability.parameter_synchronizer import ParameterSyncManager
from scripts.distributed.reliability.communication_reliability import EnhancedNodeCommunicator
from scripts.distributed.reliability.gradient_backoff import GradientBackoffHandler

# 监控组件导入 - 使用正确的类名
from scripts.distributed.monitoring.error_recovery import ErrorRecoverySystem

# 工具组件导入
from utils.component_factory import ComponentFactory
from utils import Logger

# 配置日志
logging.basicConfig(
	level=logging.INFO,
	format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class TrainingState:
	"""训练状态跟踪"""
	epoch: int = 0
	batch: int = 0
	global_step: int = 0
	best_dice: float = 0.0
	best_loss: float = float('inf')
	training_time: float = 0.0
	last_checkpoint_time: float = 0.0


class ModelMerger:
	"""模型合并器 - 将7个GPU的checkpoints合并为完整模型"""
	
	def __init__(self, world_size: int, model_config: Dict, export_config: Dict, device: torch.device):
		self.world_size = world_size
		self.model_config = model_config
		self.export_config = export_config
		self.device = device
		self.logger = logging.getLogger(__name__)
	
	def merge_pipeline_checkpoints(self, checkpoint_paths: Dict[int, str]) -> torch.nn.Module:
		"""
		从7个stage的checkpoints合并为完整VesselSegmenter

		Args:
			checkpoint_paths: {rank: checkpoint_path} 映射

		Returns:
			完整的VesselSegmenter模型
		"""
		self.logger.info("开始合并流水线模型...")
		
		# 1. 创建完整模型实例
		full_model = create_vessel_segmenter(self.model_config)
		full_model.to(self.device)
		
		# 2. 逐个加载各stage的参数
		stage_mapping = {
			0: 'preprocessor',
			1: 'tier_sampler',
			2: 'ch_branch',
			3: 'spatial_branch',
			4: 'feature_fusion',
			5: 'multiscale_fusion',
			6: 'seg_head'
		}
		
		for rank, checkpoint_path in checkpoint_paths.items():
			if not Path(checkpoint_path).exists():
				self.logger.warning(f"Checkpoint不存在: {checkpoint_path}")
				continue
			
			try:
				checkpoint = torch.load(checkpoint_path, map_location=self.device)
				stage_state_dict = checkpoint.get('model_state_dict', {})
				
				stage_name = stage_mapping.get(rank)
				if stage_name and hasattr(full_model, stage_name):
					# 加载对应stage的参数
					stage_module = getattr(full_model, stage_name)
					if stage_module is not None:
						stage_module.load_state_dict(stage_state_dict, strict=False)
						self.logger.info(f"成功加载rank {rank} ({stage_name}) 的参数")
			
			except Exception as e:
				self.logger.error(f"读取checkpoint失败 {checkpoint_path}: {e}")
				continue
		
		# 3. 验证模型完整性
		if self._validate_merged_model(full_model):
			self.logger.info("模型合并完成，验证通过")
		else:
			self.logger.warning("模型合并完成，但验证发现问题")
		
		return full_model
	
	def _validate_merged_model(self, model: torch.nn.Module) -> bool:
		"""验证合并后模型的完整性"""
		try:
			# 检查模型是否可以前向传播
			model.eval()
			dummy_input = torch.randn(1, 1, 64, 64, 64, device=self.device)
			
			with torch.no_grad():
				output = model(dummy_input)
			
			if output is not None and output.shape[0] == 1:
				return True
			else:
				self.logger.error("模型输出格式不正确")
				return False
		
		except Exception as e:
			self.logger.error(f"模型验证失败: {e}")
			return False
	
	def export_for_deployment(self, model: torch.nn.Module, export_path: str) -> bool:
		"""
		导出为部署用的模型文件

		Args:
			model: 要导出的模型
			export_path: 导出路径

		Returns:
			是否导出成功
		"""
		try:
			export_path = Path(export_path)
			export_path.parent.mkdir(parents=True, exist_ok=True)
			
			# 导出PyTorch模型
			formats = self.export_config.get('export_formats', ['pth'])
			
			if 'pth' in formats:
				pth_path = export_path.with_suffix('.pth')
				torch.save({
					'model_state_dict': model.state_dict(),
					'model_config': self.model_config,
					'export_time': time.time(),
					'pytorch_version': torch.__version__
				}, pth_path)
				self.logger.info(f"PyTorch模型已导出: {pth_path}")
			
			if 'onnx' in formats:
				import torch.onnx
				onnx_path = export_path.with_suffix('.onnx')
				dummy_input = torch.randn(1, 1, 64, 64, 64, device=self.device)
				
				torch.onnx.export(
					model,
					dummy_input,
					onnx_path,
					export_params=True,
					opset_version=11,
					do_constant_folding=True,
					input_names=['input'],
					output_names=['output']
				)
				self.logger.info(f"ONNX模型已导出: {onnx_path}")
			
			return True
		
		except Exception as e:
			self.logger.error(f"模型导出失败: {e}")
			return False
	
	def collect_all_checkpoints(self, checkpoint_dir: str, epoch: int) -> Dict[int, str]:
		"""收集所有rank的checkpoint路径"""
		checkpoint_paths = {}
		
		for rank in range(self.world_size):
			checkpoint_name = f"checkpoint_epoch_{epoch}_rank_{rank}.pt"
			checkpoint_path = Path(checkpoint_dir) / checkpoint_name
			
			if checkpoint_path.exists():
				checkpoint_paths[rank] = str(checkpoint_path)
			else:
				self.logger.warning(f"Checkpoint不存在: {checkpoint_path}")
		
		return checkpoint_paths


class IntegratedPipelineTrainer:
	"""集成化流水线训练器 - 统一管理所有分布式组件"""
	
	def __init__(self, rank: int, world_size: int, local_rank: int, config: Dict, args):
		"""
		初始化集成化流水线训练器

		Args:
			rank: 全局进程排名
			world_size: 总进程数
			local_rank: 本地GPU编号
			config: 配置字典
			args: 命令行参数
		"""
		self.rank = rank
		self.world_size = world_size
		self.local_rank = local_rank
		self.args = args
		self.device = torch.cuda.current_device()
		
		# 配置验证和默认值合并
		self.config = self._validate_and_merge_config(config)
		
		# 训练状态
		self.training_state = TrainingState()
		
		# 设置流水线配置
		self.pipeline_config = {
			0: {'stage': 'preprocessing', 'next_ranks': [1], 'prev_ranks': []},
			1: {'stage': 'patch_scheduling', 'next_ranks': [2, 3], 'prev_ranks': [0]},
			2: {'stage': 'ch_branch', 'next_ranks': [4], 'prev_ranks': [1]},
			3: {'stage': 'spatial_branch', 'next_ranks': [4], 'prev_ranks': [1]},
			4: {'stage': 'feature_fusion', 'next_ranks': [5], 'prev_ranks': [2, 3]},
			5: {'stage': 'multiscale_fusion', 'next_ranks': [6], 'prev_ranks': [4]},
			6: {'stage': 'segmentation_head', 'next_ranks': [], 'prev_ranks': [5]},
			7: {'stage': 'dummy_stage', 'next_ranks': [], 'prev_ranks': []}
		}
		
		self.stage_config = self.pipeline_config[rank]
		self.next_ranks = self.stage_config['next_ranks']
		self.prev_ranks = self.stage_config['prev_ranks']
		
		# 初始化所有组件
		self._setup_basic_components()
		self._setup_enhanced_components()
		self._setup_evaluation_and_logging()
		self._setup_model_merger()
		
		logger.info(f"Rank {rank}: IntegratedPipelineTrainer初始化完成")
	
	def _validate_and_merge_config(self, config: Dict) -> Dict:
		"""验证配置并与默认值合并"""
		
		# 提供默认的分布式配置
		default_distributed_config = {
			'communication': {
				'enable_reliability': True,
				'max_chunk_size': 50 * 1024 * 1024,  # 50MB
				'compression_level': 6,
				'send_timeout': 30.0,
				'recv_timeout': 60.0,
				'enable_ack': True,
				'ack_timeout': 5.0,
				'node_count': 2
			},
			'parameter_sync': {
				'enable_sync': True,
				'sync_frequency': 'batch',
				'enable_version_control': True
			},
			'gradient_reliability': {
				'enable_gradient_recovery': True,
				'max_recovery_attempts': 3,
				'enable_double_precision_fallback': True,
				'enable_gradient_checkpointing': False
			},
			'health_monitoring': {
				'enable_monitoring': True,
				'check_interval': 30,
				'gpu_temp_threshold': 85,
				'memory_threshold': 0.9,
				'cpu_threshold': 0.8,
				'network_latency_threshold': 100
			},
			'checkpoint': {
				'enable_compression': True,
				'compression_level': 6,
				'save_metadata': True,
				'checkpoint_version': "1.0.0"
			}
		}
		
		default_eval_config = {
			'eval_full_interval': 10,
			'eval_quick_interval': 2,
			'quick_samples': 5,
			'group_by_tier': True,
			'feature_mmap_enabled': True,
			'feature_mmap_dir': 'eval_tier_features',
			'clear_cache_interval': 3
		}
		
		default_logging_config = {
			'enable_tensorboard': True,
			'log_dir': './logs',
			'experiment_name': 'liver_vessel_pipeline',
			'log_level': 'INFO',
			'log_interval': 10
		}
		
		default_export_config = {
			'enable_export': True,
			'export_formats': ['pth'],
			'validate_merged_model': True,
			'export_dir': './exported_models'
		}
		
		# 深度合并配置
		merged_config = copy.deepcopy(config)
		
		if 'distributed' not in merged_config:
			merged_config['distributed'] = {}
		merged_config['distributed'] = self._deep_merge(
			default_distributed_config,
			merged_config['distributed']
		)
		
		if 'evaluation' not in merged_config:
			merged_config['evaluation'] = default_eval_config
		else:
			merged_config['evaluation'] = self._deep_merge(
				default_eval_config,
				merged_config['evaluation']
			)
		
		if 'logging' not in merged_config:
			merged_config['logging'] = default_logging_config
		else:
			merged_config['logging'] = self._deep_merge(
				default_logging_config,
				merged_config['logging']
			)
		
		if 'model_export' not in merged_config:
			merged_config['model_export'] = default_export_config
		else:
			merged_config['model_export'] = self._deep_merge(
				default_export_config,
				merged_config['model_export']
			)
		
		# 验证关键配置项
		self._validate_config(merged_config)
		
		return merged_config
	
	def _deep_merge(self, default_dict: Dict, user_dict: Dict) -> Dict:
		"""递归合并字典"""
		result = copy.deepcopy(default_dict)
		
		for key, value in user_dict.items():
			if key in result and isinstance(result[key], dict) and isinstance(value, dict):
				result[key] = self._deep_merge(result[key], value)
			else:
				result[key] = value
		
		return result
	
	def _validate_config(self, config: Dict):
		"""验证配置文件的完整性"""
		required_sections = ['model', 'data', 'loss']
		for section in required_sections:
			if section not in config:
				raise ValueError(f"配置文件缺少必需的section: {section}")
		
		# 验证distributed配置的合理性
		distributed_config = config.get('distributed', {})
		comm_config = distributed_config.get('communication', {})
		
		send_timeout = comm_config.get('send_timeout', 30)
		if send_timeout < 10:
			logger.warning("send_timeout设置过小，可能导致传输失败")
		
		recv_timeout = comm_config.get('recv_timeout', 60)
		if recv_timeout < send_timeout:
			logger.warning("recv_timeout应该大于send_timeout")
	
	def _setup_basic_components(self):
		"""设置基础组件"""
		
		# 创建节点通信器并升级为增强版本
		from scripts.distributed.reliability.communication_reliability import upgrade_node_communicator
		
		comm_config = self.config.get('distributed', {}).get('communication', {})
		node_count = comm_config.get('node_count', 2)
		
		# 创建原始通信器
		original_node_comm = NodeCommunicator(
			world_size=self.world_size,
			rank=self.rank,
			local_rank=self.local_rank,
			node_rank=self.rank // (self.world_size // node_count),
			node_count=node_count
		)
		
		# 升级为增强版本
		self.node_comm = upgrade_node_communicator(original_node_comm)
		
		# 记录通信器版本
		logger.info(f"🔗 通信器初始化完成 - Rank {self.rank}:")
		logger.info(f"  - 可靠传输: ✅ 启用")
		logger.info(f"  - 复杂数据类型: ✅ 支持")
		logger.info(f"  - 回退模式: ✅ 支持")
		
		# 创建完整模型（用于提取组件）
		self.full_model = create_vessel_segmenter(self.config)
		
		# 创建当前rank的stage
		self.stage = self._create_stage(self.full_model)
		
		# 收集所有参数（用于优化器）
		self.all_params = list(self.full_model.parameters())
		
		# 创建数据加载器（仅rank 0）
		self._setup_dataloader()
		
		# 创建训练组件
		self._setup_training_components()
	
	def _create_stage(self, full_model):
		"""创建当前rank对应的stage"""
		# 使用工厂函数创建stage
		stage_dict = create_pipeline_stages(
			config=self.config,
			node_comm=self.node_comm
		)
		return stage_dict.get(self.stage_config['stage'])
	
	def _setup_dataloader(self):
		"""设置数据加载器"""
		if self.rank == 0:
			logger.info("📊 开始创建数据加载器...")
			logger.info("🔍 正在创建训练数据集，这可能需要几分钟...")
			
			data_config = self.config.get('data', {})
			train_dataset = LiverVesselDataset(
				image_dir=self.args.image_dir,
				label_dir=self.args.label_dir,
				max_cases=data_config.get('max_cases', 5),
				random_sampling=data_config.get('random_sampling', True),
				enable_smart_sampling=data_config.get('enable_smart_sampling', True),
				
			)
			
			self.train_loader = DataLoader(
				train_dataset,
				batch_size=self.args.batch_size,
				shuffle=True,
				num_workers=self.args.num_workers,
				pin_memory=True,
				drop_last=True
			)
			
			# 创建验证数据加载器
			val_dataset = LiverVesselDataset(
				image_dir=self.args.image_dir,
				label_dir=self.args.label_dir,
				max_cases=data_config.get('val_max_cases', 20),
				random_sampling=False,
				enable_smart_sampling=False,
			)
			
			self.val_loader = DataLoader(
				val_dataset,
				batch_size=1,  # 验证时使用batch_size=1
				shuffle=False,
				num_workers=2,
				pin_memory=True
			)
			
			self.labels_cache = {}  # 缓存labels用于rank 6
		else:
			self.train_loader = None
			self.val_loader = None
	
	def _setup_training_components(self):
		"""设置训练组件"""
		# 创建优化器
		self.optimizer = torch.optim.AdamW(
			self.all_params,
			lr=self.args.lr,
			weight_decay=1e-4
		)
		
		# 创建损失函数（仅rank 6）
		if self.rank == 6:
			self.loss_fn = CombinedLoss()
		
		# 学习率调度器
		self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
			self.optimizer, T_max=self.args.epochs
		)
		
		# 混合精度
		self.scaler = torch.cuda.amp.GradScaler() if self.args.amp else None




	def _setup_enhanced_components(self):
		"""设置增强组件"""
		distributed_config = self.config.get('distributed', {})
		
		# 1. 通信可靠性增强
		comm_config = distributed_config.get('communication', {})
		if comm_config.get('enable_reliability', True):
			self.enhanced_comm = EnhancedNodeCommunicator(self.node_comm)
			# 替换原有通信器
			self.node_comm = self.enhanced_comm
		
		# 2. 参数同步管理
		sync_config = distributed_config.get('parameter_sync', {})
		if sync_config.get('enable_sync', True):
			self.param_sync = ParameterSyncManager(
				model=self.stage,
				rank=self.rank,
				world_size=self.world_size,
				device=self.device
			)
		else:
			self.param_sync = None
		
		# 3. 梯度可靠性管理
		grad_config = distributed_config.get('gradient_reliability', {})
		if grad_config.get('enable_gradient_recovery', True):
			self.gradient_manager = GradientBackoffHandler(
				model=self.stage,
				save_dir=self.args.output_dir
			)
		else:
			self.gradient_manager = None
		
		# 4. 系统健康监控
		health_config = distributed_config.get('health_monitoring', {})
		if health_config.get('enable_monitoring', True):
			self.health_monitor = ErrorRecoverySystem(
				rank=self.rank,
				device=self.device
			)
		else:
			self.health_monitor = None
		
		# 5. 分布式检查点管理器
		checkpoint_config = distributed_config.get('checkpoint', {})
		self.distributed_checkpoint = DistributedCheckpointManager(
			rank=self.rank,
			world_size=self.world_size,
			base_dir=self.args.output_dir
		)
	
	
	
	
	def _setup_evaluation_and_logging(self):
		"""设置评估和日志组件"""
		
		# 评估管理器（所有rank都需要，但只有rank 6真正执行）
		eval_config = self.config.get('evaluation', {})
		self.evaluator = EvaluationManager(
			config=eval_config,
			logger=None  # 暂时不传logger，后面会设置
		)
		
		# 日志记录器（只在rank 0初始化）
		if self.rank == 0:
			log_config = self.config.get('logging', {})
			
			self.logger = Logger(
				log_dir=log_config.get('log_dir', self.args.output_dir),
				experiment_name=log_config.get('experiment_name', 'liver_vessel_pipeline')
			)
			
			# 将logger传递给评估器
			self.evaluator.logger = self.logger
			
			logger.info("Tensorboard日志器初始化成功")
		else:
			self.logger = None
	
	def _setup_model_merger(self):
		"""设置模型合并器"""
		export_config = self.config.get('model_export', {})
		self.model_merger = ModelMerger(
			world_size=self.world_size,
			model_config=self.config.get('model', {}),
			export_config=export_config,
			device=self.device
		)
	
	def train_epoch(self, epoch: int) -> Dict[str, float]:
		"""
		训练一个epoch

		Args:
			epoch: 当前epoch

		Returns:
			训练指标字典
		"""
		self.training_state.epoch = epoch
		
		# Epoch开始前的同步和检查
		self._pre_epoch_sync_and_check(epoch)
		
		# 执行增强训练循环
		epoch_metrics = self._execute_enhanced_training_loop(epoch)
		
		# Epoch结束后的同步和保存
		self._post_epoch_sync_and_save(epoch, epoch_metrics)
		
		return epoch_metrics
	
	def _pre_epoch_sync_and_check(self, epoch: int):
		"""Epoch开始前的同步和检查"""
		
		# 参数同步
		if self.param_sync:
			if not self.param_sync.sync_on_epoch_start():
				logger.warning(f"Epoch {epoch} 开始参数同步失败")
		
		# 健康检查
		if self.health_monitor:
			self.health_monitor.start_monitoring()
			if not self.health_monitor.check_system_health():
				logger.warning(f"Rank {self.rank} 健康检查发现问题")
		
		
		
		# 全局同步点
		dist.barrier()
		
		if self.rank == 0:
			logger.info(f"Epoch {epoch} 开始训练")
	
	def _execute_enhanced_training_loop(self, epoch: int) -> Dict[str, float]:
		"""执行增强版训练循环 - 带进度条"""
		
		if self.rank == 7:
			# DummyStage什么都不做，只是占用GPU
			self.stage.train()
			time.sleep(1)  # 模拟一些工作
			return {
				'loss': 0.0,
				'epoch_time': 1.0,
				'batch_count': 0
			}
		
		
		
		# 设置为训练模式
		self.stage.train()
		
		total_loss = 0.0
		batch_count = 0
		epoch_start_time = time.time()
		
		# ✅ 添加Batch级别进度条
		if self.rank == 0 and self.train_loader:
			# Rank 0: 显示真实的数据加载进度
			batch_pbar = tqdm(
				enumerate(self.train_loader),
				desc=f"📦 Epoch {epoch} Batches",
				total=len(self.train_loader),
				leave=False,
				ncols=120,
				colour='blue'
			)
			data_iterator = batch_pbar
		else:
			# 其他rank: 估算batch数量
			estimated_batches = len(self.train_loader) if hasattr(self, 'train_loader') and self.train_loader else 100
			batch_pbar = tqdm(
				enumerate([(i, None) for i in range(estimated_batches)]),
				desc=f"🔧 Rank {self.rank} Epoch {epoch}",
				total=estimated_batches,
				leave=False,
				ncols=100,
				colour='yellow'
			)
			data_iterator = batch_pbar
		
		try:
			for batch_idx, batch in data_iterator:
				batch_start_time = time.time()
				
				try:
					# 批次开始前的检查和同步
					if self.health_monitor:
						if not self.health_monitor.check_system_health():
							self._handle_health_issues()
					
					if self.param_sync:
						self.param_sync.sync_on_batch_start()
					
					# 执行批次训练（流水线处理）
					loss = self._process_batch_with_reliability(batch, batch_idx)
					
					if loss is not None:
						# 梯度可靠性处理
						if self.gradient_manager:
							if not self.gradient_manager.compute_gradients_with_backoff(loss, epoch, batch_idx):
								logger.warning(f"Batch {batch_idx} 梯度处理失败，跳过")
								continue
						else:
							# 标准的反向传播和优化
							if self.rank == 6:  # 只有最后一个rank计算loss
								if self.scaler:
									self.scaler.scale(loss).backward()
									self.scaler.step(self.optimizer)
									self.scaler.update()
								else:
									loss.backward()
									torch.nn.utils.clip_grad_norm_(self.all_params, max_norm=1.0)
									self.optimizer.step()
								
								self.optimizer.zero_grad()
						
						# 批次结束后的同步
						if self.param_sync:
							self.param_sync.sync_on_batch_end()
						
						# 记录批次指标到健康监控
						if self.health_monitor:
							self.health_monitor.record_batch_metrics(
								loss=loss.item() if loss is not None else 0.0,
								model=self.stage,
								optimizer=self.optimizer,
								epoch=epoch,
								batch_idx=batch_idx
							)
						
						# 统计
						if self.rank == 6:
							total_loss += loss.item()
							batch_count += 1
					
					# ✅ 更新Batch进度条
					batch_time = time.time() - batch_start_time
					
					if self.rank == 0:
						# Rank 0 显示详细信息
						batch_pbar.set_postfix({
							'Loss': f'{loss.item():.4f}' if loss else 'N/A',
							'Avg': f'{total_loss / max(batch_count, 1):.4f}',
							'Time': f'{batch_time:.2f}s',
							'Mem': f'{torch.cuda.memory_allocated() / 1e9:.1f}GB'
						})
					else:
						# 其他rank显示基本信息
						batch_pbar.set_postfix({
							'Stage': self.stage_config['stage'][:8],
							'Time': f'{batch_time:.2f}s'
						})
					
					# 保持原有的定期日志（减少频率避免刷屏）
					if batch_idx % (self.args.log_interval * 5) == 0 and self.rank == 0:
						batch_pbar.write(
							f"📊 Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item() if loss else 'N/A':.6f}")
					
					self.training_state.batch = batch_idx
					self.training_state.global_step += 1
					
					# 检查是否到达epoch结束（通过特殊信号）
					if batch_idx > 0 and batch_idx % 100 == 0:
						# 检查是否收到结束信号
						if self._check_epoch_end_signal():
							break
				
				except Exception as e:
					# ✅ 进度条中显示错误
					batch_pbar.write(f"❌ Rank {self.rank} Batch {batch_idx} 错误: {e}")
					self._handle_training_error(e, batch_idx)
					continue
		
		finally:
			# ✅ 确保关闭进度条
			if hasattr(batch_pbar, 'close'):
				batch_pbar.close()
		
		# 计算epoch指标
		epoch_time = time.time() - epoch_start_time
		self.training_state.training_time += epoch_time
		
		if self.rank == 6 and batch_count > 0:
			avg_loss = total_loss / batch_count
		else:
			avg_loss = 0.0
		
		return {
			'loss': avg_loss,
			'epoch_time': epoch_time,
			'batch_count': batch_count
		}
	
	
	
	
	def _process_batch_with_reliability(self, batch, batch_idx: int):
		"""带可靠性保障的批次处理"""
		try:
			if self.rank == 7:
				# DummyStage不参与任何实际处理
				self.stage.process()  # 调用DummyStage的process方法
				return None
			
			
			
			if self.rank == 0:
				# Rank 0: 数据预处理
				if batch is None:
					return None
				
				# 执行预处理
				processed_data = self.stage.process(batch)
				
				# 发送到下一个stage
				for next_rank in self.next_ranks:
					self.node_comm.send_tensor(processed_data, next_rank)
				
				return None
			
			elif self.rank == 6:
				# Rank 6: 分割头和损失计算
				# 从rank 5接收数据
				final_features = self.node_comm.recv_tensor(
					src_rank=self.prev_ranks[0],
					device=self.device
				)
				
				if final_features is None:
					return None
				
				# 执行分割头
				predictions = self.stage.process(final_features)
				
				# 计算损失（需要获取labels）
				if self.loss_fn:
					# 从rank 0获取对应的labels
					labels = self._receive_labels_from_rank0(batch_idx)
					if labels is not None:
						loss = self.loss_fn(predictions, labels)
						return loss
				
				return None
			
			else:
				# 中间ranks: 接收处理发送
				# 从前一个rank接收数据
				input_data = None
				for prev_rank in self.prev_ranks:
					try:
						data = self.node_comm.recv_tensor(
							src_rank=prev_rank,
							device=self.device
						)
						if input_data is None:
							input_data = data
						else:
							# 对于rank 4，需要融合来自rank 2和3的特征
							input_data = self._fuse_features(input_data, data)
					except Exception as e:
						logger.warning(f"Rank {self.rank} 接收数据失败: {e}")
						continue
				
				if input_data is None:
					return None
				
				# 处理数据
				output_data = self.stage.process(input_data)
				
				# 发送到下一个rank
				for next_rank in self.next_ranks:
					self.node_comm.send_tensor(output_data, next_rank)
				
				return None
		
		except Exception as e:
			logger.error(f"Rank {self.rank} 批次处理失败: {e}")
			return None
	
	def _fuse_features(self, feature1, feature2):
		"""融合来自不同分支的特征"""
		if feature1 is None:
			return feature2
		if feature2 is None:
			return feature1
		
		# 简单的特征融合：相加
		try:
			if feature1.shape == feature2.shape:
				return feature1 + feature2
			else:
				# 如果形状不匹配，尝试调整
				min_dim = min(feature1.shape[-1], feature2.shape[-1])
				return feature1[..., :min_dim] + feature2[..., :min_dim]
		except Exception as e:
			logger.warning(f"特征融合失败: {e}")
			return feature1
	
	def _receive_labels_from_rank0(self, batch_idx: int):
		"""从rank 0接收对应batch的labels"""
		try:
			# 通过特殊的通信tag接收labels
			labels = self.node_comm.recv_tensor(
				src_rank=0,
				tag=1000 + batch_idx,
				device=self.device
			)
			return labels
		except Exception as e:
			logger.warning(f"接收labels失败: {e}")
			return None
	
	def _check_epoch_end_signal(self) -> bool:
		"""检查是否收到epoch结束信号"""
		try:
			# 检查是否有结束信号
			if self.rank != 0:
				signal_tensor = torch.zeros(1, dtype=torch.long, device=self.device)
				dist.recv(signal_tensor, src=0, tag=9999)
				return signal_tensor.item() == -1
			return False
		except:
			return False
	
	def _handle_health_issues(self):
		"""处理健康问题"""
		logger.warning(f"Rank {self.rank} 检测到健康问题，尝试恢复...")
		
		# 简单的恢复策略：清理GPU内存
		if torch.cuda.is_available():
			torch.cuda.empty_cache()
			torch.cuda.synchronize()
		
		time.sleep(1)  # 短暂等待
	
	def _handle_training_error(self, error: Exception, batch_idx: int):
		"""处理训练错误"""
		logger.error(f"Rank {self.rank} Batch {batch_idx} 训练错误: {error}")
		
		# 根据错误类型进行不同的处理
		if "CUDA out of memory" in str(error):
			torch.cuda.empty_cache()
			logger.info("GPU内存不足，已清理缓存")
		elif "communication" in str(error).lower():
			logger.info("通信错误，等待重试")
			time.sleep(2)
		else:
			logger.info("未知错误，跳过该batch")
	
	def _post_epoch_sync_and_save(self, epoch: int, epoch_metrics: Dict[str, float]):
		"""Epoch结束后的同步和保存"""
		
		# 参数同步
		if self.param_sync:
			if not self.param_sync.sync_on_epoch_end():
				logger.warning(f"Epoch {epoch} 结束参数同步失败")
		
		# 全局同步
		dist.barrier()
		
		# 保存检查点
		if epoch % self.args.save_interval == 0:
			self._save_distributed_checkpoint(epoch, epoch_metrics)
		
		# 更新学习率
		self.scheduler.step()
		
		# 更新最佳指标
		if self.rank == 6:
			current_loss = epoch_metrics.get('loss', float('inf'))
			if current_loss < self.training_state.best_loss:
				self.training_state.best_loss = current_loss
	
	def _save_distributed_checkpoint(self, epoch: int, epoch_metrics: Dict[str, float]):
		"""保存分布式检查点"""
		try:
			success = self.distributed_checkpoint.save_distributed_checkpoint(
				model=self.stage,
				optimizer=self.optimizer,
				scheduler=self.scheduler,
				epoch=epoch,
				batch_idx=self.training_state.batch,
				global_step=self.training_state.global_step,
				train_loss=epoch_metrics.get('loss', 0.0),
				val_loss=None,  # 验证loss在验证时设置
				extra_data={
					'config': self.config,
					'args': vars(self.args),
					'training_state': self.training_state.__dict__
				}
			)
			
			if success and self.rank == 0:
				logger.info(f"Epoch {epoch} 检查点保存成功")
		
		except Exception as e:
			logger.error(f"Rank {self.rank} 检查点保存失败: {e}")
	
	def validate_epoch(self, epoch: int) -> float:
		"""
		验证一个epoch

		Args:
			epoch: 当前epoch

		Returns:
			验证损失
		"""
		if self.rank != 6:  # 只有最后一个rank做验证
			return 0.0
		
		if not hasattr(self, 'val_loader') or self.val_loader is None:
			return 0.0
		
		# 设置为评估模式
		self.stage.eval()
		
		try:
			# 使用评估管理器进行验证
			val_metrics = self.evaluator.evaluate(self, self.val_loader, epoch)
			
			if val_metrics:
				# 记录验证指标到tensorboard
				self._log_validation_metrics(val_metrics, epoch)
				
				# 更新最佳dice分数
				dice_score = val_metrics.get('dice_score', 0.0)
				if dice_score > self.training_state.best_dice:
					self.training_state.best_dice = dice_score
				
				return val_metrics.get('dice_score', 0.0)
		
		except Exception as e:
			logger.error(f"验证过程出错: {e}")
		
		finally:
			# 恢复训练模式
			self.stage.train()
		
		return 0.0
	
	def _log_validation_metrics(self, metrics: Dict[str, float], epoch: int):
		"""记录验证指标到tensorboard"""
		if self.logger is None:
			return
		
		try:
			# 核心分割指标
			self.logger.log_scalar('validation/dice_score', metrics.get('dice_score', 0), epoch)
			self.logger.log_scalar('validation/iou_score', metrics.get('iou_score', 0), epoch)
			self.logger.log_scalar('validation/precision', metrics.get('precision', 0), epoch)
			self.logger.log_scalar('validation/recall', metrics.get('recall', 0), epoch)
			
			# Hausdorff距离
			if 'hausdorff_distance' in metrics:
				self.logger.log_scalar('validation/hausdorff_distance', metrics['hausdorff_distance'], epoch)
			
			# 分类别指标
			if 'vessel_dice' in metrics:
				self.logger.log_scalar('validation/vessel_dice', metrics['vessel_dice'], epoch)
			if 'tumor_dice' in metrics:
				self.logger.log_scalar('validation/tumor_dice', metrics['tumor_dice'], epoch)
			
			# Tier级别指标
			for tier in [0, 1, 2]:
				tier_key = f'tier_{tier}_dice'
				if tier_key in metrics:
					self.logger.log_scalar(f'validation/{tier_key}', metrics[tier_key], epoch)
			
			# 训练损失
			if hasattr(self, '_last_train_loss'):
				self.logger.log_scalar('training/loss', self._last_train_loss, epoch)
			
			# 学习率
			current_lr = self.scheduler.get_last_lr()[0]
			self.logger.log_scalar('training/learning_rate', current_lr, epoch)
			
			logger.info(f"验证指标已记录到tensorboard: Dice={metrics.get('dice_score', 0):.4f}")
		
		except Exception as e:
			logger.error(f"记录验证指标失败: {e}")
	
	def finalize_training(self):
		"""训练结束后的模型合并和导出"""
		
		if self.rank == 0:  # 只在主rank执行
			logger.info("开始最终模型合并和导出...")
			
			try:
				# 收集所有rank的最新checkpoints
				checkpoint_dir = Path(self.args.output_dir) / 'checkpoints'
				
				# 找到最新的epoch
				latest_epoch = self._find_latest_epoch(checkpoint_dir)
				if latest_epoch is None:
					logger.error("未找到有效的checkpoint文件")
					return
				
				all_checkpoints = self.model_merger.collect_all_checkpoints(
					str(checkpoint_dir),
					latest_epoch
				)
				
				if len(all_checkpoints) < self.world_size:
					logger.warning(f"只收集到 {len(all_checkpoints)}/{self.world_size} 个checkpoints")
				
				# 合并为完整模型
				merged_model = self.model_merger.merge_pipeline_checkpoints(all_checkpoints)
				
				# 导出为部署用模型
				export_config = self.config.get('model_export', {})
				export_dir = Path(export_config.get('export_dir', './exported_models'))
				export_dir.mkdir(parents=True, exist_ok=True)
				
				export_path = export_dir / f'deployed_model_epoch_{latest_epoch}'
				success = self.model_merger.export_for_deployment(merged_model, str(export_path))
				
				if success:
					logger.info(f"模型成功导出到: {export_path}")
					
					# 记录最终统计信息
					final_stats = {
						'total_epochs': latest_epoch,
						'total_training_time': self.training_state.training_time,
						'best_dice_score': self.training_state.best_dice,
						'best_loss': self.training_state.best_loss,
						'export_path': str(export_path)
					}
					
					logger.info(f"训练完成统计: {final_stats}")
					
					# 保存训练摘要
					summary_path = Path(self.args.output_dir) / 'training_summary.yaml'
					with open(summary_path, 'w') as f:
						yaml.dump(final_stats, f, default_flow_style=False)
			
			except Exception as e:
				logger.error(f"模型合并和导出失败: {e}")
				traceback.print_exc()
		
		# 停止监控
		if self.health_monitor:
			self.health_monitor.stop_monitoring()
		
		# 全局同步，确保所有进程完成
		dist.barrier()
		
		if self.rank == 0:
			logger.info("分布式训练完全结束")
	
	def _find_latest_epoch(self, checkpoint_dir: Path) -> Optional[int]:
		"""找到最新的epoch"""
		if not checkpoint_dir.exists():
			return None
		
		latest_epoch = None
		for checkpoint_file in checkpoint_dir.glob("checkpoint_epoch_*_rank_0.pt"):
			try:
				# 从文件名提取epoch
				parts = checkpoint_file.stem.split('_')
				epoch_idx = parts.index('epoch') + 1
				epoch = int(parts[epoch_idx])
				
				if latest_epoch is None or epoch > latest_epoch:
					latest_epoch = epoch
			except (ValueError, IndexError):
				continue
		
		return latest_epoch


def setup_distributed():
	"""设置分布式环境"""
	if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
		rank = int(os.environ['RANK'])
		world_size = int(os.environ['WORLD_SIZE'])
		local_rank = int(os.environ['LOCAL_RANK'])
	else:
		raise RuntimeError("分布式环境变量未正确设置")
	
	# 初始化分布式进程组
	dist.init_process_group(
		backend='nccl',
		init_method='env://',
		world_size=world_size,
		rank=rank
	)
	
	# 设置当前设备
	torch.cuda.set_device(local_rank)
	
	return rank, world_size, local_rank


def parse_args():
	"""解析命令行参数"""
	parser = argparse.ArgumentParser(description='集成化流水线分布式训练')
	
	# 数据参数
	parser.add_argument('--image_dir', type=str, required=True, help='图像目录路径')
	parser.add_argument('--label_dir', type=str, required=True, help='标签目录路径')
	parser.add_argument('--output_dir', type=str, default='./output', help='输出目录路径')
	parser.add_argument('--config', type=str, default='configs/default.yaml', help='配置文件路径')
	
	# 训练参数
	parser.add_argument('--batch_size', type=int, default=1, help='批次大小')
	parser.add_argument('--epochs', type=int, default=100, help='训练轮数')
	parser.add_argument('--lr', type=float, default=1e-4, help='学习率')
	parser.add_argument('--num_workers', type=int, default=4, help='数据加载器工作进程数')
	parser.add_argument('--val_interval', type=int, default=5, help='验证间隔')
	parser.add_argument('--save_interval', type=int, default=10, help='保存间隔')
	parser.add_argument('--log_interval', type=int, default=10, help='日志间隔')
	
	# 其他参数
	parser.add_argument('--resume', type=str, help='恢复训练的检查点路径')
	parser.add_argument('--amp', action='store_true', help='使用自动混合精度')
	
	return parser.parse_args()


def load_config(config_path: str) -> Dict:
	"""加载配置文件"""
	config_path = Path(config_path)
	
	if not config_path.exists():
		logger.warning(f"配置文件不存在: {config_path}，使用默认配置")
		return {}
	
	try:
		with open(config_path, 'r', encoding='utf-8') as f:
			config = yaml.safe_load(f)
		
		logger.info(f"配置文件加载成功: {config_path}")
		return config
	
	except Exception as e:
		logger.error(f"配置文件加载失败: {e}")
		return {}


def cleanup_on_exit(trainer):
	"""退出时的清理函数"""
	
	def signal_handler(signum, frame):
		logger.info(f"Rank {trainer.rank}: 收到退出信号，开始清理...")
		
		try:
			# 保存紧急检查点
			trainer._save_distributed_checkpoint(
				trainer.training_state.epoch,
				{'loss': trainer.training_state.best_loss}
			)
			
			# 停止监控
			if trainer.health_monitor:
				trainer.health_monitor.stop_monitoring()
			
			# 清理分布式环境
			if dist.is_initialized():
				dist.destroy_process_group()
		
		except Exception as e:
			logger.error(f"清理过程出错: {e}")
		
		sys.exit(0)
	
	signal.signal(signal.SIGINT, signal_handler)
	signal.signal(signal.SIGTERM, signal_handler)


def main():
	"""主函数"""
	try:
		# 解析参数和设置分布式环境
		args = parse_args()
		rank, world_size, local_rank = setup_distributed()
		
		# 加载配置
		config = load_config(args.config)
		
		# 创建输出目录
		if rank == 0:
			Path(args.output_dir).mkdir(parents=True, exist_ok=True)
			logger.info(f"输出目录创建: {args.output_dir}")
		
		logger.info(f"Rank {rank}: 准备执行barrier同步...")
		
		# 让rank按顺序初始化，避免同时竞争资源
		time.sleep(rank * 2)  # 每个rank延迟不同时间
		
		
		# 全局同步
		dist.barrier()
		
		logger.info(f"Rank {rank}: 开始集成化流水线训练")
		
		# 创建集成化训练器
		trainer = IntegratedPipelineTrainer(rank, world_size, local_rank, config, args)
		
		# 设置退出清理
		cleanup_on_exit(trainer)
		
		# ✅ 添加Epoch级别进度条
		best_val_score = 0.0
		
		if rank == 0:
			# 只有主进程显示总体进度条
			epoch_pbar = tqdm(
				range(args.epochs),
				desc="🚀 Training Progress",
				ncols=100,
				colour='green'
			)
		else:
			epoch_pbar = range(args.epochs)
		
		for epoch in epoch_pbar:
			try:
				# 训练一个epoch
				epoch_start_time = time.time()
				epoch_metrics = trainer.train_epoch(epoch)
				epoch_time = time.time() - epoch_start_time
				
				# 记录训练loss（用于tensorboard）
				if rank == 6:
					trainer._last_train_loss = epoch_metrics.get('loss', 0.0)
				
				# 全局同步epoch完成
				dist.barrier()
				
				# 验证（每几个epoch一次）
				val_score = 0.0
				if epoch % args.val_interval == 0:
					val_score = trainer.validate_epoch(epoch)
					
					# 更新最佳验证分数
					if val_score > best_val_score:
						best_val_score = val_score
						
						# 保存最佳模型
						if rank == 0:
							logger.info(f"新的最佳验证分数: {best_val_score:.4f}")
				
				# ✅ 更新Epoch进度条
				if rank == 0:
					epoch_pbar.set_postfix({
						'Loss': f'{epoch_metrics.get("loss", 0.0):.4f}',
						'Val': f'{val_score:.4f}',
						'Best': f'{best_val_score:.4f}',
						'Time': f'{epoch_time:.1f}s',
						'GPU': f'{torch.cuda.memory_allocated() / 1e9:.1f}GB'
					})
					
					# 同时保持原有的详细日志
					if epoch % 5 == 0:  # 每5个epoch详细日志一次
						logger.info(
							f'Epoch {epoch}: '
							f'Train Loss: {epoch_metrics.get("loss", 0.0):.6f}, '
							f'Val Score: {val_score:.4f}, '
							f'Time: {epoch_time:.2f}s'
						)
				
				# 内存清理
				if epoch % 10 == 0:
					torch.cuda.empty_cache()
			
			except Exception as e:
				if rank == 0:
					epoch_pbar.write(f"❌ Epoch {epoch} 训练失败: {e}")
				logger.error(f"Epoch {epoch} 训练失败: {e}")
				traceback.print_exc()
				
				# 尝试恢复训练
				if "CUDA out of memory" in str(e):
					torch.cuda.empty_cache()
					if rank == 0:
						epoch_pbar.write("🧹 GPU内存清理完成，尝试继续训练")
					continue
				else:
					raise e
		
		# ✅ 关闭进度条
		if rank == 0:
			epoch_pbar.close()
			print("🎉 训练完成!")
		
		# 训练结束后的模型合并
		trainer.finalize_training()
		
		# 最终同步
		dist.barrier()
		
		if rank == 0:
			logger.info("🎉 集成化分布式训练成功完成!")
			logger.info(f"最佳验证分数: {best_val_score:.4f}")
			logger.info(f"总训练时间: {trainer.training_state.training_time:.2f}秒")
	
	except Exception as e:
		logger.error(f"Rank {rank}: 训练失败: {e}")
		traceback.print_exc()
		raise
	
	finally:
		# 清理分布式环境
		try:
			if dist.is_initialized():
				dist.destroy_process_group()
		except Exception as e:
			logger.debug(f"清理分布式环境失败: {e}")


if __name__ == '__main__':
	main()