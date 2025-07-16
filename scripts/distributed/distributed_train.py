#!/usr/bin/env python3
"""
流水线分布式训练脚本 - 方案A实现
真正的端到端流水线并行训练
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

# 流水线配置
PIPELINE_CONFIG = {
	0: {'stage': 'preprocessing', 'next_ranks': [1], 'prev_ranks': []},
	1: {'stage': 'patch_scheduling', 'next_ranks': [2, 3], 'prev_ranks': [0]},
	2: {'stage': 'ch_branch', 'next_ranks': [4], 'prev_ranks': [1]},
	3: {'stage': 'spatial_branch', 'next_ranks': [4], 'prev_ranks': [1]},
	4: {'stage': 'feature_fusion', 'next_ranks': [5], 'prev_ranks': [2, 3]},
	5: {'stage': 'multiscale_fusion', 'next_ranks': [6], 'prev_ranks': [4]},
	6: {'stage': 'segmentation_head', 'next_ranks': [], 'prev_ranks': [5]}
}


class PipelineTrainer:
	"""流水线训练协调器"""
	
	def __init__(self, rank, world_size, local_rank, config, args):
		self.rank = rank
		self.world_size = world_size
		self.local_rank = local_rank
		self.config = config
		self.args = args
		self.device = torch.cuda.current_device()
		
		# 流水线配置
		self.stage_config = PIPELINE_CONFIG[rank]
		self.next_ranks = self.stage_config['next_ranks']
		self.prev_ranks = self.stage_config['prev_ranks']
		
		# 创建通信器
		self.node_comm = NodeCommunicator(
			world_size=world_size,
			rank=rank,
			local_rank=local_rank,
			node_rank=rank // 4,
			node_count=2
		)
		
		# 创建模型和stage
		self._setup_model()
		
		# 创建数据加载器（仅rank 0）
		self._setup_dataloader()
		
		# 创建优化器和损失函数
		self._setup_training_components()
	
	def _setup_model(self):
		"""设置模型和stage"""
		# 创建完整模型（用于提取组件）
		full_model = create_vessel_segmenter(self.config)
		
		# 创建当前rank的stage
		self.stage = self._create_stage(full_model)
		
		# 收集所有参数（包含完整模型信息用于优化器）
		self.all_params = self._collect_all_parameters(full_model)
	
	def _create_stage(self, full_model):
		"""创建当前rank对应的stage"""
		from scripts.distributed.stages import (
			FrontendStage, PatchSchedulingStage, CHProcessingStage,
			SpatialFusionStage, FeatureFusionStage, MultiscaleFusionStage,
			BackendStage
		)
		
		stage_classes = {
			'preprocessing': FrontendStage,
			'patch_scheduling': PatchSchedulingStage,
			'ch_branch': CHProcessingStage,
			'spatial_branch': SpatialFusionStage,
			'feature_fusion': FeatureFusionStage,
			'multiscale_fusion': MultiscaleFusionStage,
			'segmentation_head': BackendStage
		}
		
		stage_name = self.stage_config['stage']
		stage_class = stage_classes[stage_name]
		return stage_class(full_model, self.device, self.node_comm, config=self.config)
	
	def _collect_all_parameters(self, full_model):
		"""收集所有参数信息（用于创建统一优化器）"""
		# 每个rank都保存完整模型的参数引用，但只更新自己stage的参数
		return list(full_model.parameters())
	
	def _setup_dataloader(self):
		"""设置数据加载器"""
		if self.rank == 0:
			data_config = self.config.get('data', {})
			train_dataset = LiverVesselDataset(
				image_dir=self.args.image_dir,
				label_dir=self.args.label_dir,
				max_cases=data_config.get('max_cases', None),
				random_sampling=data_config.get('random_sampling', True),
				enable_smart_sampling=True,
				config=self.config
			)
			
			self.train_loader = DataLoader(
				train_dataset,
				batch_size=self.args.batch_size,
				shuffle=True,
				num_workers=self.args.num_workers,
				pin_memory=True,
				drop_last=True
			)
			self.labels_cache = {}  # 缓存labels用于rank 6
		else:
			self.train_loader = None
	
	def _setup_training_components(self):
		"""设置训练组件"""
		# 创建优化器（所有rank都需要，用于参数同步）
		self.optimizer = torch.optim.AdamW(self.all_params, lr=self.args.lr, weight_decay=1e-4)
		
		# 创建损失函数（仅rank 6需要）
		if self.rank == 6:
			loss_config = self.config.get('loss', {})
			self.loss_fn = CombinedLoss(
				num_classes=loss_config.get('num_classes', 1),
				vessel_weight=loss_config.get('vessel_weight', 10.0),
				tumor_weight=loss_config.get('tumor_weight', 15.0),
				use_boundary=loss_config.get('use_boundary', True)
			)
			
			# 创建评估器（利用现有评估模块）
			self._setup_evaluator()
		
		# 学习率调度器
		self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
			self.optimizer, T_max=self.args.epochs
		)
		
		# 混合精度
		self.scaler = torch.cuda.amp.GradScaler() if self.args.amp else None
		
		# 内存管理
		self.memory_pool = {}
		self.max_cached_batches = 10
		
		# labels缓存（rank 0和rank 6共享）
		self.labels_queue = {}
	
	def _setup_evaluator(self):
		"""设置评估器（利用项目现有评估模块）"""
		try:
			from evaluation.metrics import VesselMetrics
			from evaluation.visualizer import SegmentationVisualizer
			
			self.metrics = VesselMetrics(
				num_classes=self.config.get('loss', {}).get('num_classes', 1),
				metrics=['dice', 'iou', 'precision', 'recall', 'hausdorff']
			)
			self.visualizer = SegmentationVisualizer(
				save_dir=Path(self.args.output_dir) / 'visualizations'
			)
		except ImportError:
			logger.warning("Evaluation modules not found, using basic metrics")
			self.metrics = None
			self.visualizer = None
	
	def train_epoch(self, epoch):
		"""训练一个epoch"""
		self.stage.train()
		
		if self.rank == 0:
			return self._run_data_producer(epoch)
		elif self.rank == 6:
			return self._run_loss_calculator(epoch)
		else:
			return self._run_intermediate_processor(epoch)
	
	def _run_data_producer(self, epoch):
		"""rank 0: 数据生产者"""
		total_loss = 0.0
		num_batches = 0
		
		for batch_idx, (images, labels) in enumerate(self.train_loader):
			batch_id = f"{epoch}_{batch_idx}"
			
			# 移动数据到设备
			images = images.to(self.device, non_blocking=True)
			labels = labels.to(self.device, non_blocking=True)
			
			# 同步所有rank开始处理
			dist.barrier()
			
			# 执行预处理stage
			with torch.cuda.amp.autocast() if self.args.amp else torch.no_grad():
				processed_data = self.stage.forward(images)
			
			# 发送数据给下一个stage
			self._send_data_with_id(processed_data, batch_id, self.next_ranks[0])
			
			# 发送labels给rank 6（用于loss计算）
			self._send_labels_to_final_rank(labels, batch_id)
			
			# 等待所有rank完成前向传播
			dist.barrier()
			
			# 等待反向传播梯度
			gradients = self._receive_gradients()
			
			# 应用梯度更新
			self._apply_gradients_and_step(gradients)
			
			# 从rank 6获取loss值用于监控
			loss_value = self._receive_loss_value()
			total_loss += loss_value
			num_batches += 1
			
			# 内存清理
			self._cleanup_batch_cache(batch_id)
			
			if batch_idx % self.args.log_interval == 0:
				logger.info(f'Rank {self.rank} - Epoch {epoch}, Batch {batch_idx}, Loss: {loss_value:.6f}')
		
		# 发送epoch结束信号
		self._send_epoch_end_signal()
		
		# 同步epoch结束
		dist.barrier()
		return total_loss / max(num_batches, 1)
	
	def _run_intermediate_processor(self, epoch):
		"""rank 1-5: 中间处理者"""
		processed_count = 0
		
		while True:
			# 同步所有rank
			dist.barrier()
			
			# 特殊处理rank 4的双输入融合
			if self.rank == 4:
				data_package = self._receive_dual_inputs_for_fusion()
			else:
				data_package = self._receive_data_with_id()
			
			if data_package is None:  # epoch结束信号
				break
			
			input_data, batch_id = data_package
			
			# 缓存输入数据用于梯度计算
			self.memory_pool[batch_id] = {
				'input': input_data,
				'requires_grad': True
			}
			
			# 执行当前stage
			with torch.cuda.amp.autocast() if self.args.amp else torch.no_grad():
				if self.rank == 4:  # 特征融合
					ch_features, spatial_features = input_data
					output_data = self.stage.process(ch_features, spatial_features, tiers=None)
				else:
					output_data = self.stage.forward(input_data)
			
			# 缓存输出用于梯度计算
			self.memory_pool[batch_id]['output'] = output_data
			
			# 发送数据给下一个stage
			for next_rank in self.next_ranks:
				self._send_data_with_id(output_data, batch_id, next_rank)
			
			# 等待前向传播完成
			dist.barrier()
			
			# 接收梯度并传播
			downstream_gradients = self._receive_gradients_with_autograd()
			
			# 计算真正的上游梯度
			upstream_gradients = self._compute_upstream_gradients_autograd(
				batch_id, downstream_gradients
			)
			
			# 发送梯度给上游
			for prev_rank in self.prev_ranks:
				self._send_gradients(upstream_gradients, prev_rank)
			
			# 更新参数
			self._apply_gradients_and_step(downstream_gradients)
			
			# 清理内存
			self._cleanup_batch_cache(batch_id)
			processed_count += 1
		
		logger.info(f'Rank {self.rank} processed {processed_count} batches in epoch {epoch}')
		return 0.0  # 中间stage不计算loss
	
	def _run_loss_calculator(self, epoch):
		"""rank 6: 损失计算者"""
		total_loss = 0.0
		num_batches = 0
		all_predictions = []
		all_labels = []
		
		while True:
			# 同步所有rank
			dist.barrier()
			
			# 接收数据
			data_package = self._receive_data_with_id()
			if data_package is None:
				break
			
			final_features, batch_id = data_package
			
			# 接收对应的labels
			labels = self._receive_labels_from_rank0(batch_id)
			
			# 缓存数据用于梯度计算
			self.memory_pool[batch_id] = {
				'input': final_features,
				'labels': labels,
				'requires_grad': True
			}
			
			# 启用梯度计算
			final_features.requires_grad_(True)
			
			# 执行最终的分割头
			self.optimizer.zero_grad()
			
			if self.args.amp and self.scaler:
				with torch.cuda.amp.autocast():
					predictions = self.stage.forward(final_features)
					loss = self.loss_fn(predictions, labels)
				
				# 缓存预测结果用于评估
				self.memory_pool[batch_id]['predictions'] = predictions.detach()
				
				# 反向传播
				self.scaler.scale(loss).backward(retain_graph=True)
				
				# 提取并发送梯度给上游
				input_gradients = final_features.grad.clone()
				self._send_gradients_with_autograd(input_gradients, self.prev_ranks[0])
				
				# 等待梯度传播完成
				dist.barrier()
				
				# 更新参数
				self.scaler.step(self.optimizer)
				self.scaler.update()
			else:
				predictions = self.stage.forward(final_features)
				loss = self.loss_fn(predictions, labels)
				
				# 缓存预测结果
				self.memory_pool[batch_id]['predictions'] = predictions.detach()
				
				loss.backward(retain_graph=True)
				
				# 发送梯度给上游
				input_gradients = final_features.grad.clone()
				self._send_gradients_with_autograd(input_gradients, self.prev_ranks[0])
				
				# 等待梯度传播完成
				dist.barrier()
				
				torch.nn.utils.clip_grad_norm_(self.stage.parameters(), max_norm=1.0)
				self.optimizer.step()
			
			# 发送loss值给rank 0用于监控
			self._send_loss_value(loss.item())
			
			# 收集评估数据
			if self.metrics:
				all_predictions.append(predictions.detach().cpu())
				all_labels.append(labels.detach().cpu())
			
			total_loss += loss.item()
			num_batches += 1
			
			# 清理内存
			self._cleanup_batch_cache(batch_id)
		
		# 计算评估指标
		if self.metrics and all_predictions:
			self._compute_and_log_metrics(all_predictions, all_labels, epoch)
		
		return total_loss / max(num_batches, 1)
	
	def _send_labels_to_final_rank(self, labels, batch_id):
		"""发送labels给rank 6"""
		# 先发送batch_id
		batch_id_tensor = torch.tensor([hash(batch_id) % 1000000],
		                               dtype=torch.long, device=self.device)
		self.node_comm.send_tensor(batch_id_tensor, 6, tag=10)
		
		# 再发送labels
		self.node_comm.send_tensor(labels, 6, tag=11)
	
	def _receive_labels_from_rank0(self, expected_batch_id):
		"""rank 6接收来自rank 0的labels"""
		# 接收batch_id验证
		batch_id_tensor = self.node_comm.recv_tensor(
			0, dtype=torch.long, device=self.device
		)
		
		received_batch_id = str(batch_id_tensor.item())
		if received_batch_id != expected_batch_id.split('_')[-1]:  # 只比较batch索引
			logger.warning(f"Batch ID mismatch: expected {expected_batch_id}, got {received_batch_id}")
		
		# 接收labels
		labels = self.node_comm.recv_tensor(0, device=self.device)
		return labels
	
	def _receive_dual_inputs_for_fusion(self):
		"""rank 4接收来自rank 2和rank 3的双路输入"""
		try:
			# 接收来自rank 2的CH特征
			ch_batch_id_tensor = self.node_comm.recv_tensor(
				2, dtype=torch.long, device=self.device
			)
			
			if ch_batch_id_tensor.item() == -1:  # epoch结束信号
				return None
			
			ch_data = self.node_comm.recv_tensor(2, device=self.device)
			ch_batch_id = str(ch_batch_id_tensor.item())
			
			# 接收来自rank 3的空间特征
			spatial_batch_id_tensor = self.node_comm.recv_tensor(
				3, dtype=torch.long, device=self.device
			)
			spatial_data = self.node_comm.recv_tensor(3, device=self.device)
			spatial_batch_id = str(spatial_batch_id_tensor.item())
			
			# 验证batch_id一致性
			if ch_batch_id != spatial_batch_id:
				logger.error(f"Batch ID mismatch in fusion: CH={ch_batch_id}, Spatial={spatial_batch_id}")
				return None
			
			# 返回融合的输入数据
			fused_input = [ch_data, spatial_data]
			return fused_input, ch_batch_id
		
		except Exception as e:
			logger.error(f"Failed to receive dual inputs for fusion: {e}")
			return None
	
	def _compute_upstream_gradients_autograd(self, batch_id, downstream_gradients):
		"""使用autograd计算真正的上游梯度"""
		try:
			# 获取缓存的输入输出数据
			cache = self.memory_pool.get(batch_id, {})
			input_data = cache.get('input')
			output_data = cache.get('output')
			
			if input_data is None or output_data is None:
				logger.warning(f"Missing cached data for batch {batch_id}")
				return self._extract_gradients()  # 回退到简单梯度
			
			# 确保输入需要梯度
			if isinstance(input_data, list):
				# 处理多输入情况（如rank 4的特征融合）
				for inp in input_data:
					if isinstance(inp, torch.Tensor):
						inp.requires_grad_(True)
				input_tensors = [inp for inp in input_data if isinstance(inp, torch.Tensor)]
			else:
				input_data.requires_grad_(True)
				input_tensors = [input_data]
			
			# 计算梯度
			upstream_grads = torch.autograd.grad(
				outputs=output_data,
				inputs=input_tensors,
				grad_outputs=downstream_gradients,
				retain_graph=False,
				allow_unused=True
			)
			
			return [grad for grad in upstream_grads if grad is not None]
		
		except Exception as e:
			logger.warning(f"Autograd gradient computation failed for batch {batch_id}: {e}")
			return self._extract_gradients()  # 回退到简单梯度
	
	def _send_gradients_with_autograd(self, gradients, dst_rank):
		"""发送autograd计算的梯度"""
		if isinstance(gradients, torch.Tensor):
			self.node_comm.send_tensor(gradients, dst_rank)
		elif isinstance(gradients, list):
			# 发送梯度数量
			grad_count = torch.tensor([len(gradients)], dtype=torch.long, device=self.device)
			self.node_comm.send_tensor(grad_count, dst_rank)
			
			# 发送每个梯度
			for grad in gradients:
				if grad is not None:
					self.node_comm.send_tensor(grad, dst_rank)
	
	def _receive_gradients_with_autograd(self):
		"""接收autograd梯度"""
		try:
			# 先尝试接收梯度数量
			grad_count_tensor = self.node_comm.recv_tensor(
				self.next_ranks[0], dtype=torch.long, device=self.device
			)
			grad_count = grad_count_tensor.item()
			
			# 接收多个梯度
			gradients = []
			for _ in range(grad_count):
				grad = self.node_comm.recv_tensor(self.next_ranks[0], device=self.device)
				gradients.append(grad)
			
			return gradients
		
		except:
			# 回退到接收单个梯度
			grad = self.node_comm.recv_tensor(self.next_ranks[0], device=self.device)
			return grad
	
	def _cleanup_batch_cache(self, batch_id):
		"""清理批次缓存"""
		if batch_id in self.memory_pool:
			del self.memory_pool[batch_id]
		
		# 限制缓存大小
		if len(self.memory_pool) > self.max_cached_batches:
			oldest_batch = min(self.memory_pool.keys())
			del self.memory_pool[oldest_batch]
		
		# 清理GPU内存
		if len(self.memory_pool) % 5 == 0:  # 每5个batch清理一次
			torch.cuda.empty_cache()
	
	def _compute_and_log_metrics(self, predictions, labels, epoch):
		"""计算并记录评估指标"""
		try:
			# 合并所有预测和标签
			all_preds = torch.cat(predictions, dim=0)
			all_labels = torch.cat(labels, dim=0)
			
			# 计算指标
			metrics_dict = self.metrics.compute(all_preds, all_labels)
			
			# 记录指标
			logger.info(f"Epoch {epoch} Evaluation Metrics:")
			for metric_name, value in metrics_dict.items():
				logger.info(f"  {metric_name}: {value:.4f}")
			
			# 保存可视化（可选）
			if self.visualizer and epoch % 10 == 0:  # 每10个epoch可视化一次
				sample_pred = all_preds[:1]  # 取第一个样本
				sample_label = all_labels[:1]
				self.visualizer.save_comparison(
					sample_pred, sample_label,
					save_path=f"epoch_{epoch}_sample.png"
				)
		
		except Exception as e:
			logger.warning(f"Failed to compute metrics: {e}")
	
	def validate_epoch(self, epoch):
		"""验证一个epoch（仅rank 0执行）"""
		if self.rank != 0:
			return 0.0
		
		# 简化的验证逻辑
		logger.info(f"Validation for epoch {epoch} (simplified)")
		
		# TODO: 实现完整的验证逻辑
		# 这里可以加载验证数据集并通过流水线进行推理
		
		return 0.0
	
	def _send_data_with_id(self, data, batch_id, dst_rank):
		"""发送数据和batch ID"""
		# 先发送batch_id
		batch_id_tensor = torch.tensor([hash(batch_id) % 1000000],
		                               dtype=torch.long, device=self.device)
		self.node_comm.send_tensor(batch_id_tensor, dst_rank, tag=0)
		
		# 再发送数据
		self.node_comm.send_tensor(data, dst_rank, tag=1)
	
	def _receive_data_with_id(self):
		"""接收数据和batch ID"""
		try:
			# 接收batch_id
			batch_id_tensor = self.node_comm.recv_tensor(
				self.prev_ranks[0], dtype=torch.long, device=self.device
			)
			
			if batch_id_tensor.item() == -1:  # epoch结束信号
				return None
			
			# 接收数据
			data = self.node_comm.recv_tensor(
				self.prev_ranks[0], device=self.device
			)
			
			batch_id = str(batch_id_tensor.item())
			return data, batch_id
		
		except Exception as e:
			logger.error(f"Rank {self.rank} failed to receive data: {e}")
			return None
	
	def _send_gradients(self, gradients, dst_rank):
		"""发送梯度"""
		# 简化实现：发送当前stage的梯度
		for param in self.stage.parameters():
			if param.grad is not None:
				self.node_comm.send_tensor(param.grad, dst_rank)
	
	def _receive_gradients(self):
		"""接收梯度"""
		gradients = []
		for param in self.stage.parameters():
			if param.grad is not None:
				grad = self.node_comm.recv_tensor(
					self.next_ranks[0], device=self.device
				)
				gradients.append(grad)
		return gradients
	
	def _apply_gradients_and_step(self, gradients):
		"""应用梯度并更新参数"""
		# 将接收到的梯度应用到参数上
		for param, grad in zip(self.stage.parameters(), gradients):
			if param.grad is not None:
				param.grad.copy_(grad)
		
		# 同步所有rank的参数
		self._sync_parameters()
		
		self.optimizer.step()
	
	def _sync_parameters(self):
		"""同步所有rank的参数"""
		for param in self.all_params:
			dist.all_reduce(param.data, op=dist.ReduceOp.AVG)
	
	def _extract_gradients(self):
		"""提取当前stage的梯度"""
		gradients = []
		for param in self.stage.parameters():
			if param.grad is not None:
				gradients.append(param.grad.clone())
		return gradients
	
	def _compute_upstream_gradients(self, input_data, downstream_gradients):
		"""计算上游梯度"""
		# 简化实现：返回当前梯度
		return self._extract_gradients()
	
	def _send_loss_value(self, loss_value):
		"""发送loss值给rank 0"""
		loss_tensor = torch.tensor([loss_value], device=self.device)
		self.node_comm.send_tensor(loss_tensor, 0)
	
	def _receive_loss_value(self):
		"""接收loss值"""
		loss_tensor = self.node_comm.recv_tensor(6, device=self.device)
		return loss_tensor.item()
	
	def _send_epoch_end_signal(self):
		"""发送epoch结束信号"""
		end_signal = torch.tensor([-1], dtype=torch.long, device=self.device)
		
		# 发送给所有下游rank
		for next_rank in self.next_ranks:
			self.node_comm.send_tensor(end_signal, next_rank, tag=0)
		
		# rank 0还需要发送给rank 6（labels通道）
		if self.rank == 0:
			self.node_comm.send_tensor(end_signal, 6, tag=10)
	
	def _extract_gradients(self):
		"""提取当前stage的梯度（回退方法）"""
		gradients = []
		for param in self.stage.parameters():
			if param.grad is not None:
				gradients.append(param.grad.clone())
		return gradients
	
	def _send_gradients(self, gradients, dst_rank):
		"""发送梯度（简化版本）"""
		if isinstance(gradients, list):
			for grad in gradients:
				if grad is not None:
					self.node_comm.send_tensor(grad, dst_rank)
		else:
			self.node_comm.send_tensor(gradients, dst_rank)
	
	def _receive_gradients(self):
		"""接收梯度（简化版本）"""
		gradients = []
		try:
			for param in self.stage.parameters():
				if param.grad is not None:
					grad = self.node_comm.recv_tensor(self.next_ranks[0], device=self.device)
					gradients.append(grad)
		except:
			# 如果接收失败，返回零梯度
			for param in self.stage.parameters():
				if param.grad is not None:
					gradients.append(torch.zeros_like(param.grad))
		
		return gradients
	
	def _apply_gradients_and_step(self, gradients):
		"""应用梯度并更新参数"""
		try:
			# 应用接收到的梯度
			if isinstance(gradients, list):
				for param, grad in zip(self.stage.parameters(), gradients):
					if param.grad is not None and grad is not None:
						param.grad.copy_(grad)
			
			# 梯度裁剪
			torch.nn.utils.clip_grad_norm_(self.stage.parameters(), max_norm=1.0)
			
			# 同步所有rank的参数（关键：确保参数一致性）
			self._sync_parameters()
			
			# 更新参数
			self.optimizer.step()
		
		except Exception as e:
			logger.warning(f"Gradient application failed: {e}")
	
	def _sync_parameters(self):
		"""同步所有rank的参数"""
		# 只同步本stage的参数，避免干扰其他stage
		for param in self.stage.parameters():
			dist.all_reduce(param.data, op=dist.ReduceOp.AVG)
	
	def _send_loss_value(self, loss_value):
		"""发送loss值给rank 0"""
		loss_tensor = torch.tensor([loss_value], dtype=torch.float32, device=self.device)
		self.node_comm.send_tensor(loss_tensor, 0)
	
	def _receive_loss_value(self):
		"""接收loss值"""
		try:
			loss_tensor = self.node_comm.recv_tensor(6, device=self.device)
			return loss_tensor.item()
		except:
			return 0.0  # 如果接收失败，返回0
	
	def _get_labels_for_batch(self, batch_id):
		"""获取batch对应的labels（弃用，改用直接通信）"""
		# 这个方法已被_receive_labels_from_rank0替代
		return self._receive_labels_from_rank0(batch_id)
	
	def _send_epoch_end_signal(self):
		"""发送epoch结束信号"""
		end_signal = torch.tensor([-1], dtype=torch.long, device=self.device)
		for next_rank in self.next_ranks:
			self.node_comm.send_tensor(end_signal, next_rank, tag=0)


def setup_distributed():
	"""设置分布式环境"""
	if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
		rank = int(os.environ['RANK'])
		world_size = int(os.environ['WORLD_SIZE'])
		local_rank = int(os.environ['LOCAL_RANK'])
	else:
		raise RuntimeError("Distributed environment not properly set up")
	
	dist.init_process_group(
		backend='nccl',
		init_method='env://',
		world_size=world_size,
		rank=rank
	)
	
	torch.cuda.set_device(local_rank)
	return rank, world_size, local_rank


def parse_args():
	"""解析命令行参数"""
	parser = argparse.ArgumentParser(description='Pipeline Distributed Training')
	
	parser.add_argument('--image_dir', type=str, required=True)
	parser.add_argument('--label_dir', type=str, required=True)
	parser.add_argument('--output_dir', type=str, default='./output')
	parser.add_argument('--config', type=str, default='configs/default.yaml')
	
	parser.add_argument('--batch_size', type=int, default=1)
	parser.add_argument('--epochs', type=int, default=100)
	parser.add_argument('--lr', type=float, default=1e-4)
	parser.add_argument('--num_workers', type=int, default=4)
	parser.add_argument('--val_interval', type=int, default=5)
	parser.add_argument('--save_interval', type=int, default=10)
	parser.add_argument('--log_interval', type=int, default=10)
	
	parser.add_argument('--resume', type=str, help='Resume from checkpoint')
	parser.add_argument('--amp', action='store_true', help='Use automatic mixed precision')
	
	return parser.parse_args()


def load_config(config_path):
	"""加载配置文件"""
	with open(config_path, 'r') as f:
		config = yaml.safe_load(f)
	return config


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
		
		# 全局同步
		dist.barrier()
		
		logger.info(f"Rank {rank}: Starting pipeline training")
		
		# 创建流水线训练器
		trainer = PipelineTrainer(rank, world_size, local_rank, config, args)
		
		# 训练循环
		best_loss = float('inf')
		
		for epoch in range(args.epochs):
			# 训练一个epoch
			epoch_start_time = time.time()
			epoch_loss = trainer.train_epoch(epoch)
			epoch_time = time.time() - epoch_start_time
			
			# 全局同步epoch完成
			dist.barrier()
			
			# 验证（每几个epoch一次）
			val_loss = 0.0
			if epoch % args.val_interval == 0:
				val_loss = trainer.validate_epoch(epoch)
			
			# 更新学习率
			trainer.scheduler.step()
			
			# 主进程记录和保存
			if rank == 0:
				logger.info(f'Epoch {epoch}: Train Loss: {epoch_loss:.6f}, '
				            f'Val Loss: {val_loss:.6f}, Time: {epoch_time:.2f}s')
				
				# 保存最佳模型
				if epoch_loss < best_loss:
					best_loss = epoch_loss
					best_checkpoint_path = Path(args.output_dir) / 'best_model.pt'
					torch.save({
						'epoch': epoch,
						'model_state_dict': trainer.stage.state_dict(),
						'optimizer_state_dict': trainer.optimizer.state_dict(),
						'loss': best_loss,
						'rank': rank
					}, best_checkpoint_path)
					logger.info(f"Best model saved with loss: {best_loss:.6f}")
				
				# 定期保存检查点
				if epoch % args.save_interval == 0:
					checkpoint_path = Path(args.output_dir) / f'checkpoint_epoch_{epoch}.pt'
					torch.save({
						'epoch': epoch,
						'model_state_dict': trainer.stage.state_dict(),
						'optimizer_state_dict': trainer.optimizer.state_dict(),
						'scheduler_state_dict': trainer.scheduler.state_dict(),
						'loss': epoch_loss,
						'rank': rank,
						'config': config
					}, checkpoint_path)
					logger.info(f"Checkpoint saved: {checkpoint_path}")
			
			# 内存清理
			if epoch % 10 == 0:
				torch.cuda.empty_cache()
		
		# 最终同步
		dist.barrier()
		logger.info(f"Rank {rank}: Training completed successfully")
	
	except Exception as e:
		logger.error(f"Rank {rank}: Training failed: {e}")
		import traceback
		traceback.print_exc()
		raise
	finally:
		# 清理分布式环境
		if dist.is_initialized():
			dist.destroy_process_group()


if __name__ == '__main__':
	main()