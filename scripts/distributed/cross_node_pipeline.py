# cross_node_pipeline.py

import time
import threading
import queue
import torch
import torch.nn.functional as F
import numpy as np
import logging
from typing import Dict, List, Optional, Any

# 配置日志
logger = logging.getLogger("cross_node_pipeline")


class CrossNodePipeline:
	"""管理跨节点的数据流和处理流水线 - 优化版"""
	
	def __init__(self, stages, node_comm):
		"""
		初始化跨节点流水线

		参数:
			stages: 处理阶段字典
			node_comm: 节点通信管理器
		"""
		self.stages = stages
		self.node_comm = node_comm
		
		# 获取节点和GPU信息
		self.node_rank = node_comm.node_rank
		self.local_rank = node_comm.local_rank
		self.world_rank = node_comm.rank
		
		# 创建工作队列和线程
		self.input_queue = queue.Queue(maxsize=3)
		self.output_queue = queue.Queue(maxsize=3)
		self.worker_thread = None
		self.running = False
		self.exception = None
		
		# 性能统计（简化版）
		self.batch_times = []
		self.processed_batches = 0
		
		# 流水线状态
		self.pipeline_lock = threading.Lock()
		self.is_training = True
		
		logger.info(
			f"CrossNodePipeline initialized on node {self.node_rank}, GPU {self.local_rank} (world rank {self.world_rank})")
	
	def start_worker(self):
		"""启动工作线程"""
		if self.worker_thread is not None and self.worker_thread.is_alive():
			return
		
		self.running = True
		self.exception = None
		
		self.worker_thread = threading.Thread(
			target=self._worker_loop,
			daemon=True,
			name=f"node{self.node_rank}_gpu{self.local_rank}_worker"
		)
		self.worker_thread.start()
		logger.info(f"Started worker thread on node {self.node_rank}, GPU {self.local_rank}")
	
	def stop_worker(self):
		"""停止工作线程"""
		self.running = False
		
		if self.worker_thread and self.worker_thread.is_alive():
			self.worker_thread.join(timeout=3.0)
			logger.info(f"Stopped worker thread on node {self.node_rank}, GPU {self.local_rank}")
	
	def _worker_loop(self):
		"""工作循环 - 简化版，专注核心功能"""
		while self.running:
			try:
				# 从输入队列获取批次
				batch = self.input_queue.get(timeout=1.0)
				if batch is None:
					continue
				
				batch_start = time.time()
				
				# 执行当前节点的处理阶段
				result = self._execute_stages(batch)
				
				# 记录批次处理时间
				batch_time = time.time() - batch_start
				self.batch_times.append(batch_time)
				
				# 输出结果
				self.output_queue.put(result)
				self.processed_batches += 1
			
			except queue.Empty:
				continue
			except Exception as e:
				self.exception = e
				logger.error(f"Worker loop error: {e}")
				break
	
	def _execute_stages(self, batch):
		"""执行当前节点的所有阶段"""
		current_data = batch
		
		# 按顺序执行当前节点的所有阶段
		for stage_name, stage in self.stages.items():
			try:
				if hasattr(stage, 'forward'):
					current_data = stage.forward(current_data)
				else:
					current_data = stage.process(current_data)
			except Exception as e:
				logger.error(f"Stage {stage_name} failed: {e}")
				return None
		
		return current_data
	
	def forward(self, batch):
		"""
		前向传播

		参数:
			batch: 输入批次

		返回:
			处理结果
		"""
		try:
			# 将批次放入输入队列
			self.input_queue.put(batch, timeout=5.0)
			
			# 获取处理结果
			result = self.output_queue.get(timeout=30.0)
			
			return result
		
		except queue.Full:
			logger.error("Input queue is full")
			return None
		except queue.Empty:
			logger.error("No result received within timeout")
			return None
		except Exception as e:
			logger.error(f"Forward pass failed: {e}")
			return None
	
	def backward(self, loss, batch):
		"""执行反向传播 - 优化版"""
		if not self.is_training:
			return
		
		# 简化的反向传播：直接执行，不需要复杂的缓存机制
		try:
			loss.backward()
			
			# 如果是分布式训练，确保梯度同步
			if hasattr(self, 'stages'):
				for stage in self.stages.values():
					if hasattr(stage, 'module') and hasattr(stage.module, 'parameters'):
						# 简单的梯度裁剪
						torch.nn.utils.clip_grad_norm_(stage.module.parameters(), max_norm=1.0)
		
		except Exception as e:
			logger.error(f"Backward pass failed: {e}")
	
	def get_all_parameters(self):
		"""获取所有阶段的参数"""
		all_params = []
		for stage in self.stages.values():
			if hasattr(stage, 'parameters'):
				all_params.extend(stage.parameters())
			elif hasattr(stage, 'module') and hasattr(stage.module, 'parameters'):
				all_params.extend(stage.module.parameters())
		return all_params
	
	def train(self, mode=True):
		"""设置训练模式"""
		self.is_training = mode
		for stage in self.stages.values():
			if hasattr(stage, 'train'):
				stage.train(mode)
		return self
	
	def eval(self):
		"""设置评估模式"""
		return self.train(False)
	
	def get_performance_stats(self):
		"""
		获取性能统计（简化版）

		返回:
			性能统计字典
		"""
		# 计算吞吐量 (批次/秒)
		if self.batch_times:
			avg_batch_time = sum(self.batch_times) / len(self.batch_times)
			throughput = 1.0 / avg_batch_time if avg_batch_time > 0 else 0
		else:
			throughput = 0
		
		# 计算平均延迟
		latency = sum(self.batch_times) / max(1, len(self.batch_times)) * 1000  # 毫秒
		
		stats = {
			'throughput': throughput,
			'latency_ms': latency,
			'processed_batches': self.processed_batches,
		}
		
		return stats
	
	def reset_stats(self):
		"""重置性能统计"""
		self.batch_times = []
		self.processed_batches = 0
		
		for stage in self.stages.values():
			if hasattr(stage, 'reset_stats'):
				stage.reset_stats()
	
	def __del__(self):
		"""析构函数，确保停止工作线程"""
		self.stop_worker()
	
	# 为了兼容原始接口，添加一些方法
	def __call__(self, batch):
		"""使流水线可调用"""
		return self.forward(batch)
	
	def parameters(self):
		"""获取所有参数"""
		for stage in self.stages.values():
			if hasattr(stage, 'parameters'):
				yield from stage.parameters()
			elif hasattr(stage, 'module') and hasattr(stage.module, 'parameters'):
				yield from stage.module.parameters()
	
	def state_dict(self):
		"""获取状态字典"""
		state_dict = {}
		for stage_name, stage in self.stages.items():
			if hasattr(stage, 'get_state_dict_prefix'):
				stage_state = stage.get_state_dict_prefix()
				for key, value in stage_state.items():
					state_dict[f"{stage_name}.{key}"] = value
		return state_dict
	
	def load_state_dict(self, state_dict):
		"""加载状态字典"""
		for stage_name, stage in self.stages.items():
			stage_prefix = f"{stage_name}."
			stage_state = {}
			
			for key, value in state_dict.items():
				if key.startswith(stage_prefix):
					new_key = key[len(stage_prefix):]
					stage_state[new_key] = value
			
			if stage_state and hasattr(stage, 'load_state_dict'):
				try:
					stage.load_state_dict(stage_state, strict=False)
				except Exception as e:
					logger.warning(f"Failed to load state for stage {stage_name}: {e}")


# 工厂函数，用于创建流水线
def create_pipeline(config, node_comm):
	"""
	创建跨节点流水线的工厂函数

	参数:
		config: 配置字典
		node_comm: 节点通信管理器

	返回:
		CrossNodePipeline实例
	"""
	from .stages import create_pipeline_stages
	
	# 创建阶段
	stages = create_pipeline_stages(config, node_comm)
	
	# 创建流水线
	pipeline = CrossNodePipeline(stages, node_comm)
	
	return pipeline