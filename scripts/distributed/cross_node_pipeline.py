#cross_node_pipeline.py

import time
import threading
import queue
import torch
import numpy as np
import logging
from typing import Dict, List, Optional, Any

# 配置日志
logger = logging.getLogger("cross_node_pipeline")


class CrossNodePipeline:
	"""管理跨节点的数据流和处理流水线"""
	
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
		
		# 性能统计
		self.batch_times = []
		self.stage_times = {}
		self.last_batch_start = 0
		self.processed_batches = 0
		
		# 流水线状态
		self.pipeline_lock = threading.Lock()
		self.is_training = True
		
		# 结果缓存 (用于反向传播)
		self.result_cache = {}
		self.cache_lock = threading.Lock()
		
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
		"""工作线程主循环"""
		try:
			while self.running:
				try:
					# 获取输入数据
					if self.input_queue.empty():
						time.sleep(0.001)  # 避免CPU忙等
						continue
					
					batch_data = self.input_queue.get(timeout=1.0)
					if batch_data is None:  # 终止信号
						break
					
					batch, batch_id = batch_data
					
					# 记录开始时间
					start_time = time.time()
					
					# 执行处理
					result = self._process_batch(batch)
					
					# 记录处理时间
					elapsed = time.time() - start_time
					self.batch_times.append(elapsed)
					
					# 放入输出队列
					self.output_queue.put((result, batch_id))
					
					# 完成任务
					self.input_queue.task_done()
					self.processed_batches += 1
				
				except queue.Empty:
					continue
		
		except Exception as e:
			logger.error(f"Error in worker thread: {str(e)}")
			self.exception = e
			self.running = False
	
	def _process_batch(self, batch):
		"""根据当前节点和GPU处理批次"""
		if self.node_rank == 0:  # 节点1
			if self.local_rank == 0:
				# GPU 0: 数据预处理
				start = time.time()
				result = self._execute_preprocessing_stage(batch)
				self._update_stage_time('preprocessing', time.time() - start)
				return result
			
			elif self.local_rank == 1:
				# GPU 1: Patch调度
				start = time.time()
				result = self._execute_patch_scheduling_stage(batch)
				self._update_stage_time('patch_scheduling', time.time() - start)
				return result
			
			elif self.local_rank == 2:
				# GPU 2: CH分支
				start = time.time()
				result = self._execute_ch_branch_stage(batch)
				self._update_stage_time('ch_branch', time.time() - start)
				return result
			
			elif self.local_rank == 3:
				# GPU 3: 空间分支
				start = time.time()
				result = self._execute_spatial_branch_stage(batch)
				self._update_stage_time('spatial_branch', time.time() - start)
				return result
		
		else:  # 节点2
			if self.local_rank == 0:
				# GPU 4: 特征融合
				start = time.time()
				result = self._execute_feature_fusion_stage(batch)
				self._update_stage_time('feature_fusion', time.time() - start)
				return result
			
			elif self.local_rank == 1:
				# GPU 5: 多尺度融合
				start = time.time()
				result = self._execute_multiscale_fusion_stage(batch)
				self._update_stage_time('multiscale_fusion', time.time() - start)
				return result
			
			elif self.local_rank == 2:
				# GPU 6: 分割头和损失计算
				start = time.time()
				result = self._execute_segmentation_head_stage(batch)
				self._update_stage_time('segmentation_head', time.time() - start)
				return result
		
		return None
	
	def forward(self, batch, is_training=True):
		"""
		执行前向传播

		参数:
			batch: 输入批次
			is_training: 是否处于训练模式

		返回:
			模型输出
		"""
		# 保存训练模式
		self.is_training = is_training
		
		# 设置所有阶段的训练模式
		for stage in self.stages.values():
			if hasattr(stage, 'train'):
				stage.train(is_training)
		
		# 确保工作线程正在运行
		if not self.running:
			self.start_worker()
		
		# 检查是否有异常发生
		if self.exception:
			raise RuntimeError(f"Pipeline error: {str(self.exception)}")
		
		# 记录批次开始时间
		self.last_batch_start = time.time()
		
		# 生成批次ID
		batch_id = id(batch)
		
		# 放入输入队列
		self.input_queue.put((batch, batch_id))
		
		# 等待结果
		while True:
			try:
				result, result_id = self.output_queue.get(timeout=10.0)
				self.output_queue.task_done()
				
				if result_id == batch_id:
					# 计算批次处理延迟
					latency = time.time() - self.last_batch_start
					
					# 缓存结果用于反向传播
					if is_training:
						with self.cache_lock:
							self.result_cache[batch_id] = {
								'result': result,
								'time': time.time()
							}
					
					return result
				else:
					# 不是我们等待的结果，放回队列
					self.output_queue.put((result, result_id))
					time.sleep(0.001)
			
			except queue.Empty:
				if not self.running or self.exception:
					raise RuntimeError(
						f"Timeout waiting for result, pipeline may be stalled. Error: {str(self.exception)}")
				continue
	
	def backward(self, loss, batch):
		"""
		执行反向传播

		参数:
			loss: 损失值
			batch: 对应的输入批次
		"""
		# 如果不在训练模式，直接返回
		if not self.is_training:
			return
		
		# 找到对应的缓存结果
		batch_id = id(batch)
		
		with self.cache_lock:
			if batch_id not in self.result_cache:
				logger.warning(f"No cached result found for batch {batch_id}, using standard backward")
				loss.backward()
				return
			
			# 获取缓存的结果
			cached = self.result_cache[batch_id]
			
			# 执行反向传播
			if 'segmentation_head' in self.stages and self.node_rank == 1 and self.local_rank == 2:
				# 最后一个阶段处理反向传播
				self.stages['segmentation_head'].backward(loss)
			else:
				# 其他阶段不需要特殊处理
				loss.backward()
			
			# 清理缓存
			del self.result_cache[batch_id]
	
	def _execute_preprocessing_stage(self, batch):
		"""执行预处理阶段"""
		if 'preprocessing' not in self.stages:
			return batch
		
		return self.stages['preprocessing'].forward(batch)
	
	def _execute_patch_scheduling_stage(self, preprocessed_data):
		"""
		执行Patch调度阶段

		参数:
			preprocessed_data: 预处理阶段的输出
		"""
		if 'patch_scheduling' not in self.stages:
			return preprocessed_data
		
		# 解构预处理结果
		if isinstance(preprocessed_data, tuple) and len(preprocessed_data) == 2:
			processed_images, labels = preprocessed_data
		else:
			processed_images, labels = preprocessed_data, None
		
		return self.stages['patch_scheduling'].forward(processed_images, labels)
	
	def _execute_ch_branch_stage(self, scheduling_output):
		"""
		执行CH分支阶段

		参数:
			scheduling_output: Patch调度阶段的输出
		"""
		if 'ch_branch' not in self.stages:
			return scheduling_output
		
		# 解构调度结果
		if isinstance(scheduling_output, tuple) and len(scheduling_output) == 2:
			patches, case_patches = scheduling_output
		else:
			patches, case_patches = scheduling_output, None
		
		return self.stages['ch_branch'].forward(patches)
	
	def _execute_spatial_branch_stage(self, scheduling_output):
		"""
		执行空间分支阶段

		参数:
			scheduling_output: Patch调度阶段的输出 (与CH分支共享输入)
		"""
		if 'spatial_branch' not in self.stages:
			return scheduling_output
		
		# 解构调度结果
		if isinstance(scheduling_output, tuple) and len(scheduling_output) == 2:
			patches, case_patches = scheduling_output
			
			# 获取tiers (如果CH分支已经处理)
			tiers = [patch['tier'] for patch in patches] if isinstance(patches, list) else None
			
			return self.stages['spatial_branch'].forward(patches, tiers)
		else:
			return self.stages['spatial_branch'].forward(scheduling_output)
	
	def _execute_feature_fusion_stage(self, input_data):
		"""
		执行特征融合阶段

		参数:
			input_data: 从前一阶段接收的数据，可能是CH特征或需要接收从node_comm
		"""
		if 'feature_fusion' not in self.stages:
			return input_data
		
		# 在节点2的第一个GPU上，需要从两个分支接收数据
		if self.node_rank == 1 and self.local_rank == 0:
			# 从分布式通信接收数据
			ch_features, ch_tiers = self._receive_ch_features()
			spatial_features, spatial_tiers = self._receive_spatial_features()
			
			# 确保tiers匹配
			if ch_tiers and spatial_tiers and ch_tiers != spatial_tiers:
				logger.warning("Tiers from CH branch and spatial branch don't match")
			
			tiers = ch_tiers if ch_tiers else spatial_tiers
			
			return self.stages['feature_fusion'].forward(ch_features, spatial_features, tiers)
		else:
			# 直接使用上一阶段的输出
			return self.stages['feature_fusion'].forward(input_data)
	
	def _execute_multiscale_fusion_stage(self, fusion_output):
		"""
		执行多尺度融合阶段

		参数:
			fusion_output: 特征融合阶段的输出
		"""
		if 'multiscale_fusion' not in self.stages:
			return fusion_output
		
		return self.stages['multiscale_fusion'].forward(fusion_output)
	
	def _execute_segmentation_head_stage(self, multiscale_output):
		"""
		执行分割头阶段

		参数:
			multiscale_output: 多尺度融合阶段的输出
		"""
		if 'segmentation_head' not in self.stages:
			return multiscale_output
		
		# 获取原始标签
		labels = self._get_labels_for_segmentation()
		
		return self.stages['segmentation_head'].forward(multiscale_output, labels)
	
	def _receive_ch_features(self):
		"""从CH分支接收特征"""
		# 实现从节点1的GPU 2接收CH特征的代码
		if not hasattr(self.node_comm, 'recv_tensor'):
			logger.warning("node_comm does not have recv_tensor method")
			return None, None
		
		try:
			# 从源节点接收数据
			ch_source_rank = self.node_comm.node_ranks[0] + 2  # 节点1的GPU 2
			
			# 接收features数量
			count_tensor = self.node_comm.recv_tensor(
				src_rank=ch_source_rank,
				dtype=torch.long,
				device=f"cuda:{self.local_rank}"
			)
			ch_count = count_tensor.item()
			
			# 接收每个CH特征
			ch_features = []
			ch_tiers = []
			
			for i in range(ch_count):
				# 接收CH特征
				ch_feat = self.node_comm.recv_tensor(
					src_rank=ch_source_rank,
					device=f"cuda:{self.local_rank}"
				)
				
				# 接收tier信息
				tier_tensor = self.node_comm.recv_tensor(
					src_rank=ch_source_rank,
					dtype=torch.long,
					device=f"cuda:{self.local_rank}"
				)
				
				ch_features.append(ch_feat)
				ch_tiers.append(tier_tensor.item())
			
			return ch_features, ch_tiers
		
		except Exception as e:
			logger.error(f"Error receiving CH features: {str(e)}")
			return None, None
	
	def _receive_spatial_features(self):
		"""从空间分支接收特征"""
		# 实现从节点1的GPU 3接收空间特征的代码
		if not hasattr(self.node_comm, 'recv_tensor'):
			logger.warning("node_comm does not have recv_tensor method")
			return None, None
		
		try:
			# 从源节点接收数据
			spatial_source_rank = self.node_comm.node_ranks[0] + 3  # 节点1的GPU 3
			
			# 接收features数量
			count_tensor = self.node_comm.recv_tensor(
				src_rank=spatial_source_rank,
				dtype=torch.long,
				device=f"cuda:{self.local_rank}"
			)
			spatial_count = count_tensor.item()
			
			# 接收每个空间特征
			spatial_features = []
			spatial_tiers = []
			
			for i in range(spatial_count):
				# 接收空间特征
				spatial_feat = self.node_comm.recv_tensor(
					src_rank=spatial_source_rank,
					device=f"cuda:{self.local_rank}"
				)
				
				# 接收tier信息
				tier_tensor = self.node_comm.recv_tensor(
					src_rank=spatial_source_rank,
					dtype=torch.long,
					device=f"cuda:{self.local_rank}"
				)
				
				spatial_features.append(spatial_feat)
				spatial_tiers.append(tier_tensor.item())
			
			return spatial_features, spatial_tiers
		
		except Exception as e:
			logger.error(f"Error receiving spatial features: {str(e)}")
			return None, None
	
	def _get_labels_for_segmentation(self):
		"""获取分割用的标签"""
		# 在实际应用中，可能需要从node_comm接收标签
		# 这里简化为返回None
		return None
	
	def _update_stage_time(self, stage_name, time_taken):
		"""更新阶段处理时间统计"""
		if stage_name not in self.stage_times:
			self.stage_times[stage_name] = []
		
		self.stage_times[stage_name].append(time_taken)
		
		# 保持统计列表的合理大小
		if len(self.stage_times[stage_name]) > 100:
			self.stage_times[stage_name] = self.stage_times[stage_name][-100:]
	
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
		获取性能统计

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
			'stages': {}
		}
		
		# 计算各阶段的平均时间
		for stage_name, times in self.stage_times.items():
			if times:
				avg_time = sum(times) / len(times)
				stats['stages'][stage_name] = {
					'avg_time_ms': avg_time * 1000,  # 转换为毫秒
					'samples': len(times)
				}
		
		# 收集各阶段的详细统计
		for name, stage in self.stages.items():
			if hasattr(stage, 'get_stats'):
				stage_stats = stage.get_stats()
				if name in stats['stages']:
					stats['stages'][name].update(stage_stats)
				else:
					stats['stages'][name] = stage_stats
		
		return stats
	
	def reset_stats(self):
		"""重置性能统计"""
		self.batch_times = []
		self.stage_times = {}
		self.processed_batches = 0
		
		for stage in self.stages.values():
			if hasattr(stage, 'reset_stats'):
				stage.reset_stats()
	
	def __del__(self):
		"""析构函数，确保停止工作线程"""
		self.stop_worker()