import torch
import torch.cuda.comm as comm
import threading
import queue
import time
from collections import deque
import numpy as np
from typing import List, Dict, Tuple, Any, Optional


class StagePipeline:
	"""管理处理阶段间的数据流动和同步"""
	
	def __init__(self, stages):
		"""
		初始化流水线管理器

		参数:
			stages: 处理阶段列表
		"""
		self.stages = stages
		self.num_stages = len(stages)
		
		# 阶段间队列 (stage i → stage i+1)
		self.queues = self._create_queues()
		
		# 缓存区 (用于反向传播)
		self.cache = {}
		self.cache_lock = threading.Lock()
		
		# 流水线状态
		self.active = True
		self.pipeline_lock = threading.Lock()
		
		# 用于评估的锁
		self.eval_lock = threading.Lock()
		
		# 当前tier
		self.current_tier = None
		
		print(f"StagePipeline 初始化：{self.num_stages}个阶段")
	
	def _create_queues(self):
		"""
		创建阶段间通信队列

		返回:
			队列列表
		"""
		queues = []
		for i in range(self.num_stages - 1):
			queues.append(queue.Queue(maxsize=2))  # 限制队列大小防止内存溢出
		return queues
	
	def set_tier(self, tier):
		"""
		设置当前tier

		参数:
			tier: tier编号
		"""
		self.current_tier = tier
		for stage in self.stages:
			if hasattr(stage, 'set_tier'):
				stage.set_tier(tier)
	
	def clear_tier_features(self):
		"""清除所有阶段的tier特征缓存"""
		for stage in self.stages:
			if hasattr(stage, 'clear_tier_features'):
				stage.clear_tier_features()
			elif hasattr(stage, 'tier_features'):
				stage.tier_features.clear()
	
	def forward(self, batch, is_training=True, amp_enabled=False):
		"""
		执行前向传播

		参数:
			batch: 输入批次
			is_training: 是否处于训练模式
			amp_enabled: 是否启用混合精度

		返回:
			模型输出
		"""
		# 评估模式下，使用顺序处理避免状态混乱
		if not is_training:
			with self.eval_lock:
				return self._sequential_forward(batch, amp_enabled)
		
		# 如果流水线暂停，回退到顺序处理
		if not self.active:
			with self.eval_lock:
				return self._sequential_forward(batch, amp_enabled)
		
		# 流水线模式（目前简化为顺序处理）
		# 在完整实现中，这里应该实现真正的流水线处理
		return self._sequential_forward(batch, amp_enabled)
	
	def _sequential_forward(self, batch, amp_enabled=False):
		"""
		顺序前向处理(非流水线)

		参数:
			batch: 输入批次
			amp_enabled: 是否启用混合精度

		返回:
			模型输出
		"""
		# 阶段1: 前端处理
		frontend_output, input_reference = self.stages[0].forward(batch, amp_enabled)
		
		# 阶段2: CH处理
		ch_output, ch_tensors = self.stages[1].forward((frontend_output, input_reference), amp_enabled)
		
		# 阶段3: 空间处理与融合
		fusion_output, fusion_tensors = self.stages[2].forward((ch_output, ch_tensors), batch, amp_enabled)
		
		# 阶段4: 后端处理
		output = self.stages[3].forward((fusion_output, fusion_tensors), amp_enabled)
		
		# 缓存中间结果用于反向传播
		with self.cache_lock:
			batch_id = id(batch)
			self.cache[batch_id] = {
				'input_reference': input_reference,
				'ch_tensors': ch_tensors,
				'fusion_tensors': fusion_tensors
			}
		
		return output
	
	def _pipelined_forward(self, batch, amp_enabled=False):
		"""
		流水线前向处理(未实现)

		参数:
			batch: 输入批次
			amp_enabled: 是否启用混合精度

		返回:
			模型输出
		"""
		# 这里应该实现真正的流水线处理
		# 利用多线程和队列实现并行处理
		# 由于实现复杂度较高，这里暂时简化为顺序处理
		return self._sequential_forward(batch, amp_enabled)
	
	def backward(self, loss):
		"""
		执行反向传播

		参数:
			loss: 损失值
		"""
		# 简化版：直接调用标准的backward
		loss.backward()
	
	# 在实际实现中，需要协调各阶段的反向传播
	# 并正确地在阶段间传递梯度
	# 这里省略详细实现
	
	def pause(self):
		"""暂停流水线"""
		with self.pipeline_lock:
			self.active = False
			
			# 等待所有队列清空
			for q in self.queues:
				while not q.empty():
					time.sleep(0.01)
	
	def resume(self):
		"""恢复流水线"""
		with self.pipeline_lock:
			self.active = True
	
	def get_performance_stats(self):
		"""
		获取各阶段性能统计

		返回:
			性能统计字典
		"""
		stats = {}
		for i, stage in enumerate(self.stages):
			stats[f"stage_{i}"] = stage.get_stats()
		return stats
	
	def reset_stats(self):
		"""重置所有阶段的性能统计"""
		for stage in self.stages:
			stage.reset_stats()