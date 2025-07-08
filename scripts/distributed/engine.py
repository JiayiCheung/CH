import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.cuda.comm as comm
from torch.cuda.amp import autocast, GradScaler
import threading
import queue
import time
from collections import deque
import numpy as np
from typing import List, Dict, Tuple, Any, Optional, Union

from .stages import FrontendStage, CHProcessingStage, SpatialFusionStage, BackendStage
from .pipeline import StagePipeline


class DistributedEngine:
	"""分布式执行引擎 - 协调4个GPU上的处理阶段"""
	
	def __init__(self, model, gpus=[0, 1, 2, 3], amp_enabled=True):
		"""
		初始化分布式执行引擎

		参数:
			model: 完整的VesselSegmenter模型
			gpus: 使用的GPU ID列表 (默认: [0,1,2,3])
			amp_enabled: 是否启用自动混合精度
		"""
		self.model = model
		self.gpus = gpus
		self.amp_enabled = amp_enabled
		self.num_stages = len(gpus)
		
		# 确保GPU数量足够
		assert len(gpus) >= 4, f"需要至少4个GPU，但只有{len(gpus)}个"
		
		# 创建处理阶段
		self.stages = self._create_stages(model)
		
		# 创建流水线
		self.pipeline = StagePipeline(self.stages)
		
		# 训练/评估状态
		self.is_training = True
		
		# 性能指标
		self.throughput = 0
		self.latency = 0
		self.memory_usage = [0] * len(gpus)
		
		print(f"分布式执行引擎初始化完成，使用{len(gpus)}个GPU: {gpus}")
	
	def _create_stages(self, model):
		"""
		创建处理阶段，并将模型部分分配到不同GPU

		参数:
			model: 完整模型

		返回:
			处理阶段列表
		"""
		stages = []
		
		# GPU 0: 前端处理 (预处理、FFT、柱坐标映射)
		stages.append(FrontendStage(model, device=f'cuda:{self.gpus[0]}'))
		
		# GPU 1: CH核心处理 (CH分解、CH系数注意力)
		stages.append(CHProcessingStage(model, device=f'cuda:{self.gpus[1]}'))
		
		# GPU 2: 空间处理与融合 (空间分支、特征融合)
		stages.append(SpatialFusionStage(model, device=f'cuda:{self.gpus[2]}'))
		
		# GPU 3: 后端处理 (多尺度融合、分割头)
		stages.append(BackendStage(model, device=f'cuda:{self.gpus[3]}'))
		
		return stages
	
	def train(self):
		"""设置为训练模式"""
		self.is_training = True
		for stage in self.stages:
			stage.train()
	
	def eval(self):
		"""设置为评估模式"""
		self.is_training = False
		for stage in self.stages:
			stage.eval()
	
	def forward(self, batch):
		"""
		前向传播 - 协调各处理阶段

		参数:
			batch: 输入批次数据

		返回:
			模型输出
		"""
		# 由流水线管理前向传播
		return self.pipeline.forward(batch, self.is_training, self.amp_enabled)
	
	def backward(self, loss):
		"""
		反向传播 - 协调梯度计算与传递

		参数:
			loss: 损失值
		"""
		# 由流水线管理反向传播
		self.pipeline.backward(loss)
	
	def pause_pipeline(self):
		"""暂停流水线用于评估"""
		self.pipeline.pause()
	
	def resume_pipeline(self):
		"""恢复训练流水线"""
		self.pipeline.resume()
	
	def set_tier(self, tier):
		"""
		设置当前tier

		参数:
			tier: tier编号
		"""
		self.pipeline.set_tier(tier)
	
	def clear_tier_features(self):
		"""清除所有阶段的tier特征缓存"""
		self.pipeline.clear_tier_features()
	
	def get_consolidated_model(self):
		"""
		将分布在不同GPU上的模型部分合并为完整模型

		返回:
			完整的VesselSegmenter模型
		"""
		from models.vessel_segmenter import VesselSegmenter
		
		# 创建新模型
		consolidated_model = VesselSegmenter()
		
		# 收集各阶段参数
		state_dict = {}
		for stage in self.stages:
			stage_dict = stage.get_state_dict_prefix()
			state_dict.update(stage_dict)
		
		# 加载参数
		consolidated_model.load_state_dict(state_dict)
		
		return consolidated_model
	
	def update_memory_stats(self):
		"""更新内存使用统计"""
		for i, gpu in enumerate(self.gpus):
			self.memory_usage[i] = torch.cuda.memory_allocated(gpu) / (1024 ** 3)  # GB
	
	def get_performance_stats(self):
		"""
		获取性能统计信息

		返回:
			性能统计字典
		"""
		self.update_memory_stats()
		
		# 获取流水线各阶段统计
		pipeline_stats = self.pipeline.get_performance_stats()
		
		return {
			'throughput': self.throughput,  # 样本/秒
			'latency': self.latency,  # 毫秒/样本
			'memory_usage': self.memory_usage,
			'pipeline': pipeline_stats
		}
	
	def reset_stats(self):
		"""重置性能统计"""
		self.throughput = 0
		self.latency = 0
		self.pipeline.reset_stats()