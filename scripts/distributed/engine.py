import torch
import torch.nn as nn
from .stages import (
	PreprocessingStage, PatchSchedulingStage, CHBranchStage,
	SpatialBranchStage, FeatureFusionStage, MultiscaleFusionStage,
	SegmentationHeadStage
)
from .node_communicator import NodeCommunicator
from .cross_node_pipeline import CrossNodePipeline


class CrossNodeDistributedEngine:
	"""跨节点分布式执行引擎，跨两个节点的7个GPU协调执行"""
	
	def __init__(self, model, world_size, rank, local_rank, node_rank, node_count=2):
		"""
		初始化跨节点分布式引擎

		参数:
			model: 完整模型
			world_size: 总进程数
			rank: 全局进程编号
			local_rank: 节点内GPU编号
			node_rank: 节点编号 (0或1)
			node_count: 总节点数
		"""
		self.model = model
		self.world_size = world_size
		self.rank = rank
		self.local_rank = local_rank
		self.node_rank = node_rank
		self.node_count = node_count
		
		# 创建节点通信管理器
		self.node_comm = NodeCommunicator(
			world_size=world_size,
			rank=rank,
			local_rank=local_rank,
			node_rank=node_rank,
			node_count=node_count
		)
		
		# 创建处理阶段
		self.stages = self._create_stages(model)
		
		# 创建跨节点流水线
		self.pipeline = CrossNodePipeline(
			self.stages,
			self.node_comm
		)
		
		# 训练/评估状态
		self.is_training = True
		self.current_tier = None
		
		print(f"跨节点分布式执行引擎初始化完成，节点{node_rank}，本地GPU {local_rank}，全局排名{rank}")
	
	def _create_stages(self, model):
		"""
		创建适合当前节点和GPU的处理阶段

		参数:
			model: 完整模型

		返回:
			处理阶段字典
		"""
		stages = {}
		
		# 节点1: GPU 0, 1, 2, 3
		if self.node_rank == 0:
			if self.local_rank == 0:
				# GPU 0: 数据预处理+ROI提取
				stages['preprocessing'] = PreprocessingStage(
					model=model,
					device=f'cuda:{self.local_rank}',
					node_comm=self.node_comm
				)
			
			elif self.local_rank == 1:
				# GPU 1: 三级采样+Patch调度
				stages['patch_scheduling'] = PatchSchedulingStage(
					model=model,
					device=f'cuda:{self.local_rank}',
					node_comm=self.node_comm
				)
			
			elif self.local_rank == 2:
				# GPU 2: CH分支
				stages['ch_branch'] = CHBranchStage(
					model=model,
					device=f'cuda:{self.local_rank}',
					node_comm=self.node_comm
				)
			
			elif self.local_rank == 3:
				# GPU 3: 空间分支
				stages['spatial_branch'] = SpatialBranchStage(
					model=model,
					device=f'cuda:{self.local_rank}',
					node_comm=self.node_comm
				)
		
		# 节点2: GPU 0, 1, 2 (映射到 GPU 4, 5, 6)
		else:
			if self.local_rank == 0:
				# GPU 4: 特征融合
				stages['feature_fusion'] = FeatureFusionStage(
					model=model,
					device=f'cuda:{self.local_rank}',
					node_comm=self.node_comm
				)
			
			elif self.local_rank == 1:
				# GPU 5: 多尺度融合
				stages['multiscale_fusion'] = MultiscaleFusionStage(
					model=model,
					device=f'cuda:{self.local_rank}',
					node_comm=self.node_comm
				)
			
			elif self.local_rank == 2:
				# GPU 6: 分割头和损失计算
				stages['segmentation_head'] = SegmentationHeadStage(
					model=model,
					device=f'cuda:{self.local_rank}',
					node_comm=self.node_comm
				)
		
		return stages
	
	def train(self):
		"""设置为训练模式"""
		self.is_training = True
		for stage in self.stages.values():
			stage.train()
	
	def eval(self):
		"""设置为评估模式"""
		self.is_training = False
		for stage in self.stages.values():
			stage.eval()
	
	def set_tier(self, tier):
		"""
		设置当前tier

		参数:
			tier: tier编号
		"""
		self.current_tier = tier
		
		# 通知所有阶段
		for stage in self.stages.values():
			if hasattr(stage, 'set_tier'):
				stage.set_tier(tier)
	
	def forward(self, batch):
		"""
		前向传播

		参数:
			batch: 输入批次

		返回:
			模型输出
		"""
		return self.pipeline.forward(batch, self.is_training)
	
	def backward(self, loss):
		"""
		反向传播

		参数:
			loss: 损失值
		"""
		if not self.is_training:
			return
		
		# 在节点2的分割头GPU上执行反向传播
		if self.node_rank == 1 and self.local_rank == 2:
			loss.backward()
	
	def get_consolidated_model(self):
		"""
		将分布在不同GPU上的模型部分合并为完整模型

		返回:
			完整模型
		"""
		# 收集所有节点的参数
		global_state_dict = {}
		
		if self.node_rank == 0:
			# 节点1收集本地参数
			local_state_dict = {}
			for name, stage in self.stages.items():
				if hasattr(stage, 'get_state_dict_prefix'):
					local_state_dict.update(stage.get_state_dict_prefix())
			
			# 将参数发送到节点2
			if self.local_rank == 0:  # 只在GPU 0上执行
				master_rank_node2 = self.node_comm.procs_per_node
				
				# 序列化状态字典
				buffer = io.BytesIO()
				torch.save(local_state_dict, buffer)
				data = buffer.getvalue()
				
				# 发送状态字典大小
				size_tensor = torch.tensor([len(data)], dtype=torch.long, device='cuda')
				self.node_comm.send_tensor(size_tensor, dst_rank=master_rank_node2)
				
				# 发送状态字典
				data_tensor = torch.ByteTensor(list(data)).to('cuda')
				self.node_comm.send_tensor(data_tensor, dst_rank=master_rank_node2)
		
		else:  # 节点2
			# 接收节点1的参数
			if self.local_rank == 0:  # 只在GPU 0上执行
				master_rank_node1 = 0
				
				# 接收状态字典大小
				size_tensor = self.node_comm.recv_tensor(
					src_rank=master_rank_node1,
					dtype=torch.long,
					device='cuda'
				)
				size = size_tensor.item()
				
				# 接收状态字典
				data_tensor = self.node_comm.recv_tensor(
					src_rank=master_rank_node1,
					dtype=torch.uint8,
					device='cuda'
				)
				
				# 反序列化状态字典
				data = bytes(data_tensor.cpu().numpy())
				buffer = io.BytesIO(data)
				node1_state_dict = torch.load(buffer)
				
				# 节点2收集本地参数
				node2_state_dict = {}
				for name, stage in self.stages.items():
					if hasattr(stage, 'get_state_dict_prefix'):
						node2_state_dict.update(stage.get_state_dict_prefix())
				
				# 合并状态字典
				global_state_dict.update(node1_state_dict)
				global_state_dict.update(node2_state_dict)
				
				# 创建新模型并加载参数
				from models.vessel_segmenter import VesselSegmenter
				consolidated_model = VesselSegmenter(
					in_channels=1,
					out_channels=1,
					ch_params=self.model.ch_params,
					tier_params=self.model.tier_params
				)
				
				consolidated_model.load_state_dict(global_state_dict)
				return consolidated_model
		
		return None