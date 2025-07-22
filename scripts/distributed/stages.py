import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import threading
from queue import Queue
import logging
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple, Any


class BaseStage(nn.Module):
	"""处理阶段基类，提供共享功能"""
	
	def __init__(self, name: str, device: str, node_comm=None):
		"""初始化处理阶段"""
		super().__init__()
		self.name = name
		self.device = device
		self.node_comm = node_comm
		self.training = True
		
		# 性能监控
		self.compute_time = 0
		self.transfer_time = 0
		self.batch_count = 0
		self.last_error = None
		
		# 输入输出队列
		self.input_queue = Queue(maxsize=4)
		self.output_queue = Queue(maxsize=4)
		
		# 工作线程
		self.worker_thread = None
		self.running = False
		
		# 🔥 新增：日志记录器
		self.logger = logging.getLogger(f"{__name__}.{name}")
		
		# 🔥 新增：通信统计
		self.comm_success_count = 0
		self.comm_failure_count = 0
	
	def train(self, mode=True):
		"""设置训练模式"""
		self.training = mode
		return super().train(mode)
	
	def eval(self):
		"""设置评估模式"""
		self.training = False
		return super().eval()
	
	def reset_stats(self):
		"""重置性能统计"""
		self.compute_time = 0
		self.transfer_time = 0
		self.batch_count = 0
	
	def get_stats(self):
		"""获取性能统计"""
		avg_compute = self.compute_time / max(1, self.batch_count)
		avg_transfer = self.transfer_time / max(1, self.batch_count)
		return {
			'name': self.name,
			'device': self.device,
			'avg_compute_ms': avg_compute * 1000,
			'avg_transfer_ms': avg_transfer * 1000,
			'batch_count': self.batch_count,
			'last_error': str(self.last_error) if self.last_error else None
		}
	
	def start_worker(self):
		"""启动工作线程"""
		if self.worker_thread is not None and self.worker_thread.is_alive():
			return
		
		self.running = True
		self.worker_thread = threading.Thread(
			target=self._worker_loop,
			daemon=True,
			name=f"{self.name}_worker"
		)
		self.worker_thread.start()
	
	def stop_worker(self):
		"""停止工作线程"""
		self.running = False
		if self.worker_thread and self.worker_thread.is_alive():
			self.worker_thread.join(timeout=3.0)
	
	def _worker_loop(self):
		"""工作线程主循环"""
		while self.running:
			try:
				# 获取输入数据
				if self.input_queue.empty():
					time.sleep(0.001)  # 避免CPU忙等
					continue
				
				inputs = self.input_queue.get(timeout=1.0)
				
				# 处理数据
				outputs = self.process(*inputs)
				
				# 发送输出
				if outputs is not None:
					self.output_queue.put(outputs)
			
			except Exception as e:
				self.last_error = e
				
				time.sleep(1.0)  # 避免错误循环消耗资源
	
	def process(self, *args, **kwargs):
		"""处理逻辑(由子类实现)"""
		raise NotImplementedError("子类必须实现process方法")
	
	def forward(self, *args, **kwargs):
		"""同步前向处理(由子类实现)"""
		raise NotImplementedError("子类必须实现forward方法")
	
	def get_state_dict_prefix(self):
		"""获取带前缀的参数字典"""
		return {}
	
	def log_communication_stats(self):
		"""记录通信状态统计"""
		if hasattr(self.node_comm, 'get_detailed_stats'):
			stats = self.node_comm.get_detailed_stats()
			
			self.logger.info(f"📊 {self.name} 通信状态:")
			self.logger.info(f"  成功: {self.comm_success_count}")
			self.logger.info(f"  失败: {self.comm_failure_count}")
			self.logger.info(f"  发送失败: {stats.get('send_failures', 0)}")
			self.logger.info(f"  接收失败: {stats.get('recv_failures', 0)}")
			
			if stats.get('send_failures', 0) + stats.get('recv_failures', 0) > 0:
				self.logger.warning("⚠️  检测到通信错误，建议检查网络连接")
	
	def update_comm_stats(self, success: bool):
		"""更新通信统计"""
		if success:
			self.comm_success_count += 1
		else:
			self.comm_failure_count += 1


class DummyStage(BaseStage):
	"""极简占位阶段 - 只占用GPU，不参与主流水线"""
	
	def __init__(self, model, device, node_comm=None, config=None):
		super().__init__("DummyStage", device, node_comm)
		
		# 创建一个极小的"装饰"网络，占用一点GPU内存
		self.dummy_layer = nn.Linear(10, 1).to(device)
	
	def process(self, dummy_input=None):
		"""极简处理 - 什么都不做"""
		# 偶尔做个无意义的计算，防止被系统回收
		if dummy_input is None:
			dummy_input = torch.randn(1, 10, device=self.device)
		
		_ = self.dummy_layer(dummy_input)  # 扔掉结果
		return None
	
	def forward(self, *args, **kwargs):
		"""不接收任何数据，不发送任何数据"""
		time.sleep(0.001)  # 装作在工作
		return None
	
	def get_state_dict_prefix(self):
		"""返回空状态字典"""
		return {}


# 节点1 (GPU 0): 数据预处理+ROI提取
class FrontendStage(BaseStage):
	"""前端处理阶段 - 处理数据预处理和ROI提取"""
	
	def __init__(self, model, device, node_comm=None, shared_components=None, config=None):
		super().__init__("FrontendStage", device, node_comm)
		self.config = config or {}
		
		# 优先使用共享组件
		self.preprocessor = None
		if shared_components and 'preprocessor' in shared_components:
			self.preprocessor = shared_components['preprocessor']
		
		elif hasattr(model, 'preprocessor') and model.preprocessor is not None:
			self.preprocessor = model.preprocessor
		
		else:
			# 只在完全找不到时才创建新实例
			from data.processing import CTPreprocessor
			
			# 从配置中读取预处理参数
			preprocessing_config = self.config.get('preprocessing', {})
			roi_threshold = preprocessing_config.get('roi_threshold')
			roi_percentile = preprocessing_config.get('roi_percentile', 99.8)
			use_largest_cc = preprocessing_config.get('use_largest_cc', True)
			
			# 创建预处理器并传递参数
			self.preprocessor = CTPreprocessor(
				roi_threshold=roi_threshold,
				roi_percentile=roi_percentile,
				use_largest_cc=use_largest_cc,
				device=device,
			
			)
		
		# 移动到指定设备
		self.to(device)
	
	def process(self, batch):
		"""异步处理批次数据"""
		start_time = time.time()
		
		# 确保数据在正确的设备上
		images = batch['image'].to(self.device)
		labels = batch['label'].to(self.device) if 'label' in batch else None
		
		# 预处理
		processed_images = []
		for i in range(images.shape[0]):
			# 归一化
			norm_img = self.preprocessor.normalize(images[i].squeeze().cpu().numpy())
			# 提取肝脏ROI
			liver_mask = self.preprocessor.extract_liver_roi(norm_img)
			
			# 转回张量
			norm_img_tensor = torch.from_numpy(norm_img).float().unsqueeze(0).to(self.device)
			liver_mask_tensor = torch.from_numpy(liver_mask).float().to(self.device)
			
			processed_images.append({
				'image': norm_img_tensor,
				'liver_mask': liver_mask_tensor,
				'case_id': batch.get('case_id', [f"case_{i}"])[i]
			})
		
		self.compute_time += time.time() - start_time
		self.batch_count += images.shape[0]
		
		# 返回处理后的数据和原始标签
		return processed_images, labels
	
	def forward(self, batch):
		"""同步前向处理"""
		processed_images, labels = self.process(batch)
		
		# 将结果发送到下一阶段(GPU 1)
		if self.node_comm:
			next_rank = self.node_comm.rank + 1
			self.node_comm.send_tensor(
				torch.tensor([len(processed_images)], dtype=torch.long),
				dst_rank=next_rank
			)
			
			for i, proc_img in enumerate(processed_images):
				# 发送图像
				self.node_comm.send_tensor(proc_img['image'], dst_rank=next_rank)
				# 发送肝脏掩码
				self.node_comm.send_tensor(proc_img['liver_mask'], dst_rank=next_rank)
				# 发送case_id (作为元数据)
				meta = {'case_id': proc_img['case_id']}
				meta_tensor = torch.tensor([ord(c) for c in str(meta)],
				                           dtype=torch.uint8, device=self.device)
				self.node_comm.send_tensor(meta_tensor, dst_rank=next_rank)
			
			# 发送标签(如果有)
			if labels is not None:
				has_labels = torch.tensor([1], dtype=torch.long, device=self.device)
				self.node_comm.send_tensor(has_labels, dst_rank=next_rank)
				self.node_comm.send_tensor(labels, dst_rank=next_rank)
			else:
				has_labels = torch.tensor([0], dtype=torch.long, device=self.device)
				self.node_comm.send_tensor(has_labels, dst_rank=next_rank)
		
		return processed_images, labels
	
	def get_state_dict_prefix(self):
		"""获取带前缀的参数字典"""
		# 只保存必要的状态
		state_dict = {}
		return state_dict


# 节点1 (GPU 1): 三级采样+Patch调度
class PatchSchedulingStage(BaseStage):
	"""Patch采样和调度阶段"""
	
	def __init__(self, model, device, node_comm=None, shared_components=None, config=None):
		super().__init__("PatchSchedulingStage", device, node_comm)
		self.config = config or {}
		
		# 优先使用共享组件
		self.tier_sampler = None
		if shared_components and 'tier_sampler' in shared_components:
			self.tier_sampler = shared_components['tier_sampler']
		
		elif hasattr(model, 'tier_sampler') and model.tier_sampler is not None:
			self.tier_sampler = model.tier_sampler
		
		else:
			# 只在完全找不到时才创建新实例
			from data.tier_sampling import TierSampler
			
			# 从配置中读取采样参数
			sampling_config = self.config.get('smart_sampling', {})
			
			# 创建采样器并传递参数
			self.tier_sampler = TierSampler(
				tier0_size=sampling_config.get('tier0_size', 256),
				tier1_size=sampling_config.get('tier1_size', 96),
				tier2_size=sampling_config.get('tier2_size', 64),
				max_tier1=sampling_config.get('maxtier1', 10),
				max_tier2=sampling_config.get('maxtier2', 20)
			)
		
		# 移动到指定设备
		self.to(device)
		
		# 缓存已处理的patches
		self.patches_cache = {}
	
	def process(self, processed_images, labels):
		"""处理预处理后的图像数据"""
		start_time = time.time()
		
		all_patches = []
		case_patches = {}
		
		# 对每个样本进行三级采样
		for i, proc_img in enumerate(processed_images):
			# 获取case_id
			case_id = proc_img['case_id']
			
			# 检查缓存
			if case_id in self.patches_cache:
				patches = self.patches_cache[case_id]
			else:
				# 提取到CPU进行采样
				image_np = proc_img['image'].squeeze().cpu().numpy()
				liver_mask_np = proc_img['liver_mask'].cpu().numpy()
				
				# 获取当前样本的标签
				label_np = None
				if labels is not None:
					label_np = labels[i].cpu().numpy() if labels.shape[0] > i else None
				
				# 应用三级采样
				patches = self.tier_sampler.sample(
					image_np, label_np, liver_mask_np, case_id=case_id
				)
				
				# 更新缓存
				self.patches_cache[case_id] = patches
			
			all_patches.extend(patches)
			case_patches[case_id] = patches
		
		# 按Tier对patches进行排序
		sorted_patches = sorted(all_patches, key=lambda p: p['tier'])
		
		self.compute_time += time.time() - start_time
		self.batch_count += len(processed_images)
		
		return sorted_patches, case_patches
	
	def forward(self, processed_images=None, labels=None):
		"""同步前向处理"""
		if processed_images is None and self.node_comm:
			# 从上一阶段(GPU 0)接收数据
			prev_rank = self.node_comm.rank - 1
			
			# 接收样本数量
			count_tensor = self.node_comm.recv_tensor(
				src_rank=prev_rank,
				dtype=torch.long,
				device=self.device
			)
			count = count_tensor.item()
			
			# 接收每个样本
			processed_images = []
			for i in range(count):
				# 接收图像
				image = self.node_comm.recv_tensor(
					src_rank=prev_rank,
					dtype=torch.float32,
					device=self.device
				)
				
				# 接收肝脏掩码
				liver_mask = self.node_comm.recv_tensor(
					src_rank=prev_rank,
					dtype=torch.float32,
					device=self.device
				)
				
				# 接收元数据
				meta_tensor = self.node_comm.recv_tensor(
					src_rank=prev_rank,
					dtype=torch.uint8,
					device=self.device
				)
				meta_str = ''.join([chr(c) for c in meta_tensor.cpu().numpy()])
				meta = eval(meta_str)  # 安全问题：实际应用中应使用更安全的序列化方法
				
				processed_images.append({
					'image': image,
					'liver_mask': liver_mask,
					'case_id': meta['case_id']
				})
			
			# 接收标签
			has_labels = self.node_comm.recv_tensor(
				src_rank=prev_rank,
				dtype=torch.long,
				device=self.device
			).item()
			
			if has_labels:
				labels = self.node_comm.recv_tensor(
					src_rank=prev_rank,
					device=self.device
				)
			else:
				labels = None
		
		# 处理数据
		patches, case_patches = self.process(processed_images, labels)
		
		# 将patches发送到CH分支(GPU 2)和空间分支(GPU 3) - 使用新的通信方式
		if self.node_comm:
			# 准备发送数据：转换为List[Tuple[torch.Tensor, int]]格式
			patch_data = []
			for patch in patches:
				# 转换patch为张量
				if isinstance(patch['image'], np.ndarray):
					patch_tensor = torch.from_numpy(patch['image']).float().unsqueeze(0).to(self.device)
				else:
					patch_tensor = patch['image'].to(self.device)
				
				tier = int(patch['tier'])
				patch_data.append((patch_tensor, tier))
			
			# 发送到CH分支 (GPU 2)
			ch_rank = self.node_comm.rank + 1
			success_ch = self._send_patches_to_branch(patch_data, ch_rank, "CH", tag=50)
			
			# 发送到空间分支 (GPU 3)
			spatial_rank = ch_rank + 1
			success_spatial = self._send_patches_to_branch(patch_data, spatial_rank, "Spatial", tag=60)
			
			# 记录发送状态
			if success_ch and success_spatial:
				self.logger.debug(f"✅ Patches发送成功: {len(patches)}个patches到两个分支")
				self.update_comm_stats(True)
			else:
				self.logger.warning(f"⚠️  Patches发送部分失败: CH={success_ch}, Spatial={success_spatial}")
				self.update_comm_stats(False)
		
		return patches, case_patches
	
	def _send_patches_to_branch(self, patch_data: List[Tuple[torch.Tensor, int]],
	                            dst_rank: int, branch_name: str, tag: int = 0) -> bool:
		"""发送patches到指定分支"""
		try:
			# 🔥 优先使用新的复杂数据类型传输
			if hasattr(self.node_comm, 'send_tensor_tuple_list'):
				success = self.node_comm.send_tensor_tuple_list(patch_data, dst_rank=dst_rank, tag=tag)
				if success:
					self.logger.debug(f"✅ {branch_name}分支发送成功: {len(patch_data)}个patches")
					return True
				else:
					self.logger.warning(f"⚠️  {branch_name}分支发送失败，尝试回退模式")
			
			elif hasattr(self.node_comm, 'send_data'):
				success = self.node_comm.send_data(patch_data, dst_rank=dst_rank, tag=tag, reliable=True)
				if success:
					self.logger.debug(f"✅ {branch_name}分支发送成功(通用模式): {len(patch_data)}个patches")
					return True
				else:
					self.logger.warning(f"⚠️  {branch_name}分支发送失败，尝试回退模式")
			
			# 回退到原有发送方式
			return self._fallback_send_patches(patch_data, dst_rank, branch_name)
		
		except Exception as e:
			self.logger.error(f"❌ {branch_name}分支发送异常: {e}")
			return self._fallback_send_patches(patch_data, dst_rank, branch_name)
	
	def _fallback_send_patches(self, patch_data: List[Tuple[torch.Tensor, int]],
	                           dst_rank: int, branch_name: str) -> bool:
		"""回退发送方式"""
		try:
			# 发送patches数量
			count_tensor = torch.tensor([len(patch_data)], dtype=torch.long, device=self.device)
			self.node_comm.send_tensor(count_tensor, dst_rank=dst_rank, reliable=False)
			
			# 逐个发送patches
			for patch_tensor, tier in patch_data:
				self.node_comm.send_tensor(patch_tensor, dst_rank=dst_rank, reliable=False)
				tier_tensor = torch.tensor([tier], dtype=torch.long, device=self.device)
				self.node_comm.send_tensor(tier_tensor, dst_rank=dst_rank, reliable=False)
			
			self.logger.debug(f"✅ {branch_name}分支发送成功(回退模式): {len(patch_data)}个patches")
			return True
		
		except Exception as e:
			self.logger.error(f"❌ {branch_name}分支回退发送也失败: {e}")
			return False
	
	def get_state_dict_prefix(self):
		"""获取带前缀的参数字典"""
		# 只保存必要的状态
		state_dict = {}
		return state_dict


# 节点1 (GPU 2): CH分支完整处理
class CHProcessingStage(BaseStage):
	"""CH处理阶段 - 处理CH分支"""
	
	def __init__(self, model, device, node_comm=None, shared_components=None, config=None):
		super().__init__("CHProcessingStage", device, node_comm)
		self.config = config or {}
		
		# 提取CH分支组件 - 这些通常来自模型，不是独立初始化的
		self.ch_branch = model.ch_branch
		self.tier_params = model.tier_params
		
		# 移动到指定设备
		self.ch_branch.to(device)
	
	def process(self, patches, tiers):
		"""处理patches - 完整实现CH分支"""
		start_time = time.time()
		
		ch_features = []
		processed_tiers = []
		
		for patch, tier in zip(patches, tiers):
			try:
				# 1. 设置当前tier的CH参数
				if hasattr(self.ch_branch, 'set_tier'):
					self.ch_branch.set_tier(tier)
				
				# 2. 获取tier特定的参数
				tier_params = self.tier_params.get(tier, {})
				r_scale = tier_params.get('r_scale', 1.0)
				
				# 3. 确保输入是张量格式
				if isinstance(patch, np.ndarray):
					patch_tensor = torch.from_numpy(patch).float().unsqueeze(0).to(self.device)
				else:
					patch_tensor = patch.to(self.device)
				
				# 4. CH分支前向传播
				if hasattr(self.ch_branch, '__call__'):
					# 检查ch_branch是否支持r_scale参数
					import inspect
					sig = inspect.signature(self.ch_branch.forward)
					if 'r_scale' in sig.parameters:
						ch_output = self.ch_branch(patch_tensor, r_scale=r_scale)
					else:
						ch_output = self.ch_branch(patch_tensor)
				else:
					ch_output = patch_tensor  # 备用方案
				
				ch_features.append(ch_output)
				processed_tiers.append(tier)
			
			except Exception as e:
				print(f"CH processing failed for tier {tier}: {e}")
				continue
		
		self.compute_time += time.time() - start_time
		self.batch_count += len(patches)
		
		return ch_features, processed_tiers
	
	def forward(self, patches=None, tiers=None):
		"""同步前向处理 - 优化版"""
		if patches is None and self.node_comm:
			# 从PatchSchedulingStage接收数据
			prev_rank = self.node_comm.rank - 1
			
			# 1. 接收数据包数量
			count_tensor = self.node_comm.recv_tensor(
				src_rank=prev_rank,
				dtype=torch.long,
				device=self.device
			)
			count = count_tensor.item()
			
			# 2. 批量接收所有patches
			patches = []
			tiers = []
			
			for i in range(count):
				# 接收patch张量
				patch_tensor = self.node_comm.recv_tensor(
					src_rank=prev_rank,
					dtype=torch.float32,
					device=self.device
				)
				
				# 接收tier信息
				tier_tensor = self.node_comm.recv_tensor(
					src_rank=prev_rank,
					dtype=torch.long,
					device=self.device
				)
				
				patches.append(patch_tensor)
				tiers.append(tier_tensor.item())
		
		# 处理数据
		ch_features, processed_tiers = self.process(patches, tiers)
		
		# 发送到FeatureFusionStage - 使用新的复杂数据类型传输
		if self.node_comm:
			# 目标rank计算（节点2的GPU 4）
			fusion_rank = self.node_comm.node_ranks[1] if hasattr(self.node_comm,
			                                                      'node_ranks') else self.node_comm.rank + 2
			
			try:
				# 🔥 新功能：直接发送List[Tuple[torch.Tensor, int]]格式
				ch_data = [(ch_feat, tier) for ch_feat, tier in zip(ch_features, processed_tiers)]
				
				# 检查是否支持新的复杂数据类型传输
				if hasattr(self.node_comm, 'send_tensor_tuple_list'):
					# 使用专门的方法发送tensor-tuple列表
					success = self.node_comm.send_tensor_tuple_list(ch_data, dst_rank=fusion_rank, tag=100)
					
					if success:
						self.logger.debug(f"✅ CH特征发送成功: {len(ch_data)}个特征")
						self.update_comm_stats(True)
					else:
						self.logger.error("❌ CH特征发送失败，尝试回退模式")
						self.update_comm_stats(False)
						# 回退到原有发送方式
						self._fallback_send_features(ch_features, processed_tiers, fusion_rank)
				
				elif hasattr(self.node_comm, 'send_data'):
					# 使用通用的复杂数据发送方法
					success = self.node_comm.send_data(ch_data, dst_rank=fusion_rank, tag=100, reliable=True)
					
					if success:
						self.logger.debug(f"✅ CH特征发送成功(通用模式): {len(ch_data)}个特征")
						self.update_comm_stats(True)
					else:
						self.logger.error("❌ CH特征发送失败，尝试回退模式")
						self.update_comm_stats(False)
						self._fallback_send_features(ch_features, processed_tiers, fusion_rank)
				
				else:
					# 回退到原有发送方式
					self.logger.warning("⚠️  使用原有发送方式(不支持复杂数据类型)")
					self._fallback_send_features(ch_features, processed_tiers, fusion_rank)
			
			except Exception as e:
				self.logger.error(f"❌ CH特征发送异常: {e}")
				self.update_comm_stats(False)
				# 回退到原有发送方式
				self._fallback_send_features(ch_features, processed_tiers, fusion_rank)
		
		return ch_features, processed_tiers
	
	def _fallback_send_features(self, ch_features, processed_tiers, fusion_rank):
		"""回退发送方式 - 兼容原有通信方式"""
		try:
			# 发送特征数量
			count_tensor = torch.tensor([len(ch_features)], dtype=torch.long, device=self.device)
			self.node_comm.send_tensor(count_tensor, dst_rank=fusion_rank, reliable=False)
			
			# 逐个发送特征
			for ch_feat, tier in zip(ch_features, processed_tiers):
				self.node_comm.send_tensor(ch_feat, dst_rank=fusion_rank, reliable=False)
				
				tier_tensor = torch.tensor([tier], dtype=torch.long, device=self.device)
				self.node_comm.send_tensor(tier_tensor, dst_rank=fusion_rank, reliable=False)
			
			self.logger.debug(f"✅ CH特征发送成功(回退模式): {len(ch_features)}个特征")
		
		except Exception as e:
			self.logger.error(f"❌ 回退发送也失败: {e}")
	
	def get_state_dict_prefix(self):
		"""获取带前缀的参数字典"""
		# 保存CH分支参数
		state_dict = {}
		for name, param in self.ch_branch.state_dict().items():
			state_dict[f'ch_branch.{name}'] = param
		return state_dict


# 节点1 (GPU 3): 空间分支完整处理
class SpatialFusionStage(BaseStage):
	"""空间融合阶段 - 处理空间分支"""
	
	def __init__(self, model, device, node_comm=None, shared_components=None, config=None):
		super().__init__("SpatialFusionStage", device, node_comm)
		self.config = config or {}
		
		# 提取空间分支组件 - 这些通常来自模型，不是独立初始化的
		self.spatial_branch = model.spatial_branch
		self.edge_enhance = model.edge_enhance
		
		# 移动到指定设备
		self.spatial_branch.to(device)
		self.edge_enhance.to(device)
		
		# 添加通道适配器（如果需要）
		self.channel_adapter = None
	
	def _build_channel_adapter(self, input_channels, target_channels, device):
		"""构建通道适配器"""
		if input_channels != target_channels:
			self.channel_adapter = nn.Conv3d(
				input_channels, target_channels,
				kernel_size=1, bias=False
			).to(device)
	
	def process(self, patches, tiers):
		"""处理patches - 完整实现空间分支"""
		start_time = time.time()
		
		spatial_features = []
		processed_tiers = []
		
		for patch, tier in zip(patches, tiers):
			try:
				# 确保输入是张量格式
				if isinstance(patch, np.ndarray):
					patch_tensor = torch.from_numpy(patch).float().unsqueeze(0).to(self.device)
				else:
					patch_tensor = patch.to(self.device)
				
				# 1. 边缘增强处理
				edge_features = self.edge_enhance(patch_tensor)
				
				# 2. 空间特征提取
				spatial_feat = self.spatial_branch(patch_tensor)
				
				# 3. 特征融合（如果需要）
				if edge_features.shape[1] != spatial_feat.shape[1]:
					# 动态构建通道适配器
					if self.channel_adapter is None:
						self._build_channel_adapter(
							edge_features.shape[1],
							spatial_feat.shape[1],
							self.device
						)
					
					if self.channel_adapter is not None:
						edge_features = self.channel_adapter(edge_features)
					else:
						# 简单的通道调整
						edge_features = F.adaptive_avg_pool3d(edge_features, (1, 1, 1))
						edge_features = F.interpolate(edge_features, size=spatial_feat.shape[2:])
						edge_features = edge_features.expand_as(spatial_feat)
				
				# 组合空间特征和边缘特征
				combined_features = spatial_feat + edge_features
				
				spatial_features.append(combined_features)
				processed_tiers.append(tier)
			
			except Exception as e:
				print(f"Spatial processing failed for tier {tier}: {e}")
				continue
		
		self.compute_time += time.time() - start_time
		self.batch_count += len(patches)
		
		return spatial_features, processed_tiers
	
	def forward(self, patches=None, tiers=None):
		"""同步前向处理"""
		if patches is None and self.node_comm:
			# 从PatchSchedulingStage(GPU 1)接收数据 - 使用新的通信方式
			prev_rank = self.node_comm.rank - 2  # PatchSchedulingStage在GPU 1
			
			patches = []
			tiers = []
			
			try:
				# 🔥 新功能：接收List[Tuple[torch.Tensor, int]]格式
				if hasattr(self.node_comm, 'recv_tensor_tuple_list'):
					patch_data = self.node_comm.recv_tensor_tuple_list(
						src_rank=prev_rank,
						tag=60  # 对应发送时的tag
					)
					
					if patch_data is not None:
						patches = [item[0] for item in patch_data]
						tiers = [item[1] for item in patch_data]
						self.logger.debug(f"✅ 空间分支接收成功: {len(patch_data)}个patches")
						self.update_comm_stats(True)
					else:
						self.logger.error("❌ 空间分支接收失败，尝试回退模式")
						self.update_comm_stats(False)
						patches, tiers = self._fallback_recv_patches(prev_rank)
				
				elif hasattr(self.node_comm, 'recv_data'):
					patch_data = self.node_comm.recv_data(
						src_rank=prev_rank,
						tag=60,
						reliable=True
					)
					
					if patch_data is not None and isinstance(patch_data, list):
						patches = [item[0] for item in patch_data]
						tiers = [item[1] for item in patch_data]
						self.logger.debug(f"✅ 空间分支接收成功(通用模式): {len(patch_data)}个patches")
						self.update_comm_stats(True)
					else:
						self.logger.error("❌ 空间分支接收失败，尝试回退模式")
						self.update_comm_stats(False)
						patches, tiers = self._fallback_recv_patches(prev_rank)
				
				else:
					# 回退到原有接收方式
					self.logger.warning("⚠️  使用原有接收方式(不支持复杂数据类型)")
					patches, tiers = self._fallback_recv_patches(prev_rank)
			
			except Exception as e:
				self.logger.error(f"❌ 空间分支接收异常: {e}")
				self.update_comm_stats(False)
				patches, tiers = self._fallback_recv_patches(prev_rank)
		
		# 处理patches
		spatial_features, processed_tiers = self.process(patches, tiers)
		
		# 将空间特征发送到特征融合阶段(GPU 4, 节点2) - 使用新的通信方式
		if self.node_comm:
			# 特征融合阶段在节点2
			fusion_rank = self.node_comm.node_ranks[1] if hasattr(self.node_comm,
			                                                      'node_ranks') else self.node_comm.rank + 1
			
			try:
				# 🔥 新功能：发送List[Tuple[torch.Tensor, int]]格式
				spatial_data = [(feature, tier) for feature, tier in zip(spatial_features, processed_tiers)]
				
				if hasattr(self.node_comm, 'send_tensor_tuple_list'):
					success = self.node_comm.send_tensor_tuple_list(spatial_data, dst_rank=fusion_rank, tag=110)
					
					if success:
						self.logger.debug(f"✅ 空间特征发送成功: {len(spatial_data)}个特征")
						self.update_comm_stats(True)
					else:
						self.logger.error("❌ 空间特征发送失败，尝试回退模式")
						self.update_comm_stats(False)
						self._fallback_send_spatial_features(spatial_features, processed_tiers, fusion_rank)
				
				elif hasattr(self.node_comm, 'send_data'):
					success = self.node_comm.send_data(spatial_data, dst_rank=fusion_rank, tag=110, reliable=True)
					
					if success:
						self.logger.debug(f"✅ 空间特征发送成功(通用模式): {len(spatial_data)}个特征")
						self.update_comm_stats(True)
					else:
						self.logger.error("❌ 空间特征发送失败，尝试回退模式")
						self.update_comm_stats(False)
						self._fallback_send_spatial_features(spatial_features, processed_tiers, fusion_rank)
				
				else:
					# 回退到原有发送方式
					self.logger.warning("⚠️  使用原有发送方式(不支持复杂数据类型)")
					self._fallback_send_spatial_features(spatial_features, processed_tiers, fusion_rank)
			
			except Exception as e:
				self.logger.error(f"❌ 空间特征发送异常: {e}")
				self.update_comm_stats(False)
				self._fallback_send_spatial_features(spatial_features, processed_tiers, fusion_rank)
		
		return spatial_features, processed_tiers
	
	def _fallback_recv_patches(self, prev_rank):
		"""回退接收方式"""
		patches = []
		tiers = []
		
		try:
			# 接收patches数量
			count_tensor = self.node_comm.recv_tensor(
				src_rank=prev_rank,
				dtype=torch.long,
				device=self.device,
				reliable=False
			)
			
			if count_tensor is None:
				self.logger.error("无法接收patches数量")
				return patches, tiers
			
			count = count_tensor.item()
			
			# 接收每个patch
			for i in range(count):
				# 接收patch图像
				patch_tensor = self.node_comm.recv_tensor(
					src_rank=prev_rank,
					dtype=torch.float32,
					device=self.device,
					reliable=False
				)
				
				# 接收tier信息
				tier_tensor = self.node_comm.recv_tensor(
					src_rank=prev_rank,
					dtype=torch.long,
					device=self.device,
					reliable=False
				)
				
				if patch_tensor is not None and tier_tensor is not None:
					patches.append(patch_tensor)
					tiers.append(tier_tensor.item())
				else:
					self.logger.warning(f"patch或tier接收失败: 第{i}个")
			
			self.logger.debug(f"✅ 空间分支接收成功(回退模式): {len(patches)}个patches")
		
		except Exception as e:
			self.logger.error(f"❌ 回退接收也失败: {e}")
		
		return patches, tiers
	
	def _fallback_send_spatial_features(self, spatial_features, processed_tiers, fusion_rank):
		"""回退发送方式"""
		try:
			# 发送features数量
			count_tensor = torch.tensor([len(spatial_features)], dtype=torch.long, device=self.device)
			self.node_comm.send_tensor(count_tensor, dst_rank=fusion_rank, reliable=False)
			
			# 逐个发送特征
			for feature, tier in zip(spatial_features, processed_tiers):
				self.node_comm.send_tensor(feature, dst_rank=fusion_rank, reliable=False)
				tier_tensor = torch.tensor([tier], dtype=torch.long, device=self.device)
				self.node_comm.send_tensor(tier_tensor, dst_rank=fusion_rank, reliable=False)
			
			self.logger.debug(f"✅ 空间特征发送成功(回退模式): {len(spatial_features)}个特征")
		
		except Exception as e:
			self.logger.error(f"❌ 回退发送也失败: {e}")
	
	def get_state_dict_prefix(self):
		"""获取带前缀的参数字典"""
		# 保存空间分支参数
		state_dict = {}
		for name, param in self.spatial_branch.state_dict().items():
			state_dict[f'spatial_branch.{name}'] = param
		for name, param in self.edge_enhance.state_dict().items():
			state_dict[f'edge_enhance.{name}'] = param
		if self.channel_adapter is not None:
			for name, param in self.channel_adapter.state_dict().items():
				state_dict[f'channel_adapter.{name}'] = param
		return state_dict


# 节点2 (GPU 4): 特征融合
class FeatureFusionStage(BaseStage):
	"""特征融合阶段"""
	
	def __init__(self, model, device, node_comm=None, shared_components=None, config=None):
		super().__init__("FeatureFusionStage", device, node_comm)
		self.config = config or {}
		
		# 提取特征融合组件 - 这些通常来自模型，不是独立初始化的
		self.attention_fusion = model.attention_fusion
		
		# 移动到指定设备
		self.attention_fusion.to(device)
		
		# 特征缓存
		self.ch_features_cache = {}
		self.spatial_features_cache = {}
		self.tiers_cache = {}
		self.fused_features = {}
	
	def process(self, ch_features, spatial_features, tiers):
		"""融合CH特征和空间特征 - 完整实现"""
		start_time = time.time()
		
		fused_features = []
		
		# 确保特征数量匹配
		min_len = min(len(ch_features), len(spatial_features))
		
		for i in range(min_len):
			ch_feat = ch_features[i]
			spatial_feat = spatial_features[i]
			tier = tiers[i] if i < len(tiers) else 0
			
			try:
				# 特征维度对齐
				if ch_feat.shape != spatial_feat.shape:
					# 调整空间维度
					if ch_feat.shape[2:] != spatial_feat.shape[2:]:
						spatial_feat = F.interpolate(
							spatial_feat,
							size=ch_feat.shape[2:],
							mode='trilinear',
							align_corners=False
						)
					
					# 调整通道维度
					if ch_feat.shape[1] != spatial_feat.shape[1]:
						if ch_feat.shape[1] > spatial_feat.shape[1]:
							# 扩展spatial特征通道
							pad_channels = ch_feat.shape[1] - spatial_feat.shape[1]
							spatial_feat = F.pad(spatial_feat, (0, 0, 0, 0, 0, 0, 0, pad_channels))
						else:
							# 裁剪spatial特征通道
							spatial_feat = spatial_feat[:, :ch_feat.shape[1]]
				
				# 应用注意力融合
				fused = self.attention_fusion(ch_feat, spatial_feat)
				
				fused_features.append((fused, tier))
			
			except Exception as e:
				print(f"Feature fusion failed for tier {tier}: {e}")
				continue
		
		self.compute_time += time.time() - start_time
		self.batch_count += len(ch_features)
		
		return fused_features
	
	def forward(self, ch_features=None, spatial_features=None, tiers=None):
		"""同步前向处理"""
		if ch_features is None and self.node_comm:
			# 接收CH特征(从节点1的GPU 2) - 使用新的复杂数据类型接收
			ch_source_rank = self._get_ch_source_rank()
			
			ch_features = []
			ch_tiers = []
			
			try:
				# 🔥 新功能：直接接收List[Tuple[torch.Tensor, int]]格式
				if hasattr(self.node_comm, 'recv_tensor_tuple_list'):
					# 使用专门的方法接收tensor-tuple列表
					ch_data = self.node_comm.recv_tensor_tuple_list(
						src_rank=ch_source_rank,
						tag=100
					)
					
					if ch_data is not None:
						ch_features = [item[0] for item in ch_data]
						ch_tiers = [item[1] for item in ch_data]
						self.logger.debug(f"✅ CH特征接收成功: {len(ch_data)}个特征")
						self.update_comm_stats(True)
					else:
						self.logger.error("❌ CH特征接收失败，尝试回退模式")
						self.update_comm_stats(False)
						ch_features, ch_tiers = self._fallback_recv_ch_features(ch_source_rank)
				
				elif hasattr(self.node_comm, 'recv_data'):
					# 使用通用的复杂数据接收方法
					ch_data = self.node_comm.recv_data(
						src_rank=ch_source_rank,
						tag=100,
						reliable=True
					)
					
					if ch_data is not None and isinstance(ch_data, list):
						ch_features = [item[0] for item in ch_data]
						ch_tiers = [item[1] for item in ch_data]
						self.logger.debug(f"✅ CH特征接收成功(通用模式): {len(ch_data)}个特征")
						self.update_comm_stats(True)
					else:
						self.logger.error("❌ CH特征接收失败，尝试回退模式")
						self.update_comm_stats(False)
						ch_features, ch_tiers = self._fallback_recv_ch_features(ch_source_rank)
				
				else:
					# 回退到原有接收方式
					self.logger.warning("⚠️  使用原有接收方式(不支持复杂数据类型)")
					ch_features, ch_tiers = self._fallback_recv_ch_features(ch_source_rank)
			
			except Exception as e:
				self.logger.error(f"❌ CH特征接收异常: {e}")
				self.update_comm_stats(False)
				# 回退到原有接收方式
				ch_features, ch_tiers = self._fallback_recv_ch_features(ch_source_rank)
			
			# 接收空间特征(从节点1的GPU 3) - 使用新的复杂数据类型接收
			spatial_source_rank = self._get_spatial_source_rank()
			
			spatial_features = []
			spatial_tiers = []
			
			try:
				# 🔥 新功能：直接接收List[Tuple[torch.Tensor, int]]格式
				if hasattr(self.node_comm, 'recv_tensor_tuple_list'):
					# 使用专门的方法接收tensor-tuple列表
					spatial_data = self.node_comm.recv_tensor_tuple_list(
						src_rank=spatial_source_rank,
						tag=110  # 对应SpatialFusionStage发送的tag
					)
					
					if spatial_data is not None:
						spatial_features = [item[0] for item in spatial_data]
						spatial_tiers = [item[1] for item in spatial_data]
						self.logger.debug(f"✅ 空间特征接收成功: {len(spatial_data)}个特征")
						self.update_comm_stats(True)
					else:
						self.logger.error("❌ 空间特征接收失败，尝试回退模式")
						self.update_comm_stats(False)
						spatial_features, spatial_tiers = self._fallback_recv_spatial_features(spatial_source_rank)
				
				elif hasattr(self.node_comm, 'recv_data'):
					# 使用通用的复杂数据接收方法
					spatial_data = self.node_comm.recv_data(
						src_rank=spatial_source_rank,
						tag=110,
						reliable=True
					)
					
					if spatial_data is not None and isinstance(spatial_data, list):
						spatial_features = [item[0] for item in spatial_data]
						spatial_tiers = [item[1] for item in spatial_data]
						self.logger.debug(f"✅ 空间特征接收成功(通用模式): {len(spatial_data)}个特征")
						self.update_comm_stats(True)
					else:
						self.logger.error("❌ 空间特征接收失败，尝试回退模式")
						self.update_comm_stats(False)
						spatial_features, spatial_tiers = self._fallback_recv_spatial_features(spatial_source_rank)
				
				else:
					# 回退到原有接收方式
					self.logger.warning("⚠️  使用原有接收方式(不支持复杂数据类型)")
					spatial_features, spatial_tiers = self._fallback_recv_spatial_features(spatial_source_rank)
			
			except Exception as e:
				self.logger.error(f"❌ 空间特征接收异常: {e}")
				self.update_comm_stats(False)
				# 回退到原有接收方式
				spatial_features, spatial_tiers = self._fallback_recv_spatial_features(spatial_source_rank)
			
			# 确保特征和tier匹配
			ch_count = len(ch_features)
			spatial_count = len(spatial_features)
			if ch_count != spatial_count:
				print(f"Warning: CH特征({ch_count})和空间特征({spatial_count})数量不匹配")
			
			if ch_tiers != spatial_tiers:
				print(f"Warning: CH特征和空间特征tier不匹配")
			
			tiers = ch_tiers
		
		# 处理特征
		fused_features = self.process(ch_features, spatial_features, tiers)
		
		# 将融合特征发送到多尺度融合阶段(GPU 5)
		if self.node_comm:
			next_rank = self.node_comm.rank + 1
			
			# 发送features数量
			count_tensor = torch.tensor([len(fused_features)], dtype=torch.long, device=self.device)
			self.node_comm.send_tensor(count_tensor, dst_rank=next_rank)
			
			# 发送每个融合特征
			for fused, tier in fused_features:
				# 发送融合特征
				self.node_comm.send_tensor(fused, dst_rank=next_rank)
				
				# 发送tier信息
				tier_tensor = torch.tensor([tier], dtype=torch.long, device=self.device)
				self.node_comm.send_tensor(tier_tensor, dst_rank=next_rank)
		
		return fused_features
	
	def _fallback_recv_ch_features(self, ch_source_rank):
		"""回退接收方式 - 兼容原有通信方式"""
		ch_features = []
		ch_tiers = []
		
		try:
			# 接收features数量
			count_tensor = self.node_comm.recv_tensor(
				src_rank=ch_source_rank,
				dtype=torch.long,
				device=self.device,
				reliable=False
			)
			
			if count_tensor is None:
				self.logger.error("无法接收CH特征数量")
				return ch_features, ch_tiers
			
			ch_count = count_tensor.item()
			
			# 接收每个CH特征
			for i in range(ch_count):
				# 接收CH特征
				ch_feat = self.node_comm.recv_tensor(
					src_rank=ch_source_rank,
					device=self.device,
					reliable=False
				)
				
				# 接收tier信息
				tier_tensor = self.node_comm.recv_tensor(
					src_rank=ch_source_rank,
					dtype=torch.long,
					device=self.device,
					reliable=False
				)
				
				if ch_feat is not None and tier_tensor is not None:
					ch_features.append(ch_feat)
					ch_tiers.append(tier_tensor.item())
				else:
					self.logger.warning(f"CH特征或tier接收失败: 第{i}个")
			
			self.logger.debug(f"✅ CH特征接收成功(回退模式): {len(ch_features)}个特征")
		
		except Exception as e:
			self.logger.error(f"❌ 回退接收也失败: {e}")
		
		return ch_features, ch_tiers
	
	def _fallback_recv_spatial_features(self, spatial_source_rank):
		"""回退接收空间特征 - 兼容原有通信方式"""
		spatial_features = []
		spatial_tiers = []
		
		try:
			# 接收features数量
			count_tensor = self.node_comm.recv_tensor(
				src_rank=spatial_source_rank,
				dtype=torch.long,
				device=self.device,
				reliable=False
			)
			
			if count_tensor is None:
				self.logger.error("无法接收空间特征数量")
				return spatial_features, spatial_tiers
			
			spatial_count = count_tensor.item()
			
			# 接收每个空间特征
			for i in range(spatial_count):
				# 接收空间特征
				spatial_feat = self.node_comm.recv_tensor(
					src_rank=spatial_source_rank,
					device=self.device,
					reliable=False
				)
				
				# 接收tier信息
				tier_tensor = self.node_comm.recv_tensor(
					src_rank=spatial_source_rank,
					dtype=torch.long,
					device=self.device,
					reliable=False
				)
				
				if spatial_feat is not None and tier_tensor is not None:
					spatial_features.append(spatial_feat)
					spatial_tiers.append(tier_tensor.item())
				else:
					self.logger.warning(f"空间特征或tier接收失败: 第{i}个")
			
			self.logger.debug(f"✅ 空间特征接收成功(回退模式): {len(spatial_features)}个特征")
		
		except Exception as e:
			self.logger.error(f"❌ 空间特征回退接收也失败: {e}")
		
		return spatial_features, spatial_tiers
	
	def _get_ch_source_rank(self):
		"""获取CH分支的源rank"""
		# CH分支通常在当前rank的前一个节点
		if hasattr(self.node_comm, 'node_ranks') and len(self.node_comm.node_ranks) > 0:
			# 计算CH分支rank (节点1的GPU 2)
			return self.node_comm.node_ranks[0] + 2  # 节点1的第3个GPU (index 2)
		else:
			# 回退计算
			return max(0, self.node_comm.rank - 2)
	
	def _get_spatial_source_rank(self):
		"""获取空间分支的源rank"""
		# 空间分支通常在当前rank的前一个节点
		if hasattr(self.node_comm, 'node_ranks') and len(self.node_comm.node_ranks) > 0:
			# 计算空间分支rank (节点1的GPU 3)
			return self.node_comm.node_ranks[0] + 3  # 节点1的第4个GPU (index 3)
		else:
			# 回退计算
			return max(0, self.node_comm.rank - 1)
	
	def get_state_dict_prefix(self):
		"""获取带前缀的参数字典"""
		# 保存注意力融合参数
		state_dict = {}
		for name, param in self.attention_fusion.state_dict().items():
			state_dict[f'attention_fusion.{name}'] = param
		return state_dict


# 节点2 (GPU 5): 多尺度融合
class MultiscaleFusionStage(BaseStage):
	"""多尺度融合阶段"""
	
	def __init__(self, model, device, node_comm=None, shared_components=None, config=None):
		super().__init__("MultiscaleFusionStage", device, node_comm)
		self.config = config or {}
		
		# 提取多尺度融合组件 - 这些通常来自模型，不是独立初始化的
		self.multiscale_fusion = model.multiscale_fusion
		
		# 移动到指定设备
		self.multiscale_fusion.to(device)
		
		# 特征缓存
		self.tier_features = {}
	
	def process(self, fused_features):
		"""执行多尺度融合"""
		start_time = time.time()
		
		try:
			# 更新tier特征字典
			self.tier_features.clear()
			for fused, tier in fused_features:
				self.tier_features[tier] = fused
			
			# 如果只有一个tier，直接返回
			if len(self.tier_features) == 1:
				tier = list(self.tier_features.keys())[0]
				result = self.tier_features[tier]
			elif len(self.tier_features) > 1:
				# 执行多尺度融合
				result = self.multiscale_fusion(self.tier_features)
			else:
				# 没有特征，返回None
				return None
			
			self.compute_time += time.time() - start_time
			self.batch_count += 1
			
			return result
		
		except Exception as e:
			print(f"Multiscale fusion failed: {e}")
			return None
	
	def forward(self, fused_features=None):
		"""同步前向处理"""
		if fused_features is None and self.node_comm:
			# 从特征融合阶段(GPU 4)接收数据
			prev_rank = self.node_comm.rank - 1
			
			# 接收features数量
			count_tensor = self.node_comm.recv_tensor(
				src_rank=prev_rank,
				dtype=torch.long,
				device=self.device
			)
			count = count_tensor.item()
			
			# 接收每个融合特征
			fused_features = []
			for i in range(count):
				# 接收融合特征
				fused = self.node_comm.recv_tensor(
					src_rank=prev_rank,
					device=self.device
				)
				
				# 接收tier信息
				tier_tensor = self.node_comm.recv_tensor(
					src_rank=prev_rank,
					dtype=torch.long,
					device=self.device
				)
				
				fused_features.append((fused, tier_tensor.item()))
		
		# 处理特征
		multiscale_result = self.process(fused_features)
		
		# 将多尺度融合结果发送到分割头阶段(GPU 6)
		if self.node_comm and multiscale_result is not None:
			next_rank = self.node_comm.rank + 1
			
			# 发送多尺度融合结果
			self.node_comm.send_tensor(multiscale_result, dst_rank=next_rank)
		
		return multiscale_result
	
	def get_state_dict_prefix(self):
		"""获取带前缀的参数字典"""
		# 保存多尺度融合参数
		state_dict = {}
		for name, param in self.multiscale_fusion.state_dict().items():
			state_dict[f'multiscale_fusion.{name}'] = param
		return state_dict


# 节点2 (GPU 6): 分割头和损失计算
class BackendStage(BaseStage):
	"""后端处理阶段 - 处理分割头和损失计算"""
	
	def __init__(self, model, device, node_comm=None, shared_components=None, config=None):
		super().__init__("BackendStage", device, node_comm)
		self.config = config or {}
		
		# 提取分割头组件
		self.seg_head_first = model.seg_head_first  # 可能是None
		self.seg_head_tail = model.seg_head_tail
		
		# 创建损失函数 - 优先使用共享组件
		self.criterion = None
		if shared_components and 'criterion' in shared_components:
			self.criterion = shared_components['criterion']
		
		else:
			from loss import CombinedLoss
			
			self.criterion = CombinedLoss()
		
		# 移动到指定设备
		if self.seg_head_first is not None:
			self.seg_head_first.to(device)
		self.seg_head_tail.to(device)
		self.criterion.to(device)
		
		# 当前tier
		self.current_tier = None
	
	def set_tier(self, tier):
		"""设置当前tier"""
		self.current_tier = tier
	
	def _build_seg_head(self, in_c, ref):
		"""构建分割头"""
		self.seg_head_first = nn.Conv3d(in_c, 32, 3, padding=1, bias=False)
		self.seg_head_first.to(ref.device, dtype=ref.dtype)
	
	def process(self, multiscale_result, labels=None):
		"""执行分割头处理和损失计算"""
		start_time = time.time()
		
		# 分割头处理
		if self.seg_head_first is None:
			# 延迟构建分割头
			self._build_seg_head(multiscale_result.shape[1], multiscale_result)
		
		# 确保分割头在正确设备上
		if next(self.seg_head_first.parameters()).device != multiscale_result.device:
			self.seg_head_first = self.seg_head_first.to(multiscale_result.device)
			self.seg_head_tail = self.seg_head_tail.to(multiscale_result.device)
		
		# 执行分割
		logits = self.seg_head_tail(self.seg_head_first(multiscale_result))
		
		# 计算损失(如果有标签)
		loss = None
		if labels is not None and self.training:
			loss = self.criterion(logits, labels)
		
		self.compute_time += time.time() - start_time
		self.batch_count += 1
		
		return logits, loss
	
	def forward(self, multiscale_result=None, labels=None):
		"""同步前向处理"""
		if multiscale_result is None and self.node_comm:
			# 从多尺度融合阶段(GPU 5)接收数据
			prev_rank = self.node_comm.rank - 1
			
			# 接收多尺度融合结果
			multiscale_result = self.node_comm.recv_tensor(
				src_rank=prev_rank,
				device=self.device
			)
		
		# 处理特征
		logits, loss = self.process(multiscale_result, labels)
		
		return logits, loss
	
	def get_state_dict_prefix(self):
		"""获取带前缀的参数字典"""
		# 保存分割头参数
		state_dict = {}
		if self.seg_head_first is not None:
			for name, param in self.seg_head_first.state_dict().items():
				state_dict[f'seg_head_first.{name}'] = param
		for name, param in self.seg_head_tail.state_dict().items():
			state_dict[f'seg_head_tail.{name}'] = param
		return state_dict


# 在 scripts/distributed/stages.py 文件末尾添加以下函数

def create_pipeline_stages(config, node_comm=None):
	"""
	创建流水线阶段的工厂函数

	参数:
		config: 配置字典
		node_comm: 节点通信管理器

	返回:
		阶段字典
	"""
	import torch
	from models import create_vessel_segmenter
	
	# 获取当前rank和设备
	rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
	device = torch.cuda.current_device()
	
	# 创建完整模型（用于提取组件）
	full_model = create_vessel_segmenter(config)
	
	stages = {}
	
	# 根据rank创建相应的阶段 - 使用正确的类名
	if rank == 0:  # 节点1, GPU 0 - 预处理
		stages['preprocessing'] = FrontendStage(  # 使用现有的FrontendStage
			full_model, device, node_comm, config=config
		)
	elif rank == 1:  # 节点1, GPU 1 - 采样调度
		stages['patch_scheduling'] = PatchSchedulingStage(
			full_model, device, node_comm, config=config
		)
	elif rank == 2:  # 节点1, GPU 2 - CH分支
		stages['ch_branch'] = CHProcessingStage(
			full_model, device, node_comm, config=config
		)
	elif rank == 3:  # 节点1, GPU 3 - 空间分支
		stages['spatial_branch'] = SpatialFusionStage(
			full_model, device, node_comm, config=config
		)
	elif rank == 4:  # 节点2, GPU 4 - 特征融合
		stages['feature_fusion'] = FeatureFusionStage(
			full_model, device, node_comm, config=config
		)
	elif rank == 5:  # 节点2, GPU 5 - 多尺度融合
		stages['multiscale_fusion'] = MultiscaleFusionStage(
			full_model, device, node_comm, config=config
		)
	elif rank == 6:  # 节点2, GPU 6 - 分割头
		stages['segmentation_head'] = BackendStage(  # 使用现有的BackendStage
			full_model, device, node_comm, config=config
		)
	
	stages[7] = lambda model, device: DummyStage(model, device, node_comm, config)
	
	return stages


class SegmentationHeadStage(BaseStage):
	"""分割头阶段"""
	
	def __init__(self, model, device, node_comm=None, shared_components=None, config=None):
		super().__init__("SegmentationHeadStage", device, node_comm)
		self.config = config or {}
		
		# 提取分割头组件
		self.seg_head_first = model.seg_head_first
		self.seg_head_tail = model.seg_head_tail
		self.final_activation = model.final_activation
		
		# 移动到指定设备
		if self.seg_head_first is not None:
			self.seg_head_first.to(device)
		self.seg_head_tail.to(device)
		self.final_activation.to(device)
	
	def process(self, multiscale_features):
		"""处理多尺度特征，输出最终分割结果"""
		start_time = time.time()
		
		try:
			# 如果seg_head_first尚未构建，则延迟构建
			if self.seg_head_first is None:
				in_channels = multiscale_features.shape[1]
				self.seg_head_first = torch.nn.Conv3d(
					in_channels, 32, 3, padding=1, bias=False
				).to(self.device)
			
			# 分割头处理
			x = self.seg_head_first(multiscale_features)
			x = self.seg_head_tail(x)
			output = self.final_activation(x)
			
			self.compute_time += time.time() - start_time
			self.batch_count += 1
			
			return output
		
		except Exception as e:
			print(f"Segmentation head processing failed: {e}")
			return None
	
	def forward(self, multiscale_features=None):
		"""同步前向处理"""
		if multiscale_features is None and self.node_comm:
			# 从多尺度融合阶段接收数据
			prev_rank = self.node_comm.rank - 1
			
			# 接收多尺度特征
			multiscale_features = self.node_comm.recv_tensor(
				src_rank=prev_rank,
				device=self.device
			)
		
		# 处理特征
		output = self.process(multiscale_features)
		
		return output
	
	def get_state_dict_prefix(self):
		"""获取带前缀的参数字典"""
		state_dict = {}
		
		if self.seg_head_first is not None:
			for name, param in self.seg_head_first.state_dict().items():
				state_dict[f'seg_head_first.{name}'] = param
		
		for name, param in self.seg_head_tail.state_dict().items():
			state_dict[f'seg_head_tail.{name}'] = param
		
		for name, param in self.final_activation.state_dict().items():
			state_dict[f'final_activation.{name}'] = param
		
		return state_dict