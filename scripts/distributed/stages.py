import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import threading
from queue import Queue
from typing import Dict, List, Optional, Tuple, Any


class BaseStage(nn.Module):
	"""处理阶段基类，提供共享功能"""
	
	def __init__(self, name: str, device: str, node_comm=None):
		"""
		初始化处理阶段

		参数:
			name: 阶段名称
			device: 设备 (例如 'cuda:0')
			node_comm: 节点通信管理器
		"""
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
		
		print(f"[{self.name}] 初始化在 {self.device}")
	
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
				print(f"Error in {self.name} worker: {e}")
				time.sleep(1.0)  # 避免错误循环消耗资源
	
	def process(self, *args, **kwargs):
		"""处理逻辑(由子类实现)"""
		raise NotImplementedError("子类必须实现process方法")
	
	def forward(self, *args, **kwargs):
		"""同步前向处理(由子类实现)"""
		raise NotImplementedError("子类必须实现forward方法")


# 节点1 (GPU 0): 数据预处理+ROI提取
class PreprocessingStage(BaseStage):
	"""数据预处理和ROI提取阶段"""
	
	def __init__(self, model, device, node_comm=None):
		super().__init__("PreprocessingStage", device, node_comm)
		
		# 从模型提取预处理组件
		self.preprocessor = model.preprocessor if hasattr(model, 'preprocessor') else None
		if self.preprocessor is None:
			from data.processing import CTPreprocessor
			self.preprocessor = CTPreprocessor()
		
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


# 节点1 (GPU 1): 三级采样+Patch调度
class PatchSchedulingStage(BaseStage):
	"""Patch采样和调度阶段"""
	
	def __init__(self, model, device, node_comm=None):
		super().__init__("PatchSchedulingStage", device, node_comm)
		
		# 从模型提取采样器
		self.tier_sampler = model.tier_sampler if hasattr(model, 'tier_sampler') else None
		if self.tier_sampler is None:
			from data.tier_sampling import TierSampler
			self.tier_sampler = TierSampler()
		
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
		
		# 将patches发送到CH分支(GPU 2)和空间分支(GPU 3)
		if self.node_comm:
			# 发送到CH分支 (GPU 2)
			ch_rank = self.node_comm.rank + 1
			
			# 发送patches数量
			count_tensor = torch.tensor([len(patches)], dtype=torch.long, device=self.device)
			self.node_comm.send_tensor(count_tensor, dst_rank=ch_rank)
			
			# 发送每个patch
			for patch in patches:
				# 转换patch为张量
				patch_tensor = torch.from_numpy(patch['image']).float().unsqueeze(0).to(self.device)
				tier_tensor = torch.tensor([patch['tier']], dtype=torch.long, device=self.device)
				
				# 发送图像和tier信息
				self.node_comm.send_tensor(patch_tensor, dst_rank=ch_rank)
				self.node_comm.send_tensor(tier_tensor, dst_rank=ch_rank)
			
			# 同时发送到空间分支 (GPU 3)
			spatial_rank = ch_rank + 1
			
			# 发送patches数量
			self.node_comm.send_tensor(count_tensor, dst_rank=spatial_rank)
			
			# 发送每个patch
			for patch in patches:
				# 转换patch为张量
				patch_tensor = torch.from_numpy(patch['image']).float().unsqueeze(0).to(self.device)
				tier_tensor = torch.tensor([patch['tier']], dtype=torch.long, device=self.device)
				
				# 发送图像和tier信息
				self.node_comm.send_tensor(patch_tensor, dst_rank=spatial_rank)
				self.node_comm.send_tensor(tier_tensor, dst_rank=spatial_rank)
		
		return patches, case_patches


# 节点1 (GPU 2): CH分支完整处理
class CHBranchStage(BaseStage):
	"""CH分支处理阶段"""
	
	def __init__(self, model, device, node_comm=None):
		super().__init__("CHBranchStage", device, node_comm)
		
		# 提取CH分支组件
		self.ch_branch = model.ch_branch
		self.tier_params = model.tier_params
		
		# 移动到指定设备
		self.ch_branch.to(device)
	
	def process(self, patches, tiers):
		"""处理patches"""
		start_time = time.time()
		
		ch_features = []
		
		# 对每个patch执行CH分支处理
		for i, (patch, tier) in enumerate(zip(patches, tiers)):
			# 设置当前tier
			self.ch_branch.set_tier(int(tier))
			
			# 获取tier特定的r_scale
			r_scale = 1.0
			if tier in self.tier_params:
				r_scale = self.tier_params[tier].get('r_scale', 1.0)
			
			# 执行CH分支处理
			ch_feature = self.ch_branch(patch, r_scale=r_scale)
			ch_features.append(ch_feature)
		
		self.compute_time += time.time() - start_time
		self.batch_count += len(patches)
		
		return ch_features, tiers
	
	def forward(self, patches=None, tiers=None):
		"""同步前向处理"""
		if patches is None and self.node_comm:
			# 从PatchSchedulingStage(GPU 1)接收数据
			prev_rank = self.node_comm.rank - 1
			
			# 接收patches数量
			count_tensor = self.node_comm.recv_tensor(
				src_rank=prev_rank,
				dtype=torch.long,
				device=self.device
			)
			count = count_tensor.item()
			
			# 接收每个patch
			patches = []
			tiers = []
			for i in range(count):
				# 接收patch图像
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
		
		# 处理patches
		ch_features, tiers = self.process(patches, tiers)
		
		# 将CH特征发送到特征融合阶段(GPU 4, 节点2)
		if self.node_comm:
			# 特征融合阶段在节点2
			fusion_rank = self.node_comm.node_ranks[1] if self.node_comm.node_rank == 0 else self.node_comm.rank + 1
			
			# 发送features数量
			count_tensor = torch.tensor([len(ch_features)], dtype=torch.long, device=self.device)
			self.node_comm.send_tensor(count_tensor, dst_rank=fusion_rank)
			
			# 发送每个特征
			for i, (feature, tier) in enumerate(zip(ch_features, tiers)):
				# 发送CH特征
				self.node_comm.send_tensor(feature, dst_rank=fusion_rank)
				
				# 发送tier信息
				tier_tensor = torch.tensor([tier], dtype=torch.long, device=self.device)
				self.node_comm.send_tensor(tier_tensor, dst_rank=fusion_rank)
		
		return ch_features, tiers


# 节点1 (GPU 3): 空间分支完整处理
class SpatialBranchStage(BaseStage):
	"""空间分支处理阶段"""
	
	def __init__(self, model, device, node_comm=None):
		super().__init__("SpatialBranchStage", device, node_comm)
		
		# 提取空间分支组件
		self.spatial_branch = model.spatial_branch
		self.edge_enhance = model.edge_enhance
		
		# 移动到指定设备
		self.spatial_branch.to(device)
		self.edge_enhance.to(device)
	
	def process(self, patches, tiers):
		"""处理patches"""
		start_time = time.time()
		
		spatial_features = []
		
		# 对每个patch执行空间分支处理
		for patch in patches:
			# 边缘增强
			edge_feat = self.edge_enhance(patch)
			
			# 空间特征提取
			spatial_feat = self.spatial_branch(patch)
			
			spatial_features.append(spatial_feat)
		
		self.compute_time += time.time() - start_time
		self.batch_count += len(patches)
		
		return spatial_features, tiers
	
	def forward(self, patches=None, tiers=None):
		"""同步前向处理"""
		if patches is None and self.node_comm:
			# 从PatchSchedulingStage(GPU 1)接收数据
			prev_rank = self.node_comm.rank - 2  # PatchSchedulingStage在GPU 1
			
			# 接收patches数量
			count_tensor = self.node_comm.recv_tensor(
				src_rank=prev_rank,
				dtype=torch.long,
				device=self.device
			)
			count = count_tensor.item()
			
			# 接收每个patch
			patches = []
			tiers = []
			for i in range(count):
				# 接收patch图像
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
		
		# 处理patches
		spatial_features, tiers = self.process(patches, tiers)
		
		# 将空间特征发送到特征融合阶段(GPU 4, 节点2)
		if self.node_comm:
			# 特征融合阶段在节点2
			fusion_rank = self.node_comm.node_ranks[1] if self.node_comm.node_rank == 0 else self.node_comm.rank + 1
			
			# 发送features数量
			count_tensor = torch.tensor([len(spatial_features)], dtype=torch.long, device=self.device)
			self.node_comm.send_tensor(count_tensor, dst_rank=fusion_rank)
			
			# 发送每个特征
			for i, (feature, tier) in enumerate(zip(spatial_features, tiers)):
				# 发送空间特征
				self.node_comm.send_tensor(feature, dst_rank=fusion_rank)
				
				# 发送tier信息
				tier_tensor = torch.tensor([tier], dtype=torch.long, device=self.device)
				self.node_comm.send_tensor(tier_tensor, dst_rank=fusion_rank)
		
		return spatial_features, tiers


# 节点2 (GPU 4): 特征融合
class FeatureFusionStage(BaseStage):
	"""特征融合阶段"""
	
	def __init__(self, model, device, node_comm=None):
		super().__init__("FeatureFusionStage", device, node_comm)
		
		# 提取特征融合组件
		self.attention_fusion = model.attention_fusion
		
		# 移动到指定设备
		self.attention_fusion.to(device)
		
		# 特征缓存
		self.ch_features_cache = {}
		self.spatial_features_cache = {}
		self.tiers_cache = {}
		self.fused_features = {}
	
	def process(self, ch_features, spatial_features, tiers):
		"""融合CH特征和空间特征"""
		start_time = time.time()
		
		fused_features = []
		
		# 确保特征形状匹配
		assert len(ch_features) == len(spatial_features), "特征数量不匹配"
		
		# 融合每对特征
		for i, (ch_feat, spatial_feat, tier) in enumerate(zip(ch_features, spatial_features, tiers)):
			# 应用注意力融合
			fused = self.attention_fusion(ch_feat, spatial_feat)
			
			# 缓存融合结果
			key = f"tier{tier}_idx{i}"
			self.fused_features[key] = fused
			
			fused_features.append((fused, tier))
		
		self.compute_time += time.time() - start_time
		self.batch_count += len(ch_features)
		
		return fused_features
	
	def forward(self, ch_features=None, spatial_features=None, tiers=None):
		"""同步前向处理"""
		if ch_features is None and self.node_comm:
			# 接收CH特征(从节点1的GPU 2)
			ch_source_rank = self.node_comm.node_ranks[0] + 2  # 节点1的GPU 2
			
			# 接收features数量
			count_tensor = self.node_comm.recv_tensor(
				src_rank=ch_source_rank,
				dtype=torch.long,
				device=self.device
			)
			ch_count = count_tensor.item()
			
			# 接收每个CH特征
			ch_features = []
			ch_tiers = []
			for i in range(ch_count):
				# 接收CH特征
				ch_feat = self.node_comm.recv_tensor(
					src_rank=ch_source_rank,
					device=self.device
				)
				
				# 接收tier信息
				tier_tensor = self.node_comm.recv_tensor(
					src_rank=ch_source_rank,
					dtype=torch.long,
					device=self.device
				)
				
				ch_features.append(ch_feat)
				ch_tiers.append(tier_tensor.item())
			
			# 接收空间特征(从节点1的GPU 3)
			spatial_source_rank = self.node_comm.node_ranks[0] + 3  # 节点1的GPU 3
			
			# 接收features数量
			count_tensor = self.node_comm.recv_tensor(
				src_rank=spatial_source_rank,
				dtype=torch.long,
				device=self.device
			)
			spatial_count = count_tensor.item()
			
			# 接收每个空间特征
			spatial_features = []
			spatial_tiers = []
			for i in range(spatial_count):
				# 接收空间特征
				spatial_feat = self.node_comm.recv_tensor(
					src_rank=spatial_source_rank,
					device=self.device
				)
				
				# 接收tier信息
				tier_tensor = self.node_comm.recv_tensor(
					src_rank=spatial_source_rank,
					dtype=torch.long,
					device=self.device
				)
				
				spatial_features.append(spatial_feat)
				spatial_tiers.append(tier_tensor.item())
			
			# 确保特征和tier匹配
			assert ch_count == spatial_count, "CH特征和空间特征数量不匹配"
			assert ch_tiers == spatial_tiers, "CH特征和空间特征tier不匹配"
			
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


# 节点2 (GPU 5): 多尺度融合
class   MultiscaleFusionStage(BaseStage):
	"""多尺度融合阶段"""
	
	def __init__(self, model, device, node_comm=None):
		super().__init__("MultiscaleFusionStage", device, node_comm)
		
		# 提取多尺度融合组件
		self.multiscale_fusion = model.multiscale_fusion
		
		# 移动到指定设备
		self.multiscale_fusion.to(device)
		
		# 特征缓存
		self.tier_features = {}
	
	def process(self, fused_features):
		"""执行多尺度融合"""
		start_time = time.time()
		
		# 更新tier特征字典
		for fused, tier in fused_features:
			self.tier_features[tier] = fused
		
		# 如果只有一个tier，直接返回
		if len(self.tier_features) == 1:
			tier = list(self.tier_features.keys())[0]
			result = self.tier_features[tier]
		else:
			# 执行多尺度融合
			result = self.multiscale_fusion(self.tier_features)
		
		self.compute_time += time.time() - start_time
		self.batch_count += 1
		
		return result
	
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
		if self.node_comm:
			next_rank = self.node_comm.rank + 1
			
			# 发送多尺度融合结果
			self.node_comm.send_tensor(multiscale_result, dst_rank=next_rank)
		
		return multiscale_result


# 节点2 (GPU 6): 分割头和损失计算
class SegmentationHeadStage(BaseStage):
	"""分割头和损失计算阶段"""
	
	def __init__(self, model, device, node_comm=None):
		super().__init__("SegmentationHeadStage", device, node_comm)
		
		# 提取分割头组件
		self.seg_head_first = model.seg_head_first
		self.seg_head_tail = model.seg_head_tail
		
		# 创建损失函数
		from losses import VesselSegmentationLoss
		self.criterion = VesselSegmentationLoss(
			num_classes=1,
			vessel_weight=10.0,
			tumor_weight=15.0,
			use_boundary=True
		)
		
		# 移动到指定设备
		self.seg_head_first.to(device)
		self.seg_head_tail.to(device)
		self.criterion.to(device)
	
	def process(self, multiscale_result, labels=None):
		"""执行分割头处理和损失计算"""
		start_time = time.time()
		
		# 分割头处理
		if self.seg_head_first is None:
			# 延迟构建分割头
			self.seg_head_first = nn.Conv3d(
				multiscale_result.shape[1], 32, 3, padding=1, bias=False
			).to(self.device)
		
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
	
	

# 添加到stages.py文件末尾

# 兼容层 - 为分布式训练提供统一的接口名称
class FrontendStage(PreprocessingStage):
    """前端处理阶段 - 处理数据预处理和ROI提取"""
    pass

class CHProcessingStage(CHBranchStage):
    """CH处理阶段 - 处理CH分支"""
    pass

class SpatialFusionStage(SpatialBranchStage):
    """空间融合阶段 - 处理空间分支"""
    pass

class BackendStage(SegmentationHeadStage):
    """后端处理阶段 - 处理分割头和损失计算"""
    pass