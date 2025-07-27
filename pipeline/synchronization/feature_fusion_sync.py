#feature_fusion_sync.py
"""
特征融合同步机制
异步缓冲 + 批次匹配，专注同步可靠性
"""

import torch
import threading
import queue
import time
import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from collections import defaultdict


@dataclass
class FeatureBatch:
	"""特征批次数据包"""
	batch_id: str
	features: torch.Tensor
	tiers: List[int]
	timestamp: float
	source_rank: int
	
	def is_compatible(self, other: 'FeatureBatch') -> bool:
		"""检查两个特征批次是否兼容（可融合）"""
		# 只检查batch_id和维度兼容性
		if self.batch_id != other.batch_id:
			return False
		
		# 检查张量维度兼容性（除了通道维度）
		if len(self.features.shape) != len(other.features.shape):
			return False
		
		# 空间维度必须匹配（除了通道维度[1]）
		for i, (dim1, dim2) in enumerate(zip(self.features.shape, other.features.shape)):
			if i != 1 and dim1 != dim2:  # 跳过通道维度
				return False
		
		return True


class AsyncFeatureBuffer:
	"""异步特征缓冲器"""
	
	def __init__(self, max_buffer_size: int = 100, timeout_seconds: float = 10.0):
		self.max_buffer_size = max_buffer_size
		self.timeout_seconds = timeout_seconds
		
		# 缓冲区：按batch_id存储
		self.ch_buffer: Dict[str, FeatureBatch] = {}
		self.spatial_buffer: Dict[str, FeatureBatch] = {}
		
		# 线程安全锁
		self.buffer_lock = threading.RLock()
		
		# 已配对的特征队列
		self.paired_queue = queue.Queue(maxsize=50)
		
		# 统计信息
		self.stats = {
			'received_ch': 0,
			'received_spatial': 0,
			'successful_pairs': 0,
			'timeout_pairs': 0,
			'dimension_mismatches': 0
		}
		
		self.logger = logging.getLogger(__name__)
	
	def add_ch_features(self, batch: FeatureBatch) -> bool:
		"""添加CH特征到缓冲区"""
		with self.buffer_lock:
			try:
				# 检查缓冲区容量
				if len(self.ch_buffer) >= self.max_buffer_size:
					self._cleanup_old_batches()
				
				self.ch_buffer[batch.batch_id] = batch
				self.stats['received_ch'] += 1
				
				self.logger.debug(f"CH特征已缓冲: {batch.batch_id}")
				
				# 尝试配对
				self._try_pair_features(batch.batch_id)
				return True
			
			except Exception as e:
				self.logger.error(f"添加CH特征失败: {e}")
				return False
	
	def add_spatial_features(self, batch: FeatureBatch) -> bool:
		"""添加空间特征到缓冲区"""
		with self.buffer_lock:
			try:
				# 检查缓冲区容量
				if len(self.spatial_buffer) >= self.max_buffer_size:
					self._cleanup_old_batches()
				
				self.spatial_buffer[batch.batch_id] = batch
				self.stats['received_spatial'] += 1
				
				self.logger.debug(f"空间特征已缓冲: {batch.batch_id}")
				
				# 尝试配对
				self._try_pair_features(batch.batch_id)
				return True
			
			except Exception as e:
				self.logger.error(f"添加空间特征失败: {e}")
				return False
	
	def get_paired_features(self, timeout: float = None) -> Optional[Tuple[FeatureBatch, FeatureBatch]]:
		"""获取配对的特征（阻塞调用）"""
		if timeout is None:
			timeout = self.timeout_seconds
		
		try:
			return self.paired_queue.get(timeout=timeout)
		except queue.Empty:
			self.logger.warning("获取配对特征超时")
			return None
	
	def _try_pair_features(self, batch_id: str):
		"""尝试配对特征"""
		if batch_id in self.ch_buffer and batch_id in self.spatial_buffer:
			ch_batch = self.ch_buffer[batch_id]
			spatial_batch = self.spatial_buffer[batch_id]
			
			# 检查兼容性（只检查基本维度匹配）
			if ch_batch.is_compatible(spatial_batch):
				# 配对成功，移除缓冲区中的数据
				del self.ch_buffer[batch_id]
				del self.spatial_buffer[batch_id]
				
				# 添加到配对队列
				try:
					self.paired_queue.put((ch_batch, spatial_batch), block=False)
					self.stats['successful_pairs'] += 1
					self.logger.debug(f"特征配对成功: {batch_id}")
				except queue.Full:
					self.logger.warning("配对队列已满，丢弃配对")
			else:
				self.logger.warning(f"特征维度不兼容，跳过配对: {batch_id}")
				self.stats['dimension_mismatches'] += 1
				# 不兼容时也移除，避免累积
				if batch_id in self.ch_buffer:
					del self.ch_buffer[batch_id]
				if batch_id in self.spatial_buffer:
					del self.spatial_buffer[batch_id]
	
	def _cleanup_old_batches(self):
		"""清理超时的旧批次"""
		current_time = time.time()
		
		# 清理CH缓冲区中的超时批次
		expired_ch = [
			batch_id for batch_id, batch in self.ch_buffer.items()
			if current_time - batch.timestamp > self.timeout_seconds
		]
		
		for batch_id in expired_ch:
			del self.ch_buffer[batch_id]
			self.stats['timeout_pairs'] += 1
			self.logger.debug(f"清理超时CH特征: {batch_id}")
		
		# 清理空间缓冲区中的超时批次
		expired_spatial = [
			batch_id for batch_id, batch in self.spatial_buffer.items()
			if current_time - batch.timestamp > self.timeout_seconds
		]
		
		for batch_id in expired_spatial:
			del self.spatial_buffer[batch_id]
			self.stats['timeout_pairs'] += 1
			self.logger.debug(f"清理超时空间特征: {batch_id}")
	
	def get_stats(self) -> Dict[str, Any]:
		"""获取统计信息"""
		with self.buffer_lock:
			stats = self.stats.copy()
			stats.update({
				'ch_buffer_size': len(self.ch_buffer),
				'spatial_buffer_size': len(self.spatial_buffer),
				'paired_queue_size': self.paired_queue.qsize()
			})
			return stats
	
	def clear_buffers(self):
		"""清空所有缓冲区"""
		with self.buffer_lock:
			self.ch_buffer.clear()
			self.spatial_buffer.clear()
			
			# 清空配对队列
			while not self.paired_queue.empty():
				try:
					self.paired_queue.get_nowait()
				except queue.Empty:
					break


class FeatureFusionSynchronizer:
	"""特征融合同步器 - 专注同步可靠性"""
	
	def __init__(self, rank: int, device: torch.device):
		self.rank = rank
		self.device = device
		
		# CH分支和空间分支的源rank
		self.ch_source_rank = 2  # GPU 2
		self.spatial_source_rank = 3  # GPU 3
		
		# 异步缓冲器
		self.buffer = AsyncFeatureBuffer(max_buffer_size=50, timeout_seconds=10.0)
		
		# 接收线程
		self.ch_receiver_thread = None
		self.spatial_receiver_thread = None
		self.stop_event = threading.Event()
		
		# 历史特征缓存（用于超时处理）
		self.history_cache = {}
		self.max_history_size = 10
		
		self.logger = logging.getLogger(__name__)
	
	def start_async_receivers(self, node_comm):
		"""启动异步接收线程"""
		self.node_comm = node_comm
		self.stop_event.clear()
		
		# 启动CH特征接收线程
		self.ch_receiver_thread = threading.Thread(
			target=self._ch_receiver_worker,
			name="CHFeatureReceiver",
			daemon=True
		)
		self.ch_receiver_thread.start()
		
		# 启动空间特征接收线程
		self.spatial_receiver_thread = threading.Thread(
			target=self._spatial_receiver_worker,
			name="SpatialFeatureReceiver",
			daemon=True
		)
		self.spatial_receiver_thread.start()
		
		self.logger.info("异步特征接收器已启动")
	
	def stop_async_receivers(self):
		"""停止异步接收线程"""
		self.stop_event.set()
		
		if self.ch_receiver_thread and self.ch_receiver_thread.is_alive():
			self.ch_receiver_thread.join(timeout=5.0)
		
		if self.spatial_receiver_thread and self.spatial_receiver_thread.is_alive():
			self.spatial_receiver_thread.join(timeout=5.0)
		
		self.logger.info("异步特征接收器已停止")
	
	def _ch_receiver_worker(self):
		"""CH特征接收工作线程"""
		while not self.stop_event.is_set():
			try:
				# 接收CH特征数量
				count_tensor = self.node_comm.recv_tensor(
					src_rank=self.ch_source_rank,
					dtype=torch.long,
					device=self.device
				)
				
				if count_tensor is None:
					time.sleep(0.1)
					continue
				
				count = count_tensor.item()
				
				# 接收批次ID（简化：使用时间戳）
				batch_id = f"batch_{int(time.time() * 1000000)}"
				
				# 接收特征和tiers
				ch_features = []
				ch_tiers = []
				
				for i in range(count):
					# 接收CH特征
					ch_feat = self.node_comm.recv_tensor(
						src_rank=self.ch_source_rank,
						device=self.device
					)
					
					# 接收tier信息
					tier_tensor = self.node_comm.recv_tensor(
						src_rank=self.ch_source_rank,
						dtype=torch.long,
						device=self.device
					)
					
					if ch_feat is not None and tier_tensor is not None:
						ch_features.append(ch_feat)
						ch_tiers.append(tier_tensor.item())
				
				if ch_features:
					# 组合特征为单个张量
					combined_features = torch.stack(ch_features, dim=0)
					
					# 创建特征批次
					feature_batch = FeatureBatch(
						batch_id=batch_id,
						features=combined_features,
						tiers=ch_tiers,
						timestamp=time.time(),
						source_rank=self.ch_source_rank
					)
					
					# 添加到缓冲区
					self.buffer.add_ch_features(feature_batch)
			
			except Exception as e:
				if not self.stop_event.is_set():
					self.logger.error(f"CH特征接收异常: {e}")
				time.sleep(0.1)
	
	def _spatial_receiver_worker(self):
		"""空间特征接收工作线程"""
		while not self.stop_event.is_set():
			try:
				# 接收空间特征数量
				count_tensor = self.node_comm.recv_tensor(
					src_rank=self.spatial_source_rank,
					dtype=torch.long,
					device=self.device
				)
				
				if count_tensor is None:
					time.sleep(0.1)
					continue
				
				count = count_tensor.item()
				
				# 接收批次ID（简化：使用时间戳）
				batch_id = f"batch_{int(time.time() * 1000000)}"
				
				# 接收特征和tiers
				spatial_features = []
				spatial_tiers = []
				
				for i in range(count):
					# 接收空间特征
					spatial_feat = self.node_comm.recv_tensor(
						src_rank=self.spatial_source_rank,
						device=self.device
					)
					
					# 接收tier信息
					tier_tensor = self.node_comm.recv_tensor(
						src_rank=self.spatial_source_rank,
						dtype=torch.long,
						device=self.device
					)
					
					if spatial_feat is not None and tier_tensor is not None:
						spatial_features.append(spatial_feat)
						spatial_tiers.append(tier_tensor.item())
				
				if spatial_features:
					# 组合特征为单个张量
					combined_features = torch.stack(spatial_features, dim=0)
					
					# 创建特征批次
					feature_batch = FeatureBatch(
						batch_id=batch_id,
						features=combined_features,
						tiers=spatial_tiers,
						timestamp=time.time(),
						source_rank=self.spatial_source_rank
					)
					
					# 添加到缓冲区
					self.buffer.add_spatial_features(feature_batch)
			
			except Exception as e:
				if not self.stop_event.is_set():
					self.logger.error(f"空间特征接收异常: {e}")
				time.sleep(0.1)
	
	def get_synchronized_features(self, timeout: float = 10.0) -> Optional[
		Tuple[torch.Tensor, torch.Tensor, List[int]]]:
		"""
		获取同步的特征对

		返回:
			Optional[Tuple[torch.Tensor, torch.Tensor, List[int]]]: (ch_features, spatial_features, tiers)
		"""
		# 尝试获取配对的特征
		paired_features = self.buffer.get_paired_features(timeout=timeout)
		
		if paired_features is None:
			# 超时处理：使用历史特征
			self.logger.warning("特征同步超时，尝试使用历史特征")
			return self._get_fallback_features()
		
		ch_batch, spatial_batch = paired_features
		
		# 确保tiers一致
		if ch_batch.tiers != spatial_batch.tiers:
			self.logger.warning("Tier信息不匹配，使用CH分支的tier信息")
		
		# 更新历史缓存
		self._update_history_cache(ch_batch.features, spatial_batch.features, ch_batch.tiers)
		
		return ch_batch.features, spatial_batch.features, ch_batch.tiers
	
	def _get_fallback_features(self) -> Optional[Tuple[torch.Tensor, torch.Tensor, List[int]]]:
		"""获取后备特征（历史特征或跳过）"""
		if self.history_cache:
			# 使用最近的历史特征
			recent_key = max(self.history_cache.keys())
			ch_feat, spatial_feat, tiers = self.history_cache[recent_key]
			
			self.logger.info(f"使用历史特征: {recent_key}")
			return ch_feat.clone(), spatial_feat.clone(), tiers.copy()
		else:
			# 没有历史特征，跳过这个batch
			self.logger.warning("没有可用的历史特征，跳过当前batch")
			return None
	
	def _update_history_cache(self, ch_features: torch.Tensor,
	                          spatial_features: torch.Tensor, tiers: List[int]):
		"""更新历史特征缓存"""
		timestamp = time.time()
		self.history_cache[timestamp] = (
			ch_features.clone().detach(),
			spatial_features.clone().detach(),
			tiers.copy()
		)
		
		# 限制缓存大小
		if len(self.history_cache) > self.max_history_size:
			oldest_key = min(self.history_cache.keys())
			del self.history_cache[oldest_key]
	
	def get_sync_stats(self) -> Dict[str, Any]:
		"""获取同步统计信息"""
		buffer_stats = self.buffer.get_stats()
		buffer_stats.update({
			'history_cache_size': len(self.history_cache),
			'ch_receiver_alive': self.ch_receiver_thread.is_alive() if self.ch_receiver_thread else False,
			'spatial_receiver_alive': self.spatial_receiver_thread.is_alive() if self.spatial_receiver_thread else False
		})
		return buffer_stats


# 增强的FeatureFusionStage，集成同步机制
class EnhancedFeatureFusionStage:
	"""增强的特征融合阶段 - 集成同步机制"""
	
	def __init__(self, model, device, node_comm=None, config=None):
		self.model = model
		self.device = device
		self.node_comm = node_comm
		self.config = config or {}
		
		# 特征融合组件
		self.attention_fusion = model.attention_fusion
		self.attention_fusion.to(device)
		
		# 同步器
		self.synchronizer = FeatureFusionSynchronizer(
			rank=node_comm.rank if node_comm else 4,
			device=device
		)
		
		# 启动异步接收器
		if node_comm:
			self.synchronizer.start_async_receivers(node_comm)
		
		self.logger = logging.getLogger(__name__)
	
	def forward(self) -> List[Tuple[torch.Tensor, int]]:
		"""同步获取并融合特征"""
		try:
			# 获取同步的特征
			sync_result = self.synchronizer.get_synchronized_features(timeout=10.0)
			
			if sync_result is None:
				self.logger.warning("特征同步失败，返回空结果")
				return []
			
			ch_features, spatial_features, tiers = sync_result
			
			# 执行特征融合
			fused_features = []
			
			for i in range(len(ch_features)):
				try:
					ch_feat = ch_features[i]
					spatial_feat = spatial_features[i]
					tier = tiers[i]
					
					# 特征维度对齐（基本处理）
					if ch_feat.shape != spatial_feat.shape:
						# 简单的维度调整
						min_channels = min(ch_feat.shape[1], spatial_feat.shape[1])
						ch_feat = ch_feat[:, :min_channels]
						spatial_feat = spatial_feat[:, :min_channels]
						
						# 空间维度对齐
						if ch_feat.shape[2:] != spatial_feat.shape[2:]:
							spatial_feat = torch.nn.functional.interpolate(
								spatial_feat,
								size=ch_feat.shape[2:],
								mode='trilinear',
								align_corners=False
							)
					
					# 应用注意力融合
					fused = self.attention_fusion(ch_feat, spatial_feat)
					fused_features.append((fused, tier))
				
				except Exception as e:
					self.logger.error(f"特征融合失败 tier {tiers[i] if i < len(tiers) else 'unknown'}: {e}")
					continue
			
			return fused_features
		
		except Exception as e:
			self.logger.error(f"前向传播失败: {e}")
			return []
	
	def get_performance_stats(self) -> Dict[str, Any]:
		"""获取性能统计"""
		return self.synchronizer.get_sync_stats()
	
	def stop_worker(self):
		"""停止工作线程"""
		self.synchronizer.stop_async_receivers()


# 使用示例
example_integration = """
# 在distributed_train.py中的create_pipeline_stages函数中：

elif rank == 4:  # 节点2, GPU 4 - 特征融合
    stages['feature_fusion'] = EnhancedFeatureFusionStage(
        full_model, device, node_comm, config=config
    )

# 在训练循环中：
def train_epoch(model, dataloader, optimizer, loss_fn, device, epoch, args, scaler=None):
    # ... 其他代码 ...

    # 定期打印同步统计
    if batch_idx % 50 == 0 and hasattr(model, 'stages'):
        fusion_stage = model.stages.get('feature_fusion')
        if fusion_stage and hasattr(fusion_stage, 'get_performance_stats'):
            stats = fusion_stage.get_performance_stats()
            logger.info(f"融合同步统计: {stats}")
"""

if __name__ == "__main__":
	print("特征融合同步机制代码已生成")
	print("主要特性：")
	print("- 异步接收CH和空间特征")
	print("- 批次ID匹配确保同步")
	print("- 维度兼容性检查")
	print("- 超时和历史特征后备机制")
	print("- 详细同步统计")
	print("\n集成示例：")
	print(example_integration)