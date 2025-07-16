#!batch_tracking.py

"""
医学级batch_id追踪机制
数据溯源 + CRC校验 + 序列连续性验证
"""

import torch
import time
import threading
import hashlib
import zlib
import logging
from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass
from collections import defaultdict, deque
import json


@dataclass
class BatchID:
	"""医学级批次ID"""
	epoch: int  # 4位
	batch_idx: int  # 6位
	timestamp_ns: int  # 10位纳秒时间戳
	rank: int  # 2位
	sequence: int  # 4位单调递增序列
	crc: int  # 2位CRC校验
	
	def __str__(self) -> str:
		"""生成ID字符串"""
		return f"{self.epoch:04d}{self.batch_idx:06d}{self.timestamp_ns:010d}{self.rank:02d}{self.sequence:04d}{self.crc:02d}"
	
	def __hash__(self) -> int:
		"""计算hash值"""
		return hash(str(self))
	
	def __eq__(self, other) -> bool:
		"""相等比较"""
		if not isinstance(other, BatchID):
			return False
		return str(self) == str(other)
	
	@classmethod
	def from_string(cls, id_str: str) -> 'BatchID':
		"""从字符串解析BatchID"""
		if len(id_str) != 28:
			raise ValueError(f"Invalid BatchID format: {id_str}")
		
		return cls(
			epoch=int(id_str[0:4]),
			batch_idx=int(id_str[4:10]),
			timestamp_ns=int(id_str[10:20]),
			rank=int(id_str[20:22]),
			sequence=int(id_str[22:26]),
			crc=int(id_str[26:28])
		)
	
	def verify_crc(self) -> bool:
		"""验证CRC校验码"""
		data = f"{self.epoch:04d}{self.batch_idx:06d}{self.timestamp_ns:010d}{self.rank:02d}{self.sequence:04d}"
		calculated_crc = zlib.crc32(data.encode()) & 0xff
		return calculated_crc == self.crc
	
	def is_consecutive(self, other: 'BatchID') -> bool:
		"""检查序列连续性"""
		if self.rank != other.rank or self.epoch != other.epoch:
			return False
		
		# 检查序列号或batch_idx连续性
		return (abs(self.sequence - other.sequence) == 1 or
		        abs(self.batch_idx - other.batch_idx) == 1)


class BatchIDGenerator:
	"""批次ID生成器"""
	
	def __init__(self, rank: int):
		self.rank = rank
		self.sequence_counter = 0
		self.last_timestamp = 0
		self.generation_lock = threading.RLock()
		
		self.logger = logging.getLogger(__name__)
	
	def generate(self, epoch: int, batch_idx: int) -> BatchID:
		"""生成新的批次ID"""
		with self.generation_lock:
			# 生成纳秒级时间戳
			current_ns = time.time_ns()
			
			# 确保时间戳单调递增
			if current_ns <= self.last_timestamp:
				current_ns = self.last_timestamp + 1
			
			self.last_timestamp = current_ns
			
			# 序列号单调递增
			self.sequence_counter += 1
			
			# 构建数据用于CRC计算
			data = f"{epoch:04d}{batch_idx:06d}{current_ns:010d}{self.rank:02d}{self.sequence_counter:04d}"
			crc = zlib.crc32(data.encode()) & 0xff
			
			batch_id = BatchID(
				epoch=epoch,
				batch_idx=batch_idx,
				timestamp_ns=current_ns % (10 ** 10),  # 保持10位
				rank=self.rank,
				sequence=self.sequence_counter % 10000,  # 保持4位
				crc=crc
			)
			
			self.logger.debug(f"生成BatchID: {batch_id}")
			return batch_id


class BatchIDValidator:
	"""批次ID验证器"""
	
	def __init__(self, rank: int):
		self.rank = rank
		self.received_ids: Set[str] = set()
		self.sequence_history: Dict[int, deque] = defaultdict(lambda: deque(maxlen=100))
		self.validation_lock = threading.RLock()
		
		# 验证统计
		self.validation_stats = {
			'total_validated': 0,
			'crc_failures': 0,
			'duplicate_ids': 0,
			'sequence_gaps': 0,
			'timestamp_anomalies': 0
		}
		
		self.logger = logging.getLogger(__name__)
	
	def validate(self, batch_id: BatchID) -> Tuple[bool, List[str]]:
		"""
		验证批次ID

		返回:
			Tuple[bool, List[str]]: (是否有效, 错误信息列表)
		"""
		with self.validation_lock:
			errors = []
			
			try:
				self.validation_stats['total_validated'] += 1
				
				# 1. CRC校验
				if not batch_id.verify_crc():
					errors.append(f"CRC校验失败: 期望 {batch_id.crc}")
					self.validation_stats['crc_failures'] += 1
				
				# 2. 唯一性检查
				id_str = str(batch_id)
				if id_str in self.received_ids:
					errors.append(f"重复的BatchID: {id_str}")
					self.validation_stats['duplicate_ids'] += 1
				else:
					self.received_ids.add(id_str)
				
				# 3. 序列连续性检查
				sequence_errors = self._check_sequence_continuity(batch_id)
				errors.extend(sequence_errors)
				
				# 4. 时间戳合理性检查
				timestamp_errors = self._check_timestamp_validity(batch_id)
				errors.extend(timestamp_errors)
				
				# 5. 记录历史
				self.sequence_history[batch_id.rank].append(batch_id)
				
				is_valid = len(errors) == 0
				
				if not is_valid:
					self.logger.warning(f"BatchID验证失败 {id_str}: {errors}")
				else:
					self.logger.debug(f"BatchID验证通过: {id_str}")
				
				return is_valid, errors
			
			except Exception as e:
				error_msg = f"验证异常: {e}"
				self.logger.error(error_msg)
				return False, [error_msg]
	
	def _check_sequence_continuity(self, batch_id: BatchID) -> List[str]:
		"""检查序列连续性"""
		errors = []
		
		try:
			rank_history = self.sequence_history[batch_id.rank]
			
			if len(rank_history) > 0:
				last_batch_id = rank_history[-1]
				
				# 检查序列连续性
				if not batch_id.is_consecutive(last_batch_id):
					# 允许一定的序列跳跃（例如由于重试或丢失）
					sequence_gap = abs(batch_id.sequence - last_batch_id.sequence)
					batch_gap = abs(batch_id.batch_idx - last_batch_id.batch_idx)
					
					if sequence_gap > 5 or batch_gap > 5:  # 允许最多5个序列跳跃
						errors.append(f"序列跳跃过大: 当前 {batch_id.sequence}, 上一个 {last_batch_id.sequence}")
						self.validation_stats['sequence_gaps'] += 1
		
		except Exception as e:
			errors.append(f"序列检查异常: {e}")
		
		return errors
	
	def _check_timestamp_validity(self, batch_id: BatchID) -> List[str]:
		"""检查时间戳有效性"""
		errors = []
		
		try:
			current_ns = time.time_ns()
			id_timestamp_full = batch_id.timestamp_ns + (current_ns // (10 ** 10)) * (10 ** 10)
			
			# 检查时间戳不能太远未来或过去
			time_diff = abs(current_ns - id_timestamp_full)
			max_diff = 24 * 3600 * 10 ** 9  # 24小时（纳秒）
			
			if time_diff > max_diff:
				errors.append(f"时间戳异常: 时间差 {time_diff // 10 ** 9} 秒")
				self.validation_stats['timestamp_anomalies'] += 1
		
		except Exception as e:
			errors.append(f"时间戳检查异常: {e}")
		
		return errors
	
	def get_validation_stats(self) -> Dict[str, any]:
		"""获取验证统计"""
		with self.validation_lock:
			stats = self.validation_stats.copy()
			stats['unique_ids_count'] = len(self.received_ids)
			stats['ranks_tracked'] = len(self.sequence_history)
			return stats


class DataTraceabilityManager:
	"""数据可追溯性管理器"""
	
	def __init__(self, rank: int):
		self.rank = rank
		self.id_generator = BatchIDGenerator(rank)
		self.id_validator = BatchIDValidator(rank)
		
		# 完整流转路径记录
		self.trace_log: Dict[str, List[Dict[str, any]]] = {}
		self.trace_lock = threading.RLock()
		
		self.logger = logging.getLogger(__name__)
	
	def create_batch_id(self, epoch: int, batch_idx: int) -> BatchID:
		"""创建批次ID"""
		batch_id = self.id_generator.generate(epoch, batch_idx)
		
		# 记录创建事件
		self._log_trace_event(batch_id, "created", {
			'rank': self.rank,
			'timestamp': time.time(),
			'action': 'batch_id_created'
		})
		
		return batch_id
	
	def validate_batch_id(self, batch_id: BatchID, source_rank: int = None) -> bool:
		"""验证批次ID"""
		is_valid, errors = self.id_validator.validate(batch_id)
		
		# 记录验证事件
		self._log_trace_event(batch_id, "validated", {
			'rank': self.rank,
			'source_rank': source_rank,
			'timestamp': time.time(),
			'is_valid': is_valid,
			'errors': errors,
			'action': 'batch_id_validated'
		})
		
		return is_valid
	
	def trace_batch_transmission(self, batch_id: BatchID,
	                             from_rank: int, to_rank: int,
	                             data_type: str, data_size: int = None):
		"""记录批次传输"""
		self._log_trace_event(batch_id, "transmitted", {
			'from_rank': from_rank,
			'to_rank': to_rank,
			'data_type': data_type,
			'data_size': data_size,
			'timestamp': time.time(),
			'action': 'batch_transmitted'
		})
	
	def trace_batch_processing(self, batch_id: BatchID,
	                           stage_name: str, processing_time: float = None,
	                           success: bool = True, error_msg: str = None):
		"""记录批次处理"""
		self._log_trace_event(batch_id, "processed", {
			'rank': self.rank,
			'stage_name': stage_name,
			'processing_time': processing_time,
			'success': success,
			'error_msg': error_msg,
			'timestamp': time.time(),
			'action': 'batch_processed'
		})
	
	def _log_trace_event(self, batch_id: BatchID, event_type: str, event_data: Dict[str, any]):
		"""记录追踪事件"""
		with self.trace_lock:
			id_str = str(batch_id)
			
			if id_str not in self.trace_log:
				self.trace_log[id_str] = []
			
			event = {
				'event_type': event_type,
				'timestamp': time.time(),
				**event_data
			}
			
			self.trace_log[id_str].append(event)
			
			# 限制日志大小
			if len(self.trace_log[id_str]) > 50:
				self.trace_log[id_str] = self.trace_log[id_str][-30:]
	
	def get_batch_trace(self, batch_id: BatchID) -> List[Dict[str, any]]:
		"""获取批次完整追踪记录"""
		with self.trace_lock:
			id_str = str(batch_id)
			return self.trace_log.get(id_str, []).copy()
	
	def export_trace_log(self, output_file: str = None) -> str:
		"""导出追踪日志"""
		with self.trace_lock:
			if output_file is None:
				output_file = f"trace_log_rank_{self.rank}_{int(time.time())}.json"
			
			# 导出完整日志
			export_data = {
				'rank': self.rank,
				'export_timestamp': time.time(),
				'validation_stats': self.id_validator.get_validation_stats(),
				'trace_log': self.trace_log
			}
			
			with open(output_file, 'w') as f:
				json.dump(export_data, f, indent=2, default=str)
			
			self.logger.info(f"追踪日志已导出: {output_file}")
			return output_file
	
	def cleanup_old_traces(self, keep_hours: int = 24):
		"""清理旧的追踪记录"""
		with self.trace_lock:
			current_time = time.time()
			cutoff_time = current_time - (keep_hours * 3600)
			
			ids_to_remove = []
			for id_str, events in self.trace_log.items():
				if events and events[-1]['timestamp'] < cutoff_time:
					ids_to_remove.append(id_str)
			
			for id_str in ids_to_remove:
				del self.trace_log[id_str]
			
			self.logger.info(f"清理了 {len(ids_to_remove)} 个旧追踪记录")
	
	def get_performance_stats(self) -> Dict[str, any]:
		"""获取性能统计"""
		with self.trace_lock:
			stats = self.id_validator.get_validation_stats()
			stats.update({
				'total_traces': len(self.trace_log),
				'sequence_counter': self.id_generator.sequence_counter,
				'rank': self.rank
			})
			return stats


# 增强的通信接口，集成BatchID追踪
class TrackedCommunication:
	"""集成BatchID追踪的通信接口"""
	
	def __init__(self, node_comm, rank: int):
		self.node_comm = node_comm
		self.trace_manager = DataTraceabilityManager(rank)
		self.rank = rank
		
		self.logger = logging.getLogger(__name__)
	
	def send_with_tracking(self, tensor: torch.Tensor, dst_rank: int,
	                       batch_id: BatchID, data_type: str, tag: int = 0) -> bool:
		"""发送数据并追踪"""
		try:
			start_time = time.time()
			
			# 记录传输开始
			self.trace_manager.trace_batch_transmission(
				batch_id, self.rank, dst_rank, data_type, tensor.numel()
			)
			
			# 发送BatchID
			id_tensor = torch.tensor([ord(c) for c in str(batch_id)], dtype=torch.uint8)
			self.node_comm.send_tensor(id_tensor, dst_rank, tag)
			
			# 发送实际数据
			success = self.node_comm.send_tensor(tensor, dst_rank, tag + 1)
			
			processing_time = time.time() - start_time
			
			# 记录传输结果
			self.trace_manager.trace_batch_processing(
				batch_id, f"send_to_rank_{dst_rank}", processing_time, success
			)
			
			return success
		
		except Exception as e:
			self.logger.error(f"追踪发送失败: {e}")
			return False
	
	def recv_with_tracking(self, src_rank: int, expected_data_type: str,
	                       tag: int = 0) -> Tuple[Optional[torch.Tensor], Optional[BatchID]]:
		"""接收数据并追踪"""
		try:
			start_time = time.time()
			
			# 接收BatchID
			id_tensor = self.node_comm.recv_tensor(src_rank, tag, dtype=torch.uint8)
			if id_tensor is None:
				return None, None
			
			id_str = ''.join([chr(c) for c in id_tensor.cpu().numpy()])
			
			try:
				batch_id = BatchID.from_string(id_str)
			except ValueError as e:
				self.logger.error(f"无效的BatchID格式: {id_str}")
				return None, None
			
			# 验证BatchID
			if not self.trace_manager.validate_batch_id(batch_id, src_rank):
				self.logger.warning(f"BatchID验证失败: {batch_id}")
			
			# 接收实际数据
			tensor = self.node_comm.recv_tensor(src_rank, tag + 1)
			
			processing_time = time.time() - start_time
			success = tensor is not None
			
			# 记录接收
			self.trace_manager.trace_batch_transmission(
				batch_id, src_rank, self.rank, expected_data_type,
				tensor.numel() if tensor is not None else 0
			)
			
			self.trace_manager.trace_batch_processing(
				batch_id, f"recv_from_rank_{src_rank}", processing_time, success
			)
			
			return tensor, batch_id
		
		except Exception as e:
			self.logger.error(f"追踪接收失败: {e}")
			return None, None
	
	def create_batch_id(self, epoch: int, batch_idx: int) -> BatchID:
		"""创建新的批次ID"""
		return self.trace_manager.create_batch_id(epoch, batch_idx)
	
	def get_trace_stats(self) -> Dict[str, any]:
		"""获取追踪统计"""
		return self.trace_manager.get_performance_stats()
	
	def export_traces(self) -> str:
		"""导出追踪日志"""
		return self.trace_manager.export_trace_log()


# 使用示例
def integrate_batch_tracking():
	"""集成批次追踪的示例代码"""
	example_code = """
# 在stages.py中的使用示例：

class TrackedCHProcessingStage(CHProcessingStage):
    def __init__(self, model, device, node_comm=None, config=None):
        super().__init__(model, device, node_comm, config)

        # 创建追踪通信接口
        self.tracked_comm = TrackedCommunication(node_comm, node_comm.rank)

    def forward(self, patches=None, tiers=None):
        if patches is None and self.node_comm:
            # 接收带追踪的数据
            count_tensor, count_batch_id = self.tracked_comm.recv_with_tracking(
                src_rank=prev_rank,
                expected_data_type="patch_count"
            )

            if count_tensor is None:
                return [], []

            count = count_tensor.item()
            patches = []
            tiers = []

            for i in range(count):
                # 接收每个patch
                patch_tensor, patch_batch_id = self.tracked_comm.recv_with_tracking(
                    src_rank=prev_rank,
                    expected_data_type="patch_data"
                )

                if patch_tensor is not None and patch_batch_id is not None:
                    patches.append(patch_tensor)

        # 处理数据
        ch_features, processed_tiers = self.process(patches, tiers)

        # 发送带追踪的结果
        if self.node_comm and ch_features:
            for i, (ch_feat, tier) in enumerate(zip(ch_features, processed_tiers)):
                # 创建新的batch_id用于输出
                output_batch_id = self.tracked_comm.create_batch_id(
                    epoch=0,  # 从训练循环传入
                    batch_idx=i
                )

                # 发送特征
                self.tracked_comm.send_with_tracking(
                    tensor=ch_feat,
                    dst_rank=fusion_rank,
                    batch_id=output_batch_id,
                    data_type="ch_features"
                )

        return ch_features, processed_tiers

# 在训练脚本中定期导出追踪日志：
def train_epoch(model, dataloader, optimizer, loss_fn, device, epoch, args, scaler=None):
    # ... 训练代码 ...

    # 每100个batch导出一次追踪日志
    if batch_idx % 100 == 0:
        for stage_name, stage in model.stages.items():
            if hasattr(stage, 'tracked_comm'):
                trace_file = stage.tracked_comm.export_traces()
                logger.info(f"{stage_name} 追踪日志: {trace_file}")

                # 打印统计信息
                stats = stage.tracked_comm.get_trace_stats()
                logger.info(f"{stage_name} 追踪统计: {stats}")
    """
	return example_code


if __name__ == "__main__":
	print("医学级batch_id追踪机制代码已生成")
	print("主要特性：")
	print("- 28位唯一ID：epoch+batch_idx+timestamp+rank+sequence+CRC")
	print("- CRC32校验确保ID完整性")
	print("- 序列连续性验证")
	print("- 完整流转路径追踪")
	print("- 数据可追溯性管理")
	print("- 医学级数据溯源支持")
	print("\n集成示例：")
	print(integrate_batch_tracking())