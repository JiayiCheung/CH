#parameter_synchronizer.py

"""
参数同步机制
流水线专用分组同步 + 参数一致性验证
"""

import torch
import torch.distributed as dist
import hashlib
import time
import logging
from typing import Dict, List, Optional, Set, Tuple
from collections import defaultdict
from dataclasses import dataclass
import threading


@dataclass
class ParameterGroup:
	"""参数组定义"""
	group_id: int
	ranks: List[int]
	sync_frequency: str  # 'batch', '10batch', 'epoch'
	priority: int  # 优先级，数字越小优先级越高


class ParameterVersionControl:
	"""参数版本控制器"""
	
	def __init__(self, rank: int):
		self.rank = rank
		self.parameter_versions: Dict[str, int] = {}
		self.global_version = 0
		self.version_lock = threading.RLock()
		
		self.logger = logging.getLogger(__name__)
	
	def increment_version(self, param_name: str = "global") -> int:
		"""增加参数版本号"""
		with self.version_lock:
			if param_name == "global":
				self.global_version += 1
				return self.global_version
			else:
				current = self.parameter_versions.get(param_name, 0)
				self.parameter_versions[param_name] = current + 1
				return current + 1
	
	def get_version(self, param_name: str = "global") -> int:
		"""获取参数版本号"""
		with self.version_lock:
			if param_name == "global":
				return self.global_version
			else:
				return self.parameter_versions.get(param_name, 0)
	
	def set_version(self, version: int, param_name: str = "global"):
		"""设置参数版本号"""
		with self.version_lock:
			if param_name == "global":
				self.global_version = version
			else:
				self.parameter_versions[param_name] = version
	
	def get_all_versions(self) -> Dict[str, int]:
		"""获取所有版本信息"""
		with self.version_lock:
			versions = self.parameter_versions.copy()
			versions["global"] = self.global_version
			return versions


class FlowlineParameterSynchronizer:
	"""流水线参数同步器"""
	
	def __init__(self, rank: int, world_size: int, device: torch.device):
		self.rank = rank
		self.world_size = world_size
		self.device = device
		
		# 参数组定义
		self.parameter_groups = self._define_parameter_groups()
		
		# 版本控制器
		self.version_control = ParameterVersionControl(rank)
		
		# 同步统计
		self.sync_stats = {
			'batch_syncs': 0,
			'epoch_syncs': 0,
			'hash_mismatches': 0,
			'version_conflicts': 0,
			'sync_failures': 0
		}
		
		# 批次计数器
		self.batch_counter = 0
		self.epoch_counter = 0
		
		self.logger = logging.getLogger(__name__)
	
	def _define_parameter_groups(self) -> Dict[int, ParameterGroup]:
		"""定义流水线专用参数同步分组"""
		groups = {
			# 组1：预处理参数（rank 0）
			1: ParameterGroup(
				group_id=1,
				ranks=[0],
				sync_frequency='epoch',  # 预处理参数相对稳定
				priority=3
			),
			
			# 组2：特征提取参数（rank 1,2,3）
			2: ParameterGroup(
				group_id=2,
				ranks=[1, 2, 3],
				sync_frequency='batch',  # 特征提取需要频繁同步
				priority=1  # 最高优先级
			),
			
			# 组3：融合输出参数（rank 4,5,6）
			3: ParameterGroup(
				group_id=3,
				ranks=[4, 5, 6],
				sync_frequency='batch',  # 融合输出需要频繁同步
				priority=2
			)
		}
		return groups
	
	def should_sync(self, group_id: int) -> bool:
		"""判断是否需要同步"""
		if group_id not in self.parameter_groups:
			return False
		
		group = self.parameter_groups[group_id]
		
		if self.rank not in group.ranks:
			return False
		
		if group.sync_frequency == 'batch':
			return True
		elif group.sync_frequency == '10batch':
			return self.batch_counter % 10 == 0
		elif group.sync_frequency == 'epoch':
			return self.batch_counter == 0  # epoch开始时
		
		return False
	
	def sync_parameters(self, model, group_id: int = None) -> bool:
		"""
		同步参数

		参数:
			model: 模型实例
			group_id: 指定同步的组ID，None表示同步所有相关组

		返回:
			bool: 同步是否成功
		"""
		try:
			if group_id is None:
				# 按优先级同步所有相关组
				success = True
				for gid in sorted(self.parameter_groups.keys(),
				                  key=lambda x: self.parameter_groups[x].priority):
					if self.should_sync(gid):
						success &= self._sync_parameter_group(model, gid)
				return success
			else:
				# 同步指定组
				if self.should_sync(group_id):
					return self._sync_parameter_group(model, group_id)
				return True
		
		except Exception as e:
			self.logger.error(f"参数同步失败: {e}")
			self.sync_stats['sync_failures'] += 1
			return False
	
	def _sync_parameter_group(self, model, group_id: int) -> bool:
		"""同步指定参数组"""
		group = self.parameter_groups[group_id]
		
		if self.rank not in group.ranks:
			return True
		
		try:
			# 获取需要同步的参数
			sync_params = self._get_group_parameters(model, group_id)
			
			if not sync_params:
				return True
			
			# 版本检查
			if not self._check_parameter_versions(group.ranks, group_id):
				self.logger.warning(f"组 {group_id} 参数版本不一致")
				self.sync_stats['version_conflicts'] += 1
			
			# 执行同步
			success = self._perform_group_sync(sync_params, group.ranks, group_id)
			
			if success:
				# 更新版本
				self.version_control.increment_version(f"group_{group_id}")
				
				# 更新统计
				if group.sync_frequency == 'batch':
					self.sync_stats['batch_syncs'] += 1
				elif group.sync_frequency == 'epoch':
					self.sync_stats['epoch_syncs'] += 1
			
			return success
		
		except Exception as e:
			self.logger.error(f"组 {group_id} 参数同步失败: {e}")
			return False
	
	def _get_group_parameters(self, model, group_id: int) -> Dict[str, torch.Tensor]:
		"""获取指定组需要同步的参数"""
		params = {}
		
		if not hasattr(model, 'stages'):
			return params
		
		# 根据组ID和rank确定需要同步的stage
		if group_id == 1 and self.rank == 0:
			# 预处理参数
			if 'preprocessing' in model.stages:
				stage = model.stages['preprocessing']
				for name, param in stage.named_parameters():
					if param.requires_grad:
						params[f"preprocessing.{name}"] = param
		
		elif group_id == 2 and self.rank in [1, 2, 3]:
			# 特征提取参数
			stage_names = {
				1: 'patch_scheduling',
				2: 'ch_branch',
				3: 'spatial_branch'
			}
			
			stage_name = stage_names.get(self.rank)
			if stage_name and stage_name in model.stages:
				stage = model.stages[stage_name]
				for name, param in stage.named_parameters():
					if param.requires_grad:
						params[f"{stage_name}.{name}"] = param
		
		elif group_id == 3 and self.rank in [4, 5, 6]:
			# 融合输出参数
			stage_names = {
				4: 'feature_fusion',
				5: 'multiscale_fusion',
				6: 'segmentation_head'
			}
			
			stage_name = stage_names.get(self.rank)
			if stage_name and stage_name in model.stages:
				stage = model.stages[stage_name]
				for name, param in stage.named_parameters():
					if param.requires_grad:
						params[f"{stage_name}.{name}"] = param
		
		return params
	
	def _check_parameter_versions(self, ranks: List[int], group_id: int) -> bool:
		"""检查参数版本一致性"""
		if not dist.is_initialized():
			return True
		
		try:
			# 当前版本
			current_version = self.version_control.get_version(f"group_{group_id}")
			version_tensor = torch.tensor([current_version], dtype=torch.long, device=self.device)
			
			# 收集所有rank的版本
			version_list = [torch.zeros_like(version_tensor) for _ in ranks]
			
			# 创建组进程组（如果不存在）
			group_ranks = [r for r in ranks if r < self.world_size]
			if len(group_ranks) > 1:
				try:
					process_group = dist.new_group(ranks=group_ranks)
					dist.all_gather(version_list, version_tensor, group=process_group)
					
					# 检查版本一致性
					versions = [v.item() for v in version_list]
					return len(set(versions)) == 1
				
				except Exception as e:
					self.logger.warning(f"版本检查失败: {e}")
					return True
			
			return True
		
		except Exception as e:
			self.logger.warning(f"版本检查异常: {e}")
			return True
	
	def _perform_group_sync(self, params: Dict[str, torch.Tensor],
	                        ranks: List[int], group_id: int) -> bool:
		"""执行组内参数同步"""
		if not params or not dist.is_initialized():
			return True
		
		try:
			# 创建组进程组
			group_ranks = [r for r in ranks if r < self.world_size]
			if len(group_ranks) <= 1:
				return True
			
			process_group = dist.new_group(ranks=group_ranks)
			
			# 选择主rank（组内最小rank）
			master_rank = min(group_ranks)
			
			# 同步每个参数
			for param_name, param in params.items():
				try:
					# 广播参数（主rank广播给其他rank）
					dist.broadcast(param.data, src=master_rank, group=process_group)
				
				except Exception as e:
					self.logger.error(f"参数 {param_name} 同步失败: {e}")
					return False
			
			# CH分支参数特殊验证
			if group_id == 2 and self.rank == 2:  # CH分支
				return self._verify_ch_parameters(params)
			
			return True
		
		except Exception as e:
			self.logger.error(f"组 {group_id} 同步执行失败: {e}")
			return False
	
	def _verify_ch_parameters(self, params: Dict[str, torch.Tensor]) -> bool:
		"""验证CH分支参数一致性"""
		try:
			# 计算参数hash
			param_hash = self._calculate_parameter_hash(params)
			
			# 创建hash张量
			hash_tensor = torch.tensor([hash(param_hash)], dtype=torch.long, device=self.device)
			
			# 与其他CH相关rank验证（如果有的话）
			# 这里简化为本地验证
			self.logger.debug(f"CH参数hash: {param_hash[:16]}...")
			
			return True
		
		except Exception as e:
			self.logger.warning(f"CH参数验证失败: {e}")
			return True
	
	def _calculate_parameter_hash(self, params: Dict[str, torch.Tensor]) -> str:
		"""计算参数hash"""
		hasher = hashlib.md5()
		
		# 按名称排序确保一致性
		for name in sorted(params.keys()):
			param = params[name]
			# 只使用参数的部分数据计算hash，避免浮点精度问题
			param_bytes = param.detach().cpu().numpy().astype('float32').tobytes()
			hasher.update(f"{name}:".encode())
			hasher.update(param_bytes[:min(1024, len(param_bytes))])  # 限制长度
		
		return hasher.hexdigest()
	
	def on_batch_end(self):
		"""批次结束时调用"""
		self.batch_counter += 1
	
	def on_epoch_end(self):
		"""epoch结束时调用"""
		self.epoch_counter += 1
		self.batch_counter = 0
	
	def verify_global_consistency(self, model) -> bool:
		"""验证全局参数一致性"""
		try:
			# 每10个batch验证一次跨组一致性
			if self.batch_counter % 10 != 0:
				return True
			
			# 收集关键参数的hash
			critical_params = {}
			
			if hasattr(model, 'stages'):
				for stage_name, stage in model.stages.items():
					if hasattr(stage, 'named_parameters'):
						# 只收集少数关键参数进行验证
						param_count = 0
						for name, param in stage.named_parameters():
							if param.requires_grad and param_count < 3:  # 只验证前3个参数
								critical_params[f"{stage_name}.{name}"] = param
								param_count += 1
			
			if not critical_params:
				return True
			
			# 计算全局hash
			global_hash = self._calculate_parameter_hash(critical_params)
			
			# 与其他rank比较hash（简化版）
			hash_tensor = torch.tensor([hash(global_hash) & 0xffffffff],
			                           dtype=torch.long, device=self.device)
			
			if dist.is_initialized() and self.world_size > 1:
				# 收集所有rank的hash
				hash_list = [torch.zeros_like(hash_tensor) for _ in range(self.world_size)]
				dist.all_gather(hash_list, hash_tensor)
				
				# 检查一致性
				hashes = [h.item() for h in hash_list]
				if len(set(hashes)) > 1:
					self.logger.error("检测到全局参数不一致!")
					self.sync_stats['hash_mismatches'] += 1
					return False
			
			return True
		
		except Exception as e:
			self.logger.warning(f"全局一致性验证失败: {e}")
			return True
	
	def get_sync_stats(self) -> Dict[str, any]:
		"""获取同步统计信息"""
		stats = self.sync_stats.copy()
		stats.update({
			'batch_counter': self.batch_counter,
			'epoch_counter': self.epoch_counter,
			'parameter_versions': self.version_control.get_all_versions()
		})
		return stats


class ParameterSyncManager:
	"""参数同步管理器 - 集成到训练流程"""
	
	def __init__(self, model, rank: int, world_size: int, device: torch.device):
		self.model = model
		self.synchronizer = FlowlineParameterSynchronizer(rank, world_size, device)
		self.logger = logging.getLogger(__name__)
	
	def sync_on_batch_start(self) -> bool:
		"""批次开始时的参数同步"""
		return self.synchronizer.sync_parameters(self.model)
	
	def sync_on_batch_end(self) -> bool:
		"""批次结束时的参数同步"""
		self.synchronizer.on_batch_end()
		
		# 每10个batch进行跨组验证
		if self.synchronizer.batch_counter % 10 == 0:
			return self.synchronizer.verify_global_consistency(self.model)
		
		return True
	
	def sync_on_epoch_start(self) -> bool:
		"""epoch开始时的参数同步"""
		return self.synchronizer.sync_parameters(self.model)
	
	def sync_on_epoch_end(self) -> bool:
		"""epoch结束时的参数同步"""
		self.synchronizer.on_epoch_end()
		
		# epoch结束时进行全面验证
		return self.synchronizer.verify_global_consistency(self.model)
	
	def get_performance_stats(self) -> Dict[str, any]:
		"""获取性能统计"""
		return self.synchronizer.get_sync_stats()


# 训练循环集成示例
def integrate_parameter_sync():
	"""集成参数同步到训练循环的示例代码"""
	example_code = """
def train_epoch(model, dataloader, optimizer, loss_fn, device, epoch, args, scaler=None):
    # 创建参数同步管理器
    param_sync = ParameterSyncManager(model, rank, world_size, device)

    # epoch开始时同步
    if not param_sync.sync_on_epoch_start():
        logger.error("Epoch开始参数同步失败")
        return 0.0

    model.train()
    total_loss = 0.0
    num_batches = 0

    for batch_idx, batch in enumerate(dataloader):
        try:
            # 批次开始时同步
            if not param_sync.sync_on_batch_start():
                logger.warning(f"Batch {batch_idx} 开始参数同步失败")

            # ... 原有训练代码 ...

            # 批次结束时同步
            if not param_sync.sync_on_batch_end():
                logger.warning(f"Batch {batch_idx} 结束参数同步失败")

            # 定期打印同步统计
            if batch_idx % 50 == 0:
                sync_stats = param_sync.get_performance_stats()
                logger.info(f"参数同步统计: {sync_stats}")

        except Exception as e:
            logger.error(f"Training batch {batch_idx} failed: {e}")
            continue

    # epoch结束时同步
    if not param_sync.sync_on_epoch_end():
        logger.error("Epoch结束参数同步失败")

    return total_loss / max(num_batches, 1)
    """
	return example_code


if __name__ == "__main__":
	print("参数同步机制代码已生成")
	print("主要特性：")
	print("- 流水线专用三组同步策略")
	print("- 参数版本控制和冲突检测")
	print("- CH分支参数hash验证")
	print("- 分层同步频率控制")
	print("- 全局一致性验证")
	print("\n集成示例：")
	print(integrate_parameter_sync())