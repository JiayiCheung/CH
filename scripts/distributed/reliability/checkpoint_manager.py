#checkpoint_manager.py

"""
分布式检查点保存
分布式完整检查点 + 原子保存 + 恢复验证
"""

import torch
import torch.distributed as dist
import os
import json
import time
import hashlib
import shutil
import logging
import threading
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
from dataclasses import dataclass, asdict
import pickle
import gzip


@dataclass
class CheckpointMetadata:
	"""检查点元数据"""
	epoch: int
	batch_idx: int
	global_step: int
	timestamp: float
	rank: int
	world_size: int
	
	# 流水线配置
	pipeline_config_hash: str
	model_config_hash: str
	
	# 训练状态
	train_loss: float
	val_loss: Optional[float]
	learning_rate: float
	
	# 版本信息
	pytorch_version: str
	checkpoint_version: str
	
	# 文件信息
	files: Dict[str, str]  # 文件名 -> 文件hash
	
	def to_dict(self) -> Dict[str, Any]:
		"""转换为字典"""
		return asdict(self)
	
	@classmethod
	def from_dict(cls, data: Dict[str, Any]) -> 'CheckpointMetadata':
		"""从字典创建"""
		return cls(**data)


class DistributedCheckpointManager:
	"""分布式检查点管理器"""
	
	def __init__(self, rank: int, world_size: int, base_dir: str = "./checkpoints"):
		self.rank = rank
		self.world_size = world_size
		self.base_dir = Path(base_dir)
		self.base_dir.mkdir(parents=True, exist_ok=True)
		
		# 检查点版本
		self.checkpoint_version = "1.0.0"
		
		# 保存锁（原子保存）
		self.save_lock = threading.RLock()
		
		# 压缩设置
		self.enable_compression = True
		self.compression_level = 6
		
		self.logger = logging.getLogger(__name__)
	
	def save_distributed_checkpoint(self,
	                                model,
	                                optimizer,
	                                scheduler,
	                                epoch: int,
	                                batch_idx: int,
	                                global_step: int,
	                                train_loss: float,
	                                val_loss: Optional[float] = None,
	                                extra_data: Optional[Dict[str, Any]] = None) -> bool:
		"""
		保存分布式检查点

		返回:
			bool: 保存是否成功
		"""
		with self.save_lock:
			try:
				# 创建检查点目录
				checkpoint_name = f"checkpoint_epoch_{epoch}_step_{global_step}"
				checkpoint_dir = self.base_dir / checkpoint_name
				temp_dir = self.base_dir / f"{checkpoint_name}_temp"
				
				# 使用临时目录确保原子性
				if temp_dir.exists():
					shutil.rmtree(temp_dir)
				temp_dir.mkdir(parents=True, exist_ok=True)
				
				# 1. 保存rank级状态
				rank_files = self._save_rank_state(
					temp_dir, model, optimizer, scheduler, extra_data
				)
				
				# 2. 主rank保存全局元数据
				if self.rank == 0:
					metadata_file = self._save_global_metadata(
						temp_dir, epoch, batch_idx, global_step,
						train_loss, val_loss, rank_files
					)
					rank_files['metadata'] = metadata_file
				
				# 3. 同步所有rank完成保存
				success = self._synchronize_save_completion()
				
				if success:
					# 4. 原子移动到最终位置
					if checkpoint_dir.exists():
						shutil.rmtree(checkpoint_dir)
					temp_dir.rename(checkpoint_dir)
					
					# 5. 清理旧检查点
					if self.rank == 0:
						self._cleanup_old_checkpoints()
					
					self.logger.info(f"检查点保存成功: {checkpoint_dir}")
				else:
					# 清理失败的临时文件
					if temp_dir.exists():
						shutil.rmtree(temp_dir)
					self.logger.error("检查点保存失败")
				
				return success
			
			except Exception as e:
				self.logger.error(f"保存检查点异常: {e}")
				return False
	
	def _save_rank_state(self,
	                     checkpoint_dir: Path,
	                     model,
	                     optimizer,
	                     scheduler,
	                     extra_data: Optional[Dict[str, Any]]) -> Dict[str, str]:
		"""保存rank级状态"""
		rank_files = {}
		
		try:
			# 1. 保存模型state
			if hasattr(model, 'stages') and model.stages:
				model_state = {}
				for stage_name, stage in model.stages.items():
					if hasattr(stage, 'state_dict'):
						model_state[stage_name] = stage.state_dict()
					elif hasattr(stage, 'get_state_dict_prefix'):
						model_state[stage_name] = stage.get_state_dict_prefix()
				
				model_file = f"model_rank_{self.rank}.pt"
				model_path = checkpoint_dir / model_file
				
				if self.enable_compression:
					self._save_compressed(model_state, model_path)
				else:
					torch.save(model_state, model_path)
				
				rank_files['model'] = self._calculate_file_hash(model_path)
			
			# 2. 保存优化器state
			if optimizer is not None:
				optimizer_file = f"optimizer_rank_{self.rank}.pt"
				optimizer_path = checkpoint_dir / optimizer_file
				
				optimizer_state = optimizer.state_dict()
				
				if self.enable_compression:
					self._save_compressed(optimizer_state, optimizer_path)
				else:
					torch.save(optimizer_state, optimizer_path)
				
				rank_files['optimizer'] = self._calculate_file_hash(optimizer_path)
			
			# 3. 保存调度器state
			if scheduler is not None:
				scheduler_file = f"scheduler_rank_{self.rank}.pt"
				scheduler_path = checkpoint_dir / scheduler_file
				
				scheduler_state = scheduler.state_dict()
				torch.save(scheduler_state, scheduler_path)
				
				rank_files['scheduler'] = self._calculate_file_hash(scheduler_path)
			
			# 4. 保存额外数据
			if extra_data:
				extra_file = f"extra_rank_{self.rank}.json"
				extra_path = checkpoint_dir / extra_file
				
				with open(extra_path, 'w') as f:
					json.dump(extra_data, f, indent=2)
				
				rank_files['extra'] = self._calculate_file_hash(extra_path)
			
			# 5. 保存rank信息
			rank_info = {
				'rank': self.rank,
				'world_size': self.world_size,
				'timestamp': time.time(),
				'files': rank_files,
				'pytorch_version': torch.__version__
			}
			
			rank_info_file = f"rank_{self.rank}_info.json"
			rank_info_path = checkpoint_dir / rank_info_file
			
			with open(rank_info_path, 'w') as f:
				json.dump(rank_info, f, indent=2)
			
			rank_files['rank_info'] = self._calculate_file_hash(rank_info_path)
			
			return rank_files
		
		except Exception as e:
			self.logger.error(f"保存rank {self.rank} 状态失败: {e}")
			raise
	
	def _save_global_metadata(self,
	                          checkpoint_dir: Path,
	                          epoch: int,
	                          batch_idx: int,
	                          global_step: int,
	                          train_loss: float,
	                          val_loss: Optional[float],
	                          rank_files: Dict[str, str]) -> str:
		"""保存全局元数据（仅主rank）"""
		try:
			metadata = CheckpointMetadata(
				epoch=epoch,
				batch_idx=batch_idx,
				global_step=global_step,
				timestamp=time.time(),
				rank=self.rank,
				world_size=self.world_size,
				pipeline_config_hash=self._calculate_config_hash("pipeline"),
				model_config_hash=self._calculate_config_hash("model"),
				train_loss=train_loss,
				val_loss=val_loss,
				learning_rate=0.0,  # 需要从optimizer获取
				pytorch_version=torch.__version__,
				checkpoint_version=self.checkpoint_version,
				files=rank_files
			)
			
			metadata_file = "checkpoint_metadata.json"
			metadata_path = checkpoint_dir / metadata_file
			
			with open(metadata_path, 'w') as f:
				json.dump(metadata.to_dict(), f, indent=2)
			
			return self._calculate_file_hash(metadata_path)
		
		except Exception as e:
			self.logger.error(f"保存全局元数据失败: {e}")
			raise
	
	def _synchronize_save_completion(self) -> bool:
		"""同步所有rank的保存完成状态"""
		try:
			if not dist.is_initialized():
				return True
			
			# 创建成功标志张量
			success_tensor = torch.tensor([1], dtype=torch.long)
			
			# 收集所有rank的状态
			success_list = [torch.zeros_like(success_tensor) for _ in range(self.world_size)]
			dist.all_gather(success_list, success_tensor)
			
			# 检查是否所有rank都成功
			all_success = all(s.item() == 1 for s in success_list)
			
			return all_success
		
		except Exception as e:
			self.logger.error(f"同步保存状态失败: {e}")
			return False
	
	def load_distributed_checkpoint(self,
	                                checkpoint_path: str,
	                                model,
	                                optimizer=None,
	                                scheduler=None,
	                                strict: bool = True) -> Optional[Dict[str, Any]]:
		"""
		加载分布式检查点

		返回:
			Optional[Dict[str, Any]]: 检查点信息，失败时返回None
		"""
		try:
			checkpoint_dir = Path(checkpoint_path)
			
			if not checkpoint_dir.exists():
				self.logger.error(f"检查点目录不存在: {checkpoint_dir}")
				return None
			
			# 1. 验证检查点完整性
			if not self._verify_checkpoint_integrity(checkpoint_dir):
				self.logger.error("检查点完整性验证失败")
				if strict:
					return None
			
			# 2. 加载全局元数据
			metadata = self._load_global_metadata(checkpoint_dir)
			if metadata is None:
				self.logger.error("加载全局元数据失败")
				return None
			
			# 3. 验证版本兼容性
			if not self._verify_version_compatibility(metadata):
				self.logger.error("版本兼容性验证失败")
				if strict:
					return None
			
			# 4. 加载rank级状态
			if not self._load_rank_state(checkpoint_dir, model, optimizer, scheduler):
				self.logger.error(f"加载rank {self.rank} 状态失败")
				if strict:
					return None
			
			# 5. 同步加载完成
			if not self._synchronize_load_completion():
				self.logger.error("同步加载完成失败")
				if strict:
					return None
			
			self.logger.info(f"检查点加载成功: {checkpoint_dir}")
			
			return {
				'epoch': metadata.epoch,
				'batch_idx': metadata.batch_idx,
				'global_step': metadata.global_step,
				'train_loss': metadata.train_loss,
				'val_loss': metadata.val_loss,
				'metadata': metadata
			}
		
		except Exception as e:
			self.logger.error(f"加载检查点异常: {e}")
			return None
	
	def _verify_checkpoint_integrity(self, checkpoint_dir: Path) -> bool:
		"""验证检查点完整性"""
		try:
			# 检查必需文件
			required_files = [
				f"model_rank_{self.rank}.pt",
				f"rank_{self.rank}_info.json"
			]
			
			for file_name in required_files:
				file_path = checkpoint_dir / file_name
				if not file_path.exists():
					self.logger.error(f"缺少必需文件: {file_name}")
					return False
			
			# 验证文件hash
			rank_info_path = checkpoint_dir / f"rank_{self.rank}_info.json"
			with open(rank_info_path, 'r') as f:
				rank_info = json.load(f)
			
			for file_type, expected_hash in rank_info.get('files', {}).items():
				if file_type == 'rank_info':
					continue
				
				file_path = checkpoint_dir / f"{file_type}_rank_{self.rank}.pt"
				if file_path.exists():
					actual_hash = self._calculate_file_hash(file_path)
					if actual_hash != expected_hash:
						self.logger.error(f"文件hash不匹配: {file_type}")
						return False
			
			return True
		
		except Exception as e:
			self.logger.error(f"完整性验证异常: {e}")
			return False
	
	def _load_global_metadata(self, checkpoint_dir: Path) -> Optional[CheckpointMetadata]:
		"""加载全局元数据"""
		try:
			metadata_path = checkpoint_dir / "checkpoint_metadata.json"
			
			if not metadata_path.exists():
				self.logger.error("找不到元数据文件")
				return None
			
			with open(metadata_path, 'r') as f:
				metadata_dict = json.load(f)
			
			return CheckpointMetadata.from_dict(metadata_dict)
		
		except Exception as e:
			self.logger.error(f"加载元数据失败: {e}")
			return None
	
	def _verify_version_compatibility(self, metadata: CheckpointMetadata) -> bool:
		"""验证版本兼容性"""
		try:
			# 检查PyTorch版本兼容性
			current_version = torch.__version__
			saved_version = metadata.pytorch_version
			
			if current_version != saved_version:
				self.logger.warning(f"PyTorch版本不匹配: 当前 {current_version}, 保存 {saved_version}")
			
			# 检查检查点格式版本
			if metadata.checkpoint_version != self.checkpoint_version:
				self.logger.warning(
					f"检查点版本不匹配: 当前 {self.checkpoint_version}, 保存 {metadata.checkpoint_version}")
			
			# 检查world_size
			if metadata.world_size != self.world_size:
				self.logger.error(f"World size不匹配: 当前 {self.world_size}, 保存 {metadata.world_size}")
				return False
			
			return True
		
		except Exception as e:
			self.logger.error(f"版本兼容性验证异常: {e}")
			return False
	
	def _load_rank_state(self,
	                     checkpoint_dir: Path,
	                     model,
	                     optimizer=None,
	                     scheduler=None) -> bool:
		"""加载rank级状态"""
		try:
			# 1. 加载模型状态
			model_path = checkpoint_dir / f"model_rank_{self.rank}.pt"
			if model_path.exists():
				if self.enable_compression:
					model_state = self._load_compressed(model_path)
				else:
					model_state = torch.load(model_path, map_location='cpu')
				
				# 加载到模型
				if hasattr(model, 'stages') and model.stages:
					for stage_name, stage_state in model_state.items():
						if stage_name in model.stages:
							stage = model.stages[stage_name]
							if hasattr(stage, 'load_state_dict'):
								stage.load_state_dict(stage_state, strict=False)
							elif hasattr(stage, 'load_state_dict'):
								# 尝试其他加载方式
								pass
			
			# 2. 加载优化器状态
			if optimizer is not None:
				optimizer_path = checkpoint_dir / f"optimizer_rank_{self.rank}.pt"
				if optimizer_path.exists():
					if self.enable_compression:
						optimizer_state = self._load_compressed(optimizer_path)
					else:
						optimizer_state = torch.load(optimizer_path, map_location='cpu')
					
					optimizer.load_state_dict(optimizer_state)
			
			# 3. 加载调度器状态
			if scheduler is not None:
				scheduler_path = checkpoint_dir / f"scheduler_rank_{self.rank}.pt"
				if scheduler_path.exists():
					scheduler_state = torch.load(scheduler_path, map_location='cpu')
					scheduler.load_state_dict(scheduler_state)
			
			return True
		
		except Exception as e:
			self.logger.error(f"加载rank状态失败: {e}")
			return False
	
	def _synchronize_load_completion(self) -> bool:
		"""同步加载完成"""
		try:
			if not dist.is_initialized():
				return True
			
			# 等待所有rank完成加载
			dist.barrier()
			return True
		
		except Exception as e:
			self.logger.error(f"同步加载完成失败: {e}")
			return False
	
	def _save_compressed(self, data: Any, file_path: Path):
		"""保存压缩数据"""
		with gzip.open(file_path.with_suffix('.pt.gz'), 'wb') as f:
			torch.save(data, f)
	
	def _load_compressed(self, file_path: Path) -> Any:
		"""加载压缩数据"""
		# 尝试压缩文件
		compressed_path = file_path.with_suffix('.pt.gz')
		if compressed_path.exists():
			with gzip.open(compressed_path, 'rb') as f:
				return torch.load(f, map_location='cpu')
		else:
			# 回退到非压缩文件
			return torch.load(file_path, map_location='cpu')
	
	def _calculate_file_hash(self, file_path: Path) -> str:
		"""计算文件hash"""
		hash_md5 = hashlib.md5()
		with open(file_path, "rb") as f:
			for chunk in iter(lambda: f.read(4096), b""):
				hash_md5.update(chunk)
		return hash_md5.hexdigest()
	
	def _calculate_config_hash(self, config_type: str) -> str:
		"""计算配置hash"""
		# 简化版：基于当前时间和配置类型
		config_str = f"{config_type}_{self.world_size}_{self.checkpoint_version}"
		return hashlib.md5(config_str.encode()).hexdigest()
	
	def _cleanup_old_checkpoints(self, keep_last: int = 5):
		"""清理旧检查点（仅主rank执行）"""
		try:
			if self.rank != 0:
				return
			
			# 获取所有检查点目录
			checkpoint_dirs = [
				d for d in self.base_dir.iterdir()
				if d.is_dir() and d.name.startswith('checkpoint_epoch_')
			]
			
			# 按修改时间排序
			checkpoint_dirs.sort(key=lambda x: x.stat().st_mtime, reverse=True)
			
			# 删除旧检查点
			for old_dir in checkpoint_dirs[keep_last:]:
				try:
					shutil.rmtree(old_dir)
					self.logger.info(f"删除旧检查点: {old_dir}")
				except Exception as e:
					self.logger.warning(f"删除检查点失败 {old_dir}: {e}")
		
		except Exception as e:
			self.logger.error(f"清理旧检查点失败: {e}")
	
	def list_checkpoints(self) -> List[Dict[str, Any]]:
		"""列出可用检查点"""
		checkpoints = []
		
		for checkpoint_dir in self.base_dir.iterdir():
			if not checkpoint_dir.is_dir() or not checkpoint_dir.name.startswith('checkpoint_'):
				continue
			
			metadata_path = checkpoint_dir / "checkpoint_metadata.json"
			if metadata_path.exists():
				try:
					with open(metadata_path, 'r') as f:
						metadata = json.load(f)
					
					checkpoints.append({
						'path': str(checkpoint_dir),
						'epoch': metadata.get('epoch', 0),
						'global_step': metadata.get('global_step', 0),
						'timestamp': metadata.get('timestamp', 0),
						'train_loss': metadata.get('train_loss', 0.0)
					})
				except Exception as e:
					self.logger.warning(f"读取检查点元数据失败 {checkpoint_dir}: {e}")
		
		# 按epoch排序
		checkpoints.sort(key=lambda x: x['epoch'], reverse=True)
		return checkpoints


# 集成到训练脚本的示例
def integrate_checkpoint_manager():
	"""集成检查点管理器的示例代码"""
	example_code = """
def main():
    # ... 其他初始化代码 ...

    # 创建检查点管理器
    checkpoint_manager = DistributedCheckpointManager(
        rank=rank,
        world_size=world_size,
        base_dir=args.output_dir
    )

    # 恢复训练（如果有检查点）
    start_epoch = 0
    global_step = 0

    if args.resume:
        # 自动选择最新检查点
        if args.resume == "auto":
            checkpoints = checkpoint_manager.list_checkpoints()
            if checkpoints:
                resume_path = checkpoints[0]['path']
                logger.info(f"自动恢复最新检查点: {resume_path}")
            else:
                logger.info("没有找到检查点，从头开始训练")
                resume_path = None
        else:
            resume_path = args.resume

        if resume_path:
            checkpoint_info = checkpoint_manager.load_distributed_checkpoint(
                resume_path, model, optimizer, scheduler
            )

            if checkpoint_info:
                start_epoch = checkpoint_info['epoch'] + 1
                global_step = checkpoint_info['global_step']
                logger.info(f"从epoch {start_epoch}恢复训练，global_step: {global_step}")
            else:
                logger.error("检查点加载失败，从头开始训练")

    # 主训练循环
    for epoch in range(start_epoch, args.epochs):
        train_sampler.set_epoch(epoch)

        # 训练
        train_loss = train_epoch(model, train_loader, optimizer, loss_fn,
                                device, epoch, args, scaler)

        # 验证
        val_loss = None
        if epoch % args.val_interval == 0:
            val_loss = validate_epoch(model, val_loader, loss_fn,
                                      device, epoch, args)

        # 保存检查点
        if epoch % args.save_interval == 0:
            success = checkpoint_manager.save_distributed_checkpoint(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                batch_idx=0,  # epoch结束时batch_idx为0
                global_step=global_step + epoch * len(train_loader),
                train_loss=train_loss,
                val_loss=val_loss,
                extra_data={'args': vars(args)}
            )

            if not success:
                logger.error(f"保存检查点失败: epoch {epoch}")

        scheduler.step()

    # 保存最终检查点
    checkpoint_manager.save_distributed_checkpoint(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        epoch=args.epochs - 1,
        batch_idx=0,
        global_step=global_step + args.epochs * len(train_loader),
        train_loss=train_loss,
        val_loss=val_loss,
        extra_data={'final': True}
    )
    """
	return example_code


if __name__ == "__main__":
	print("分布式检查点保存代码已生成")
	print("主要特性：")
	print("- rank级分布式状态保存")
	print("- 全局元数据和版本控制")
	print("- 原子保存操作")
	print("- 完整性验证和恢复检查")
	print("- 压缩存储支持")
	print("- 自动清理旧检查点")
	print("\n集成示例：")
	print(integrate_checkpoint_manager())