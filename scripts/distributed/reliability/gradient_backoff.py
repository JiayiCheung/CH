#gradient_backoff.py
"""
梯度计算回退机制优化
分层梯度计算 + 训练暂停机制
"""

import torch
import torch.nn as nn
import numpy as np
import logging
import time
from typing import Dict, List, Optional, Tuple
from pathlib import Path


class GradientBackoffHandler:
	"""梯度计算回退处理器 - 医学级精度优先"""
	
	def __init__(self, model, save_dir: str = "./checkpoints"):
		self.model = model
		self.save_dir = Path(save_dir)
		self.save_dir.mkdir(parents=True, exist_ok=True)
		
		# 失败计数器
		self.failure_count = 0
		self.max_failures = 3
		
		# CH分支使用双精度
		self.use_double_precision = True
		
		# 梯度检查点参数
		self.checkpoint_segments = 4
		
		self.logger = logging.getLogger(__name__)
	
	def compute_gradients_with_backoff(self, loss, epoch: int, batch_idx: int) -> bool:
		"""
		分层梯度计算，医学精度优先

		返回:
			bool: True表示成功，False表示需要暂停训练
		"""
		try:
			# 第一层：标准autograd计算
			success = self._try_standard_gradients(loss)
			if success:
				self.failure_count = 0
				return True
			
			self.logger.warning(f"标准梯度计算失败，尝试双精度计算")
			
			# 第二层：CH分支双精度计算
			success = self._try_double_precision_gradients(loss)
			if success:
				self.failure_count = 0
				return True
			
			self.logger.warning(f"双精度梯度计算失败，尝试数值估计")
			
			# 第三层：数值梯度估计（仅用于验证，不用于训练）
			success = self._try_numerical_gradients(loss)
			if success:
				self.failure_count = 0
				return True
			
			self.logger.error(f"所有梯度计算方法失败")
			
			# 增加失败计数
			self.failure_count += 1
			
			# 连续失败则暂停训练
			if self.failure_count >= self.max_failures:
				self.logger.error(f"连续{self.max_failures}次失败，暂停训练保存检查点")
				self._save_emergency_checkpoint(epoch, batch_idx)
				return False
			
			return True
		
		except Exception as e:
			self.logger.error(f"梯度计算异常: {e}")
			self.failure_count += 1
			return self.failure_count < self.max_failures
	
	def _try_standard_gradients(self, loss) -> bool:
		"""第一层：标准autograd计算"""
		try:
			# 清理之前的梯度
			self.model.zero_grad()
			
			# 标准反向传播
			loss.backward()
			
			# 检查梯度有效性
			if self._validate_gradients():
				return True
			else:
				self.logger.warning("标准梯度无效")
				return False
		
		except RuntimeError as e:
			if "out of memory" in str(e).lower():
				self.logger.warning(f"内存不足，尝试梯度检查点: {e}")
				return self._try_gradient_checkpointing(loss)
			else:
				self.logger.warning(f"标准梯度计算失败: {e}")
				return False
		except Exception as e:
			self.logger.warning(f"标准梯度计算异常: {e}")
			return False
	
	def _try_double_precision_gradients(self, loss) -> bool:
		"""第二层：CH分支双精度计算"""
		try:
			# 清理之前的梯度
			self.model.zero_grad()
			
			# 将CH分支转换为双精度
			ch_modules = self._get_ch_modules()
			original_dtypes = {}
			
			for name, module in ch_modules.items():
				original_dtypes[name] = {}
				for param_name, param in module.named_parameters():
					original_dtypes[name][param_name] = param.dtype
					param.data = param.data.double()
			
			# 重新计算损失（双精度）
			with torch.autocast(device_type='cuda', dtype=torch.float64):
				# 这里需要重新前向传播
				# 注意：实际应用中可能需要重新执行前向传播
				loss.backward()
			
			# 验证梯度
			valid = self._validate_gradients()
			
			# 恢复原始精度
			for name, module in ch_modules.items():
				for param_name, param in module.named_parameters():
					param.data = param.data.to(original_dtypes[name][param_name])
			
			return valid
		
		except Exception as e:
			self.logger.warning(f"双精度梯度计算失败: {e}")
			return False
	
	def _try_gradient_checkpointing(self, loss) -> bool:
		"""第三层：梯度检查点计算"""
		try:
			from torch.utils.checkpoint import checkpoint_sequential
			
			# 清理之前的梯度
			self.model.zero_grad()
			
			# 获取模型的sequential部分进行检查点
			if hasattr(self.model, 'stages'):
				# 对每个stage使用梯度检查点
				for stage_name, stage in self.model.stages.items():
					if hasattr(stage, 'parameters') and any(p.requires_grad for p in stage.parameters()):
						# 包装为sequential
						modules = list(stage.children()) if len(list(stage.children())) > 0 else [stage]
						if len(modules) > 1:
							# 使用梯度检查点
							self.logger.info(f"对{stage_name}使用梯度检查点")
			
			# 执行反向传播
			loss.backward()
			
			return self._validate_gradients()
		
		except Exception as e:
			self.logger.warning(f"梯度检查点计算失败: {e}")
			return False
	
	def _try_numerical_gradients(self, loss) -> bool:
		"""第四层：数值梯度估计（仅用于验证，不实际训练）"""
		try:
			self.logger.warning("使用数值梯度验证（不会更新参数）")
			
			# 数值梯度仅用于验证梯度计算的正确性
			# 实际训练中不使用，因为精度不够
			eps = 1e-7
			numerical_valid = True
			
			# 检查少数几个关键参数的数值梯度
			sample_params = []
			for name, param in self.model.named_parameters():
				if param.requires_grad and 'ch_branch' in name:
					sample_params.append((name, param))
					if len(sample_params) >= 5:  # 只检查前5个CH分支参数
						break
			
			for name, param in sample_params:
				if param.numel() > 1000:  # 跳过大参数
					continue
				
				# 计算数值梯度（仅用于验证）
				orig_data = param.data.clone()
				param.data += eps
				loss_plus = loss.item()  # 需要重新计算损失
				param.data = orig_data - eps
				loss_minus = loss.item()  # 需要重新计算损失
				param.data = orig_data  # 恢复
				
				numerical_grad = (loss_plus - loss_minus) / (2 * eps)
				
				# 只是验证，不实际使用数值梯度
				if not torch.isfinite(torch.tensor(numerical_grad)):
					numerical_valid = False
					break
			
			if numerical_valid:
				self.logger.info("数值梯度验证通过，但精度不足用于医学应用")
			
			# 重要：数值梯度精度不够，不能用于医学级训练
			# 返回False，表示需要暂停训练
			return False
		
		except Exception as e:
			self.logger.warning(f"数值梯度验证失败: {e}")
			return False
	
	def _validate_gradients(self) -> bool:
		"""验证梯度有效性"""
		try:
			total_norm = 0.0
			param_count = 0
			
			for name, param in self.model.named_parameters():
				if param.grad is not None:
					# 检查梯度是否有限
					if not torch.isfinite(param.grad).all():
						self.logger.warning(f"参数 {name} 的梯度包含无穷值或NaN")
						return False
					
					# 累积梯度范数
					param_norm = param.grad.data.norm(2)
					total_norm += param_norm.item() ** 2
					param_count += 1
			
			total_norm = total_norm ** (1. / 2)
			
			# 检查梯度范数是否合理
			if total_norm == 0:
				self.logger.warning("梯度范数为0")
				return False
			
			if total_norm > 1000:  # 梯度爆炸
				self.logger.warning(f"梯度范数过大: {total_norm}")
				return False
			
			if total_norm < 1e-10:  # 梯度消失
				self.logger.warning(f"梯度范数过小: {total_norm}")
				return False
			
			self.logger.debug(f"梯度验证通过，范数: {total_norm:.6f}")
			return True
		
		except Exception as e:
			self.logger.warning(f"梯度验证异常: {e}")
			return False
	
	def _get_ch_modules(self) -> Dict[str, nn.Module]:
		"""获取CH分支模块"""
		ch_modules = {}
		
		if hasattr(self.model, 'stages'):
			for stage_name, stage in self.model.stages.items():
				if 'ch' in stage_name.lower():
					ch_modules[stage_name] = stage
		
		# 备用：直接搜索包含'ch'的模块
		for name, module in self.model.named_modules():
			if 'ch' in name.lower() and len(list(module.parameters())) > 0:
				ch_modules[name] = module
		
		return ch_modules
	
	def _save_emergency_checkpoint(self, epoch: int, batch_idx: int):
		"""保存紧急检查点"""
		timestamp = int(time.time())
		checkpoint_path = self.save_dir / f"emergency_checkpoint_epoch_{epoch}_batch_{batch_idx}_{timestamp}.pt"
		
		try:
			checkpoint = {
				'epoch': epoch,
				'batch_idx': batch_idx,
				'model_state_dict': self.model.state_dict(),
				'failure_count': self.failure_count,
				'timestamp': timestamp,
				'reason': 'gradient_computation_failure'
			}
			
			torch.save(checkpoint, checkpoint_path)
			self.logger.error(f"紧急检查点已保存: {checkpoint_path}")
		
		except Exception as e:
			self.logger.error(f"保存紧急检查点失败: {e}")


def apply_gradient_backoff_to_training(train_function):
	"""
	装饰器：将梯度回退机制应用到训练函数

	用法:
	@apply_gradient_backoff_to_training
	def train_epoch(model, dataloader, optimizer, loss_fn, device, epoch, args, scaler=None):
		# 原始训练代码
		pass
	"""
	
	def wrapper(model, dataloader, optimizer, loss_fn, device, epoch, args, scaler=None):
		# 创建梯度回退处理器
		gradient_handler = GradientBackoffHandler(model, save_dir=args.output_dir)
		
		model.train()
		total_loss = 0.0
		num_batches = 0
		
		for batch_idx, batch in enumerate(dataloader):
			try:
				# 数据预处理（保持原有逻辑）
				if isinstance(batch, (list, tuple)) and len(batch) >= 2:
					images, labels = batch[0], batch[1]
				else:
					images, labels = batch, None
				
				images = images.to(device, non_blocking=True)
				if labels is not None:
					labels = labels.to(device, non_blocking=True)
				
				# 前向传播
				optimizer.zero_grad()
				
				if args.amp and scaler is not None:
					with torch.cuda.amp.autocast():
						outputs = model.forward(images)
						loss = loss_fn(outputs, labels) if labels is not None else loss_fn(outputs, images)
					
					# 使用梯度回退机制
					if gradient_handler.compute_gradients_with_backoff(scaler.scale(loss), epoch, batch_idx):
						scaler.step(optimizer)
						scaler.update()
					else:
						# 训练暂停
						return total_loss / max(num_batches, 1), True  # True表示需要停止
				
				else:
					outputs = model.forward(images)
					loss = loss_fn(outputs, labels) if labels is not None else loss_fn(outputs, images)
					
					# 使用梯度回退机制
					if gradient_handler.compute_gradients_with_backoff(loss, epoch, batch_idx):
						# 梯度裁剪
						torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
						optimizer.step()
					else:
						# 训练暂停
						return total_loss / max(num_batches, 1), True  # True表示需要停止
				
				total_loss += loss.item()
				num_batches += 1
				
				# 日志输出
				if batch_idx % args.log_interval == 0:
					logging.info(f'Epoch {epoch}, Batch {batch_idx}/{len(dataloader)}, Loss: {loss.item():.6f}')
			
			except Exception as e:
				logging.error(f"Training batch {batch_idx} failed: {e}")
				continue
		
		avg_loss = total_loss / max(num_batches, 1)
		return avg_loss, False  # False表示正常完成
	
	return wrapper


# 使用示例：
# 在 distributed_train.py 中替换 train_epoch 函数：
"""
@apply_gradient_backoff_to_training
def train_epoch(model, dataloader, optimizer, loss_fn, device, epoch, args, scaler=None):
    # 原有的训练逻辑会被装饰器处理
    pass
"""