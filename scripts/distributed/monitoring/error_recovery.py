#!/usr/bin/env python3
"""
错误恢复机制
简化监控 + 实时日志输出，专注问题定位
"""

import torch
import psutil
from typing import Tuple
import time
import logging
import threading
import subprocess
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass
from collections import deque
import json
import os


@dataclass
class HealthMetrics:
	"""健康指标"""
	# GPU指标
	gpu_temperature: float
	gpu_memory_used: float
	gpu_memory_total: float
	gpu_utilization: float
	
	# 系统指标
	cpu_usage: float
	ram_usage: float
	
	# 网络指标
	network_latency: Optional[float]
	
	# 时间戳
	timestamp: float
	
	def to_dict(self) -> Dict[str, Any]:
		"""转换为字典"""
		return {
			'gpu_temperature': self.gpu_temperature,
			'gpu_memory_used': self.gpu_memory_used,
			'gpu_memory_total': self.gpu_memory_total,
			'gpu_utilization': self.gpu_utilization,
			'gpu_memory_usage_percent': (
						self.gpu_memory_used / self.gpu_memory_total * 100) if self.gpu_memory_total > 0 else 0,
			'cpu_usage': self.cpu_usage,
			'ram_usage': self.ram_usage,
			'network_latency': self.network_latency,
			'timestamp': self.timestamp
		}


@dataclass
class TrainingMetrics:
	"""训练指标"""
	current_loss: float
	loss_change_rate: float
	gradient_norm: float
	parameter_update_norm: float
	validation_accuracy: Optional[float]
	learning_rate: float
	epoch: int
	batch_idx: int
	timestamp: float
	
	def to_dict(self) -> Dict[str, Any]:
		"""转换为字典"""
		return {
			'current_loss': self.current_loss,
			'loss_change_rate': self.loss_change_rate,
			'gradient_norm': self.gradient_norm,
			'parameter_update_norm': self.parameter_update_norm,
			'validation_accuracy': self.validation_accuracy,
			'learning_rate': self.learning_rate,
			'epoch': self.epoch,
			'batch_idx': self.batch_idx,
			'timestamp': self.timestamp
		}


class HealthMonitor:
	"""系统健康监控器"""
	
	def __init__(self, rank: int, device: torch.device):
		self.rank = rank
		self.device = device
		
		# 历史指标缓存
		self.health_history = deque(maxlen=100)
		self.monitoring_active = False
		self.monitor_thread = None
		self.monitor_interval = 5.0  # 5秒监控间隔
		
		self.logger = logging.getLogger(__name__)
	
	def start_monitoring(self):
		"""启动健康监控"""
		if self.monitoring_active:
			return
		
		self.monitoring_active = True
		self.monitor_thread = threading.Thread(
			target=self._monitoring_loop,
			name=f"HealthMonitor-{self.rank}",
			daemon=True
		)
		self.monitor_thread.start()
		
		self.logger.info(f"健康监控已启动: rank {self.rank}")
	
	def stop_monitoring(self):
		"""停止健康监控"""
		self.monitoring_active = False
		
		if self.monitor_thread and self.monitor_thread.is_alive():
			self.monitor_thread.join(timeout=5.0)
		
		self.logger.info(f"健康监控已停止: rank {self.rank}")
	
	def _monitoring_loop(self):
		"""监控循环"""
		while self.monitoring_active:
			try:
				metrics = self._collect_health_metrics()
				if metrics:
					self.health_history.append(metrics)
					self._log_health_status(metrics)
					self._check_health_alerts(metrics)
				
				time.sleep(self.monitor_interval)
			
			except Exception as e:
				self.logger.error(f"健康监控异常: {e}")
				time.sleep(self.monitor_interval)
	
	def _collect_health_metrics(self) -> Optional[HealthMetrics]:
		"""收集健康指标"""
		try:
			# GPU指标
			gpu_temp = self._get_gpu_temperature()
			gpu_memory = self._get_gpu_memory()
			gpu_util = self._get_gpu_utilization()
			
			# 系统指标
			cpu_usage = psutil.cpu_percent(interval=0.1)
			ram_usage = psutil.virtual_memory().percent
			
			# 网络延迟（简化测试）
			network_latency = self._test_network_latency()
			
			return HealthMetrics(
				gpu_temperature=gpu_temp,
				gpu_memory_used=gpu_memory[0],
				gpu_memory_total=gpu_memory[1],
				gpu_utilization=gpu_util,
				cpu_usage=cpu_usage,
				ram_usage=ram_usage,
				network_latency=network_latency,
				timestamp=time.time()
			)
		
		except Exception as e:
			self.logger.error(f"收集健康指标失败: {e}")
			return None
	
	def _get_gpu_temperature(self) -> float:
		"""获取GPU温度"""
		try:
			if torch.cuda.is_available():
				# 使用nvidia-smi获取温度
				result = subprocess.run(
					['nvidia-smi', '--query-gpu=temperature.gpu', '--format=csv,noheader,nounits'],
					capture_output=True, text=True, timeout=5
				)
				if result.returncode == 0:
					temps = result.stdout.strip().split('\n')
					if len(temps) > self.device.index:
						return float(temps[self.device.index])
			return 0.0
		except Exception:
			return 0.0
	
	def _get_gpu_memory(self) -> Tuple[float, float]:
		"""获取GPU内存使用情况 (使用量, 总量) MB"""
		try:
			if torch.cuda.is_available():
				used = torch.cuda.memory_allocated(self.device) / 1024 ** 2
				total = torch.cuda.max_memory_allocated(self.device) / 1024 ** 2
				if total == 0:
					total = torch.cuda.get_device_properties(self.device).total_memory / 1024 ** 2
				return used, total
			return 0.0, 0.0
		except Exception:
			return 0.0, 0.0
	
	def _get_gpu_utilization(self) -> float:
		"""获取GPU利用率"""
		try:
			if torch.cuda.is_available():
				result = subprocess.run(
					['nvidia-smi', '--query-gpu=utilization.gpu', '--format=csv,noheader,nounits'],
					capture_output=True, text=True, timeout=5
				)
				if result.returncode == 0:
					utils = result.stdout.strip().split('\n')
					if len(utils) > self.device.index:
						return float(utils[self.device.index])
			return 0.0
		except Exception:
			return 0.0
	
	def _test_network_latency(self) -> Optional[float]:
		"""测试网络延迟（简化版）"""
		try:
			# 简单的本地回环测试
			start_time = time.time()
			result = subprocess.run(['ping', '-c', '1', 'localhost'],
			                        capture_output=True, timeout=3)
			if result.returncode == 0:
				return (time.time() - start_time) * 1000  # 毫秒
			return None
		except Exception:
			return None
	
	def _log_health_status(self, metrics: HealthMetrics):
		"""记录健康状态日志"""
		#self.logger.info(
			#f"[HEALTH] Rank {self.rank} | "
			#f"GPU: {metrics.gpu_temperature:.1f}°C, "
			#f"Mem: {metrics.gpu_memory_used:.0f}/{metrics.gpu_memory_total:.0f}MB "
			#f"({metrics.gpu_memory_used / metrics.gpu_memory_total * 100:.1f}%), "
			#f"Util: {metrics.gpu_utilization:.1f}% | "
			#f"CPU: {metrics.cpu_usage:.1f}%, "
			#f"RAM: {metrics.ram_usage:.1f}% | "
			#f"Net: {metrics.network_latency:.1f}ms" if metrics.network_latency else "Net: N/A")
	
		pass
	
	
	def _check_health_alerts(self, metrics: HealthMetrics):
		"""检查健康警报"""
		alerts = []
		
		# GPU温度警报
		if metrics.gpu_temperature > 85:
			alerts.append(f"GPU温度过高: {metrics.gpu_temperature:.1f}°C")
		
		# GPU内存警报
		memory_usage = metrics.gpu_memory_used / metrics.gpu_memory_total * 100
		
		
		# CPU使用率警报
		if metrics.cpu_usage > 90:
			alerts.append(f"CPU使用率过高: {metrics.cpu_usage:.1f}%")
		
		# RAM使用率警报
		if metrics.ram_usage > 90:
			alerts.append(f"RAM使用率过高: {metrics.ram_usage:.1f}%")
		
		# 网络延迟警报
		if metrics.network_latency and metrics.network_latency > 1000:
			alerts.append(f"网络延迟过高: {metrics.network_latency:.1f}ms")
		
		# 输出警报
		for alert in alerts:
			self.logger.warning(f"[ALERT] {alert}")
	
	def get_latest_metrics(self) -> Optional[HealthMetrics]:
		"""获取最新的健康指标"""
		return self.health_history[-1] if self.health_history else None
	
	def get_metrics_history(self, minutes: int = 10) -> List[HealthMetrics]:
		"""获取指定时间内的指标历史"""
		cutoff_time = time.time() - (minutes * 60)
		return [m for m in self.health_history if m.timestamp >= cutoff_time]


class TrainingMonitor:
	"""训练过程监控器"""
	
	def __init__(self, rank: int):
		self.rank = rank
		
		# 训练指标历史
		self.training_history = deque(maxlen=1000)
		self.loss_history = deque(maxlen=100)
		
		self.logger = logging.getLogger(__name__)
	
	def record_training_metrics(self,
	                            loss: float,
	                            gradient_norm: float,
	                            param_update_norm: float,
	                            learning_rate: float,
	                            epoch: int,
	                            batch_idx: int,
	                            validation_accuracy: Optional[float] = None):
		"""记录训练指标"""
		try:
			# 计算损失变化率
			loss_change_rate = 0.0
			if len(self.loss_history) > 0:
				prev_loss = self.loss_history[-1]
				loss_change_rate = (loss - prev_loss) / prev_loss if prev_loss != 0 else 0.0
			
			self.loss_history.append(loss)
			
			# 创建训练指标
			metrics = TrainingMetrics(
				current_loss=loss,
				loss_change_rate=loss_change_rate,
				gradient_norm=gradient_norm,
				parameter_update_norm=param_update_norm,
				validation_accuracy=validation_accuracy,
				learning_rate=learning_rate,
				epoch=epoch,
				batch_idx=batch_idx,
				timestamp=time.time()
			)
			
			self.training_history.append(metrics)
			
			# 记录日志
			self._log_training_status(metrics)
			
			# 检查训练异常
			self._check_training_alerts(metrics)
		
		except Exception as e:
			self.logger.error(f"记录训练指标失败: {e}")
	
	def _log_training_status(self, metrics: TrainingMetrics):
		"""记录训练状态日志"""
		self.logger.info(
			f"[TRAINING] Rank {self.rank} | "
			f"Epoch {metrics.epoch}, Batch {metrics.batch_idx} | "
			f"Loss: {metrics.current_loss:.6f} "
			f"(Δ: {metrics.loss_change_rate:+.2%}), "
			f"Grad: {metrics.gradient_norm:.3f}, "
			f"Update: {metrics.parameter_update_norm:.3f}, "
			f"LR: {metrics.learning_rate:.2e}"
			+ (f", Val Acc: {metrics.validation_accuracy:.3f}" if metrics.validation_accuracy else "")
		)
	
	def _check_training_alerts(self, metrics: TrainingMetrics):
		"""检查训练异常"""
		alerts = []
		
		# 损失爆炸
		if metrics.current_loss > 1000 or not torch.isfinite(torch.tensor(metrics.current_loss)):
			alerts.append(f"损失爆炸: {metrics.current_loss}")
		
		# 梯度爆炸
		if metrics.gradient_norm > 100:
			alerts.append(f"梯度爆炸: {metrics.gradient_norm:.3f}")
		
		# 梯度消失
		if metrics.gradient_norm < 1e-8:
			alerts.append(f"梯度消失: {metrics.gradient_norm:.3e}")
		
		# 参数更新过大
		if metrics.parameter_update_norm > 10:
			alerts.append(f"参数更新过大: {metrics.parameter_update_norm:.3f}")
		
		# 损失不收敛（连续增长）
		if len(self.loss_history) >= 10:
			recent_losses = list(self.loss_history)[-10:]
			if all(recent_losses[i] <= recent_losses[i + 1] for i in range(9)):
				alerts.append("损失连续增长，可能不收敛")
		
		# 输出警报
		for alert in alerts:
			self.logger.error(f"[TRAINING ALERT] {alert}")
	
	def get_training_summary(self) -> Dict[str, Any]:
		"""获取训练摘要"""
		if not self.training_history:
			return {}
		
		latest = self.training_history[-1]
		recent_losses = [m.current_loss for m in list(self.training_history)[-20:]]
		
		return {
			'latest_loss': latest.current_loss,
			'latest_gradient_norm': latest.gradient_norm,
			'latest_lr': latest.learning_rate,
			'epoch': latest.epoch,
			'batch_idx': latest.batch_idx,
			'avg_recent_loss': sum(recent_losses) / len(recent_losses),
			'loss_trend': recent_losses[-1] - recent_losses[0] if len(recent_losses) > 1 else 0,
			'total_batches_recorded': len(self.training_history)
		}


class ErrorRecoverySystem:
	"""错误恢复系统 - 简化监控版"""
	
	def __init__(self, rank: int, device: torch.device):
		self.rank = rank
		self.device = device
		
		# 监控组件
		self.health_monitor = HealthMonitor(rank, device)
		self.training_monitor = TrainingMonitor(rank)
		
		# 系统状态
		self.system_healthy = True
		self.last_status_report = time.time()
		self.status_report_interval = 60.0  # 60秒状态报告间隔
		
		self.logger = logging.getLogger(__name__)
	
	def start_monitoring(self):
		"""启动监控系统"""
		self.health_monitor.start_monitoring()
		self.logger.info(f"错误恢复系统已启动: rank {self.rank}")
	
	def stop_monitoring(self):
		"""停止监控系统"""
		self.health_monitor.stop_monitoring()
		self.logger.info(f"错误恢复系统已停止: rank {self.rank}")
	
	def record_batch_metrics(self,
	                         loss: float,
	                         model,
	                         optimizer,
	                         epoch: int,
	                         batch_idx: int,
	                         validation_accuracy: Optional[float] = None):
		"""记录批次指标"""
		try:
			# 计算梯度范数
			gradient_norm = 0.0
			if model:
				total_norm = 0.0
				for p in model.parameters():
					if p.grad is not None:
						param_norm = p.grad.data.norm(2)
						total_norm += param_norm.item() ** 2
				gradient_norm = total_norm ** (1. / 2)
			
			# 计算参数更新范数（简化）
			param_update_norm = 0.0
			if optimizer:
				# 简化：使用学习率作为代理
				for group in optimizer.param_groups:
					param_update_norm = group.get('lr', 0.0)
					break
			
			# 获取学习率
			learning_rate = 0.0
			if optimizer:
				for group in optimizer.param_groups:
					learning_rate = group.get('lr', 0.0)
					break
			
			# 记录训练指标
			self.training_monitor.record_training_metrics(
				loss=loss,
				gradient_norm=gradient_norm,
				param_update_norm=param_update_norm,
				learning_rate=learning_rate,
				epoch=epoch,
				batch_idx=batch_idx,
				validation_accuracy=validation_accuracy
			)
			
			# 定期状态报告
			if time.time() - self.last_status_report > self.status_report_interval:
				self._generate_status_report()
				self.last_status_report = time.time()
		
		except Exception as e:
			self.logger.error(f"记录批次指标失败: {e}")
	
	def _generate_status_report(self):
		"""生成状态报告"""
		try:
			# 健康指标
			health_metrics = self.health_monitor.get_latest_metrics()
			
			# 训练摘要
			training_summary = self.training_monitor.get_training_summary()
			
			# 输出状态报告
			self.logger.info("=" * 80)
			self.logger.info(f"状态报告 - Rank {self.rank}")
			self.logger.info("=" * 80)
			
			if health_metrics:
				self.logger.info(f"系统健康状态:")
				health_dict = health_metrics.to_dict()
				for key, value in health_dict.items():
					if isinstance(value, float):
						self.logger.info(f"  {key}: {value:.2f}")
					else:
						self.logger.info(f"  {key}: {value}")
			
			if training_summary:
				self.logger.info(f"训练状态摘要:")
				for key, value in training_summary.items():
					if isinstance(value, float):
						self.logger.info(f"  {key}: {value:.6f}")
					else:
						self.logger.info(f"  {key}: {value}")
			
			self.logger.info("=" * 80)
		
		except Exception as e:
			self.logger.error(f"生成状态报告失败: {e}")
	
	def check_system_health(self) -> bool:
		"""检查系统健康状态"""
		try:
			health_metrics = self.health_monitor.get_latest_metrics()
			if not health_metrics:
				return True  # 无数据时假设健康
			
			# 简单的健康检查
			critical_issues = []
			
			# GPU温度检查
			if health_metrics.gpu_temperature > 90:
				critical_issues.append("GPU过热")
			
			# 内存检查
			memory_usage = health_metrics.gpu_memory_used / health_metrics.gpu_memory_total * 100
			if memory_usage > 98:
				critical_issues.append("GPU内存耗尽")
			
			# CPU检查
			if health_metrics.cpu_usage > 95:
				critical_issues.append("CPU负载过高")
			
			if critical_issues:
				self.logger.error(f"系统健康检查失败: {critical_issues}")
				self.system_healthy = False
				return False
			
			self.system_healthy = True
			return True
		
		except Exception as e:
			self.logger.error(f"健康检查异常: {e}")
			return True  # 异常时假设健康，避免误报
	
	def export_monitoring_data(self, output_dir: str = "./monitoring_logs"):
		"""导出监控数据"""
		try:
			os.makedirs(output_dir, exist_ok=True)
			
			# 导出健康指标
			health_data = []
			for metrics in self.health_monitor.health_history:
				health_data.append(metrics.to_dict())
			
			health_file = f"{output_dir}/health_metrics_rank_{self.rank}_{int(time.time())}.json"
			with open(health_file, 'w') as f:
				json.dump(health_data, f, indent=2)
			
			# 导出训练指标
			training_data = []
			for metrics in self.training_monitor.training_history:
				training_data.append(metrics.to_dict())
			
			training_file = f"{output_dir}/training_metrics_rank_{self.rank}_{int(time.time())}.json"
			with open(training_file, 'w') as f:
				json.dump(training_data, f, indent=2)
			
			self.logger.info(f"监控数据已导出: {health_file}, {training_file}")
		
		except Exception as e:
			self.logger.error(f"导出监控数据失败: {e}")


# 集成到训练循环的示例
def integrate_error_recovery():
	"""集成错误恢复系统的示例代码"""
	example_code = """
def train_epoch(model, dataloader, optimizer, loss_fn, device, epoch, args, scaler=None):
    # 创建错误恢复系统
    error_recovery = ErrorRecoverySystem(rank, device)
    error_recovery.start_monitoring()

    model.train()
    total_loss = 0.0
    num_batches = 0

    try:
        for batch_idx, batch in enumerate(dataloader):
            try:
                # 检查系统健康状态
                if not error_recovery.check_system_health():
                    logger.warning("系统健康检查失败，但继续训练")

                # ... 原有训练代码 ...

                # 记录批次指标
                error_recovery.record_batch_metrics(
                    loss=loss.item(),
                    model=model,
                    optimizer=optimizer,
                    epoch=epoch,
                    batch_idx=batch_idx
                )

                total_loss += loss.item()
                num_batches += 1

                # 详细的批次日志
                if batch_idx % args.log_interval == 0:
                    logger.info(
                        f'详细训练状态 - Epoch {epoch}, Batch {batch_idx}/{len(dataloader)}, '
                        f'Loss: {loss.item():.6f}, '
                        f'LR: {optimizer.param_groups[0]["lr"]:.2e}, '
                        f'Memory: {torch.cuda.memory_allocated(device)/1024**2:.0f}MB'
                    )

            except Exception as e:
                logger.error(f"Training batch {batch_idx} failed: {e}")

                # 记录失败信息
                error_recovery.record_batch_metrics(
                    loss=float('inf'),
                    model=model,
                    optimizer=optimizer,
                    epoch=epoch,
                    batch_idx=batch_idx
                )
                continue

        # 导出监控数据
        if epoch % 10 == 0:  # 每10个epoch导出一次
            error_recovery.export_monitoring_data()

    finally:
        # 确保停止监控
        error_recovery.stop_monitoring()

    avg_loss = total_loss / max(num_batches, 1)
    return avg_loss
    """
	return example_code


if __name__ == "__main__":
	print("错误恢复机制代码已生成")
	print("主要特性：")
	print("- 系统健康监控（GPU温度、内存、CPU、网络）")
	print("- 训练指标监控（损失、梯度、参数更新）")
	print("- 实时日志输出和警报")
	print("- 异常检测（损失爆炸、梯度异常等）")
	print("- 监控数据导出")
	print("- 简化设计专注问题定位")
	print("\n集成示例：")
	print(integrate_error_recovery())