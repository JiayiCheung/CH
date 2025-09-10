# utils/tensorboard_logger.py
import os
import time
import datetime
import torch
from torch.utils.tensorboard import SummaryWriter


class TensorBoardLogger:
	"""
	TensorBoard日志记录器，用于记录训练过程中的各种指标和可视化数据
	"""
	
	def __init__(self, log_dir, config=None, model=None, rank=0):
		"""
		初始化TensorBoard日志记录器

		参数:
			log_dir: 日志保存目录
			config: 训练配置字典
			model: 模型实例
			rank: 当前进程的rank（分布式训练中使用）
		"""
		self.log_dir = log_dir
		self.config = config
		self.model = model
		self.rank = rank
		self.start_time = time.time()
		
		# 只有rank=0的进程负责记录日志
		if self.rank == 0:
			os.makedirs(log_dir, exist_ok=True)
			self.writer = SummaryWriter(log_dir)
			print(f"TensorBoard日志保存在: {log_dir}")
			
			# 记录超参数与版本信息
			if config:
				self._log_hyperparameters()
			
			# 记录模型信息
			if model:
				self._log_model_info()
			
			# 记录开始时间
			start_time_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
			self.writer.add_text("Training/start_time", start_time_str, 0)
	
	def _log_hyperparameters(self):
		"""记录超参数信息"""
		hparam_text = ""
		
		# 训练相关参数
		if 'train' in self.config:
			hparam_text += "## 训练参数\n"
			train_config = self.config['train']
			hparam_text += f"- Batch Size: {train_config.get('batch_size', 'N/A')}\n"
			hparam_text += f"- Learning Rate: {train_config.get('lr', 'N/A')}\n"
			hparam_text += f"- Weight Decay: {train_config.get('weight_decay', 'N/A')}\n"
			hparam_text += f"- Epochs: {train_config.get('epochs', 'N/A')}\n"
		
		# 数据相关参数
		if 'data' in self.config:
			hparam_text += "\n## 数据参数\n"
			data_config = self.config['data']
			hparam_text += f"- Patch Size: {data_config.get('patch_size', 'N/A')}\n"
			hparam_text += f"- Samples Per Volume: {data_config.get('samples_per_volume', 'N/A')}\n"
			hparam_text += f"- Train-Val Split: {data_config.get('train_val_split', 'N/A')}\n"
		
		# 模型相关参数
		if 'model' in self.config:
			hparam_text += "\n## 模型参数\n"
			model_config = self.config['model']
			hparam_text += f"- Input Channels: {model_config.get('input_channels', 'N/A')}\n"
			hparam_text += f"- Output Classes: {model_config.get('output_classes', 'N/A')}\n"
		
		# 数据增强参数
		if 'aug' in self.config:
			hparam_text += "\n## 数据增强参数\n"
			aug_config = self.config['aug']
			for key, value in aug_config.items():
				hparam_text += f"- {key}: {value}\n"
		
		# 记录到TensorBoard
		self.writer.add_text("Hyperparameters", hparam_text, 0)
	
	def _log_model_info(self):
		"""记录模型信息"""
		total_params = sum(p.numel() for p in self.model.parameters())
		trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
		
		model_text = f"## 模型统计\n"
		model_text += f"- 总参数数量: {total_params:,}\n"
		model_text += f"- 可训练参数数量: {trainable_params:,}\n"
		model_text += f"- 参数占用内存: {total_params * 4 / (1024 ** 2):.2f} MB\n"
		
		self.writer.add_text("Model", model_text, 0)
	
	def log_epoch(self, epoch, loss, lr, metrics=None):
		"""
		记录每个epoch的训练信息

		参数:
			epoch: 当前epoch
			loss: 训练损失
			lr: 当前学习率
			metrics: 验证指标字典，包含dice、sensitivity、precision等
		"""
		if self.rank != 0:
			return
		
		# 损失曲线
		self.writer.add_scalar("Loss/train", loss, epoch)
		
		# 学习率记录
		self.writer.add_scalar("Training/learning_rate", lr, epoch)
		
		# 记录训练时间
		elapsed_time = time.time() - self.start_time
		self.writer.add_scalar("Training/elapsed_hours", elapsed_time / 3600, epoch)
		
		# 记录验证指标
		if metrics:
			for metric_name, value in metrics.items():
				self.writer.add_scalar(f"Metrics/{metric_name}", value, epoch)
	
	def log_weights_and_gradients(self, epoch, model):
		"""记录模型权重和梯度"""
		if self.rank != 0:
			return
		
		for name, param in model.named_parameters():
			if param.requires_grad:
				self.writer.add_histogram(f"Parameters/{name}", param.data, epoch)
				if param.grad is not None:
					self.writer.add_histogram(f"Gradients/{name}", param.grad, epoch)
	
	def log_validation_samples(self, epoch, inputs, labels, predictions):
		"""记录验证样本的可视化结果"""
		if self.rank != 0:
			return
		
		# 确保只取前几个样本，避免日志过大
		inputs_vis = inputs[:4].detach().cpu()
		labels_vis = labels[:4].detach().cpu()
		preds_vis = predictions[:4].detach().cpu()
		
		# 处理3D数据，取中心切片
		if inputs_vis.dim() == 5:  # 如果是3D数据(N,C,D,H,W)
			d_center = inputs_vis.shape[2] // 2
			inputs_vis = inputs_vis[:, :, d_center]
			labels_vis = labels_vis[:, :, d_center]
			preds_vis = preds_vis[:, :, d_center]
		
		# 记录输入、标签和预测
		self.writer.add_images("Samples/inputs", inputs_vis, epoch)
		self.writer.add_images("Samples/labels", labels_vis, epoch)
		self.writer.add_images("Samples/predictions", preds_vis, epoch)
		
		# 创建误差图
		error_maps = torch.zeros((preds_vis.shape[0], 3, *preds_vis.shape[2:]), device='cpu')
		error_maps[:, 0] = preds_vis[:, 0] * (1 - labels_vis[:, 0])  # 红色：假阳性
		error_maps[:, 1] = preds_vis[:, 0] * labels_vis[:, 0]  # 绿色：真阳性
		error_maps[:, 2] = (1 - preds_vis[:, 0]) * labels_vis[:, 0]  # 蓝色：假阴性
		self.writer.add_images("Samples/error_maps", error_maps, epoch)
	
	def log_best_model(self, epoch, metrics):
		"""记录最佳模型信息"""
		if self.rank != 0:
			return
		
		metrics_text = f"Epoch: {epoch}, "
		for name, value in metrics.items():
			metrics_text += f"{name}: {value:.4f}, "
		
		self.writer.add_text("Training/best_model", metrics_text, epoch)
	
	def log_scalar(self, tag, value, step):
		"""记录单个标量到 TensorBoard（如 loss, accuracy 等）"""
		if self.rank == 0:
			self.writer.add_scalar(tag, value, step)
	
	def finish_logging(self, best_metrics, final_metrics=None):
		"""完成日志记录，添加总结信息"""
		if self.rank != 0:
			return
		
		# 记录总训练时间
		total_time = time.time() - self.start_time
		time_str = f"{total_time // 3600}h {(total_time % 3600) // 60}m {total_time % 60:.2f}s"
		self.writer.add_text("Training/total_time", time_str, 0)
		
		# 记录最终结果
		results_text = f"## 训练结果\n"
		results_text += f"- 训练总时间: {time_str}\n"
		results_text += f"- 最佳指标:\n"
		
		for name, value in best_metrics.items():
			results_text += f"  - {name}: {value:.4f}\n"
		
		if final_metrics:
			results_text += f"- 最终指标:\n"
			for name, value in final_metrics.items():
				results_text += f"  - {name}: {value:.4f}\n"
		
		self.writer.add_text("Training/summary", results_text, 0)
		
		# 记录结束时间
		end_time_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
		self.writer.add_text("Training/end_time", end_time_str, 0)
		
		# 关闭writer
		self.writer.close()