import os
import time
import logging
import numpy as np
from pathlib import Path
import torch
from torch.utils.tensorboard import SummaryWriter


class Logger:
	"""日志记录器，支持文本日志和TensorBoard"""
	
	def __init__(self, log_dir, experiment_name=None):
		"""
		初始化日志记录器

		参数:
			log_dir: 日志目录
			experiment_name: 实验名称，如果为None则使用时间戳
		"""
		# 创建日志目录
		self.log_dir = Path(log_dir)
		self.log_dir.mkdir(exist_ok=True, parents=True)
		
		# 实验名称
		if experiment_name is None:
			experiment_name = time.strftime("%Y%m%d-%H%M%S")
		self.experiment_name = experiment_name
		
		# 创建实验目录
		self.experiment_dir = self.log_dir / experiment_name
		self.experiment_dir.mkdir(exist_ok=True)
		
		# 设置文件日志
		self._setup_file_logger()
		
		# 设置TensorBoard
		self.tb_writer = SummaryWriter(log_dir=str(self.experiment_dir / 'tensorboard'))
	
	def _setup_file_logger(self):
		"""设置文件日志记录器"""
		# 创建logger
		self.logger = logging.getLogger(self.experiment_name)
		self.logger.setLevel(logging.INFO)
		
		# 创建文件处理器
		fh = logging.FileHandler(str(self.experiment_dir / 'experiment.log'))
		fh.setLevel(logging.INFO)
		
		# 创建控制台处理器
		ch = logging.StreamHandler()
		ch.setLevel(logging.INFO)
		
		# 创建格式器
		formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
		fh.setFormatter(formatter)
		ch.setFormatter(formatter)
		
		# 添加处理器
		self.logger.addHandler(fh)
		self.logger.addHandler(ch)
	
	def log_info(self, message):
		"""记录信息级别的日志"""
		self.logger.info(message)
	
	def log_warning(self, message):
		"""记录警告级别的日志"""
		self.logger.warning(message)
	
	def log_error(self, message):
		"""记录错误级别的日志"""
		self.logger.error(message)
	
	def log_metrics(self, metrics, step, prefix=''):
		"""
		记录指标到TensorBoard

		参数:
			metrics: 指标字典
			step: 训练步数
			prefix: 指标前缀
		"""
		for name, value in metrics.items():
			if isinstance(value, (int, float, np.number)):
				self.tb_writer.add_scalar(f'{prefix}{name}', value, step)
			elif torch.is_tensor(value) and value.numel() == 1:
				self.tb_writer.add_scalar(f'{prefix}{name}', value.item(), step)
			else:
				self.log_warning(f"Skipping logging of {name} because it's not a scalar")
	
	def log_parameters(self, model, step):
		"""
		记录模型参数到TensorBoard

		参数:
			model: PyTorch模型
			step: 训练步数
		"""
		for name, param in model.named_parameters():
			self.tb_writer.add_histogram(name, param.clone().cpu().data.numpy(), step)
	
	def log_images(self, images_dict, step):
		"""
		记录图像到TensorBoard

		参数:
			images_dict: 图像字典 {name: image_tensor}
			step: 训练步数
		"""
		for name, image in images_dict.items():
			if torch.is_tensor(image):
				image = image.clone().cpu().data.numpy()
			
			# 如果是3D图像，取中间切片
			if len(image.shape) == 3:
				image = image[image.shape[0] // 2]
			
			# 如果是灰度图像，添加通道维度
			if len(image.shape) == 2:
				image = image[np.newaxis, ...]
			
			# 标准化图像
			if np.max(image) > 1.0:
				image = image / 255.0
			
			self.tb_writer.add_image(name, image, step)
	
	def log_model(self, model, filename='model.pt'):
		"""
		保存模型

		参数:
			model: PyTorch模型
			filename: 文件名
		"""
		torch.save(model.state_dict(), str(self.experiment_dir / filename))
	
	def close(self):
		"""关闭日志记录器"""
		self.tb_writer.close()
		
		# 移除处理器
		for handler in self.logger.handlers[:]:
			handler.close()
			self.logger.removeHandler(handler)