import torch
import time
import numpy as np
from tqdm import tqdm


class EvaluationManager:
	"""
	评估管理器 - 管理模型评估流程
	"""
	
	def __init__(self, config=None):
		"""
		初始化评估管理器

		参数:
			config: 评估配置
		"""
		# 默认配置
		self.default_config = {
			'eval_full_interval': 10,  # 完整评估间隔
			'eval_quick_interval': 2,  # 快速评估间隔
			'quick_samples': 5  # 快速评估样本数量
		}
		
		# 使用提供的配置更新默认配置
		self.config = self.default_config.copy()
		if config:
			self.config.update(config)
		
		print(f"Evaluation Manager initialized: full_interval={self.config['eval_full_interval']}, "
		      f"quick_interval={self.config['eval_quick_interval']}")
	
	def should_evaluate(self, epoch):
		"""
		判断是否应该在当前epoch进行评估

		参数:
			epoch: 当前epoch

		返回:
			布尔值，指示是否应该评估
		"""
		return (epoch % self.config['eval_full_interval'] == 0 or
		        epoch % self.config['eval_quick_interval'] == 0)
	
	def should_evaluate_full(self, epoch):
		"""
		判断是否进行完整评估

		参数:
			epoch: 当前epoch

		返回:
			布尔值
		"""
		return epoch % self.config['eval_full_interval'] == 0
	
	def should_evaluate_quick(self, epoch):
		"""
		判断是否进行快速评估

		参数:
			epoch: 当前epoch

		返回:
			布尔值
		"""
		# 优先级：如果同时满足完整和快速评估条件，执行完整评估
		if self.should_evaluate_full(epoch):
			return False
		
		return epoch % self.config['eval_quick_interval'] == 0
	
	def evaluate(self, engine, val_loader, epoch):
		"""
		执行评估

		参数:
			engine: 分布式执行引擎
			val_loader: 验证数据加载器
			epoch: 当前epoch

		返回:
			评估指标字典
		"""
		# 根据epoch选择评估类型
		if self.should_evaluate_full(epoch):
			print(f"Epoch {epoch}: Running full evaluation...")
			return self.full_evaluation(engine, val_loader)
		elif self.should_evaluate_quick(epoch):
			print(f"Epoch {epoch}: Running quick evaluation...")
			return self.quick_evaluation(engine, val_loader)
		
		return None
	
	def full_evaluation(self, engine, val_loader):
		"""
		完整评估

		参数:
			engine: 分布式执行引擎
			val_loader: 验证数据加载器

		返回:
			评估指标字典
		"""
		# 设置为评估模式
		engine.eval()
		
		# 暂停流水线
		engine.pause_pipeline()
		
		try:
			# 初始化指标累积器
			metrics_sum = {}
			total_samples = 0
			
			# 使用tqdm创建进度条
			iterator = tqdm(val_loader, desc="Evaluating")
			
			# 遍历验证数据
			with torch.no_grad():
				for batch in iterator:
					# 获取数据
					images = batch['image']
					labels = batch['label']
					tiers = batch['tier']
					
					# 处理每个样本
					for j, tier in enumerate(tiers):
						# 设置当前tier
						engine.pipeline.set_tier(int(tier))
						
						# 前向传播
						output = engine.forward(images[j:j + 1])
						
						# 计算指标
						sample_metrics = self._compute_metrics(output, labels[j:j + 1])
						
						# 累加指标
						for k, v in sample_metrics.items():
							metrics_sum[k] = metrics_sum.get(k, 0) + v
						
						total_samples += 1
			
			# 计算平均指标
			metrics = {}
			for k, v in metrics_sum.items():
				metrics[k] = v / total_samples
			
			return metrics
		
		finally:
			# 恢复流水线
			engine.resume_pipeline()
			
			# 恢复训练模式
			engine.train()
	
	def quick_evaluation(self, engine, val_loader):
		"""
		快速评估(使用少量样本)

		参数:
			engine: 分布式执行引擎
			val_loader: 验证数据加载器

		返回:
			评估指标字典
		"""
		# 设置为评估模式
		engine.eval()
		
		# 暂停流水线
		engine.pause_pipeline()
		
		try:
			# 初始化指标累积器
			metrics_sum = {}
			total_samples = 0
			
			# 遍历有限数量的样本
			with torch.no_grad():
				for i, batch in enumerate(val_loader):
					if i >= self.config['quick_samples']:
						break
					
					# 获取数据
					images = batch['image']
					labels = batch['label']
					tiers = batch['tier']
					
					# 处理每个样本
					for j, tier in enumerate(tiers):
						# 设置当前tier
						engine.pipeline.set_tier(int(tier))
						
						# 前向传播
						output = engine.forward(images[j:j + 1])
						
						# 计算指标
						sample_metrics = self._compute_metrics(output, labels[j:j + 1])
						
						# 累加指标
						for k, v in sample_metrics.items():
							metrics_sum[k] = metrics_sum.get(k, 0) + v
						
						total_samples += 1
						
						# 限制样本数量
						if total_samples >= self.config['quick_samples']:
							break
			
			# 计算平均指标
			metrics = {}
			for k, v in metrics_sum.items():
				metrics[k] = v / max(total_samples, 1)
			
			return metrics
		
		finally:
			# 恢复流水线
			engine.resume_pipeline()
			
			# 恢复训练模式
			engine.train()
	
	def _compute_metrics(self, pred, target):
		"""
		计算评估指标

		参数:
			pred: 预测
			target: 目标

		返回:
			指标字典
		"""
		from utils.metrics import SegmentationMetrics
		
		# 二值化预测
		binary_pred = (pred > 0.5).float()
		
		# 计算指标
		metrics = SegmentationMetrics.evaluate_all(binary_pred, target)
		
		return metrics