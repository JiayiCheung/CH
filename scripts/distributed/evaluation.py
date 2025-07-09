import torch
import time
import numpy as np
from tqdm import tqdm
from collections import defaultdict
from pathlib import Path


class EvaluationManager:
	"""适配TA-CHNet分布式训练架构的评估管理器"""
	
	def __init__(self, config=None, logger=None):
		"""
		初始化评估管理器

		参数:
			config: 配置字典
			logger: 日志记录器实例（可选）
		"""
		# 默认配置
		self.default_config = {
			'eval_full_interval': 10,  # 完整评估间隔
			'eval_quick_interval': 2,  # 快速评估间隔
			'quick_samples': 5,  # 快速评估样本数量
			'group_by_tier': True,  # 按tier分组计算指标
			'feature_mmap_enabled': True,  # 是否使用特征内存映射
			'feature_mmap_dir': 'eval_tier_features',  # 特征存储目录
			'clear_cache_interval': 3  # 定期清理特征缓存间隔
		}
		
		# 使用提供的配置更新默认配置
		self.config = self.default_config.copy()
		if config:
			self.config.update(config)
		
		# 保存日志记录器
		self.logger = logger
		
		# 初始化MMap工具(沿用现有模式)
		if self.config['feature_mmap_enabled']:
			from data.mmap_utils import MMapManager
			self.mmap_manager = MMapManager
			
			# 确保目录存在
			mmap_dir = Path(self.config['feature_mmap_dir'])
			mmap_dir.mkdir(exist_ok=True, parents=True)
			lock_dir = mmap_dir / 'locks'
			lock_dir.mkdir(exist_ok=True, parents=True)
			
			self._log_info(f"Using memory mapping for tier features: {mmap_dir}")
		else:
			self.mmap_manager = None
		
		# 评估状态跟踪
		self.current_epoch = 0
		self.tier_metrics = defaultdict(dict)
		self.batch_count = 0
		self.active_tiers = set()
		
		self._log_info(f"Evaluation Manager initialized with config: {self.config}")
	
	def _log_info(self, message):
		"""使用logger记录信息，如果存在，否则使用print"""
		if self.logger:
			self.logger.log_info(message)
		else:
			print(message)
	
	def _log_warning(self, message):
		"""使用logger记录警告，如果存在，否则使用print"""
		if self.logger:
			self.logger.log_warning(message)
		else:
			print(f"WARNING: {message}")
	
	def _log_metrics(self, metrics, step, prefix=''):
		"""使用logger记录指标，如果存在"""
		if self.logger:
			self.logger.log_metrics(metrics, step, prefix)
	
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
		self.current_epoch = epoch
		
		# 根据epoch选择评估类型
		if epoch % self.config['eval_full_interval'] == 0:
			self._log_info(f"Epoch {epoch}: Running full evaluation...")
			return self.distributed_tier_evaluation(engine, val_loader, full_eval=True)
		elif epoch % self.config['eval_quick_interval'] == 0:
			self._log_info(f"Epoch {epoch}: Running quick evaluation...")
			return self.distributed_tier_evaluation(engine, val_loader, full_eval=False)
		
		return None
	
	def distributed_tier_evaluation(self, engine, val_loader, full_eval=True):
		"""
		适配分布式引擎的多Tier评估

		参数:
			engine: 分布式执行引擎
			val_loader: 验证数据加载器
			full_eval: 是否执行完整评估

		返回:
			评估指标字典
		"""
		# 设置为评估模式
		engine.eval()
		
		# 暂停流水线(这会让评估过程中不使用流水线并行)
		try:
			engine.pause_pipeline()
		except AttributeError:
			self._log_warning("Engine does not have pause_pipeline method. "
			                  "If using a standard model, evaluation will proceed normally.")
		
		try:
			# 清空缓存中的tier特征
			self._clear_backend_features(engine)
			self.active_tiers.clear()
			
			# 初始化指标累积器
			metrics_sum = defaultdict(float)
			tier_samples = defaultdict(int)
			total_samples = 0
			
			# 决定处理的样本数量
			max_samples = None if full_eval else self.config['quick_samples']
			
			# 创建进度条
			iterator = tqdm(val_loader, desc="Evaluating")
			
			# 记录开始时间
			start_time = time.time()
			
			# 遍历验证数据
			with torch.no_grad():
				for batch_idx, batch in enumerate(iterator):
					# 获取数据
					images = batch['image']
					labels = batch['label']
					tiers = batch['tier']
					
					# 限制样本数量(快速评估模式)
					if max_samples and total_samples >= max_samples:
						break
					
					# 定期清理特征缓存，避免GPU内存爆炸
					if batch_idx % self.config['clear_cache_interval'] == 0 and batch_idx > 0:
						self._clear_backend_features(engine)
						self.active_tiers.clear()
					
					# 处理每个样本
					for j, tier in enumerate(tiers):
						tier_int = int(tier)
						
						# 限制样本数量(快速评估模式)
						if max_samples and total_samples >= max_samples:
							break
						
						# 设置当前tier
						if hasattr(engine, 'set_tier'):
							engine.set_tier(tier_int)
						elif hasattr(engine, 'pipeline') and hasattr(engine.pipeline, 'set_tier'):
							engine.pipeline.set_tier(tier_int)
						
						# 检查是否需要从磁盘加载之前的tier特征
						if self.config['feature_mmap_enabled'] and tier_int in self.active_tiers:
							self._load_tier_feature(engine, tier_int)
						
						# 前向传播
						output = engine.forward(images[j:j + 1])
						
						# 如果是第一次处理此tier，将其添加到活动tier集合
						if tier_int not in self.active_tiers:
							self.active_tiers.add(tier_int)
							
							# 如果启用了内存映射，保存此tier特征到磁盘
							if self.config['feature_mmap_enabled']:
								self._save_tier_feature(engine, tier_int)
						
						# 计算指标
						sample_metrics = self._compute_metrics(output, labels[j:j + 1])
						
						# 按tier分组累积指标
						for k, v in sample_metrics.items():
							if self.config['group_by_tier']:
								if k not in self.tier_metrics[tier_int]:
									self.tier_metrics[tier_int][k] = 0.0
								self.tier_metrics[tier_int][k] += v
							
							# 同时累积全局指标
							metrics_sum[k] += v
						
						tier_samples[tier_int] += 1
						total_samples += 1
			
			# 计算评估时间
			eval_time = time.time() - start_time
			
			# 计算平均指标
			metrics = {}
			for k, v in metrics_sum.items():
				metrics[k] = v / max(1, total_samples)
			
			# 添加按tier分组的指标
			if self.config['group_by_tier']:
				for tier, tier_metrics in self.tier_metrics.items():
					if tier_samples[tier] > 0:
						for k, v in tier_metrics.items():
							metrics[f"tier{tier}_{k}"] = v / tier_samples[tier]
			
			# 添加评估时间
			metrics['eval_time'] = eval_time
			
			# 记录评估结果
			self._log_info(f"Evaluation completed in {eval_time:.2f}s")
			self._log_info(f"Dice score: {metrics.get('dice', 0):.4f}")
			
			# 记录到TensorBoard
			eval_type = "full_eval" if full_eval else "quick_eval"
			self._log_metrics(metrics, self.current_epoch, prefix=f'{eval_type}/')
			
			# 清理评估资源
			self._cleanup_evaluation_resources()
			
			return metrics
		
		finally:
			# 恢复流水线
			try:
				engine.resume_pipeline()
			except AttributeError:
				pass  # 如果没有此方法，忽略
			
			# 恢复训练模式
			engine.train()
	
	def _clear_backend_features(self, engine):
		"""清除后端阶段的tier特征缓存"""
		# 针对DistributedEngine的特定结构
		if hasattr(engine, 'pipeline') and hasattr(engine.pipeline, 'stages'):
			backend_stage = engine.pipeline.stages[3]  # 后端阶段在索引3
			if hasattr(backend_stage, 'tier_features'):
				backend_stage.tier_features.clear()
		elif hasattr(engine, 'tier_features'):
			engine.tier_features.clear()
		elif hasattr(engine, 'clear_tier_features'):
			engine.clear_tier_features()
		
		# 执行垃圾回收
		import gc
		gc.collect()
		torch.cuda.empty_cache()
	
	def _get_backend_features(self, engine, tier):
		"""从后端阶段获取指定tier的特征"""
		if hasattr(engine, 'pipeline') and hasattr(engine.pipeline, 'stages'):
			backend_stage = engine.pipeline.stages[3]
			if hasattr(backend_stage, 'tier_features') and tier in backend_stage.tier_features:
				return backend_stage.tier_features[tier]
		elif hasattr(engine, 'tier_features') and tier in engine.tier_features:
			return engine.tier_features[tier]
		return None
	
	def _set_backend_features(self, engine, tier, features):
		"""设置后端阶段的tier特征"""
		if hasattr(engine, 'pipeline') and hasattr(engine.pipeline, 'stages'):
			backend_stage = engine.pipeline.stages[3]
			if hasattr(backend_stage, 'tier_features'):
				backend_stage.tier_features[tier] = features
		elif hasattr(engine, 'tier_features'):
			engine.tier_features[tier] = features
	
	def _save_tier_feature(self, engine, tier):
		"""使用MMapManager保存tier特征到磁盘"""
		if not self.mmap_manager:
			return False
		
		# 获取特征
		features = self._get_backend_features(engine, tier)
		if features is None:
			return False
		
		# 转换为CPU NumPy数组
		if torch.is_tensor(features):
			features_np = features.cpu().numpy()
		else:
			features_np = features
		
		# 创建唯一键名
		key = f"tier{tier}_e{self.current_epoch}_b{self.batch_count}"
		
		# 使用MMapManager保存
		try:
			mmap_array = self.mmap_manager.create_or_load(
				self.config['feature_mmap_dir'],
				f"{self.config['feature_mmap_dir']}/locks",
				key,
				features_np.shape,
				features_np.dtype
			)
			
			if mmap_array is not None:
				mmap_array[:] = features_np
				self.mmap_manager.sync_to_disk(mmap_array)
				return True
		except Exception as e:
			self._log_warning(f"Error saving tier feature: {e}")
		
		return False
	
	def _load_tier_feature(self, engine, tier):
		"""从磁盘加载tier特征"""
		if not self.mmap_manager:
			return False
		
		# 创建唯一键名
		key = f"tier{tier}_e{self.current_epoch}_b{self.batch_count}"
		
		try:
			# 加载内存映射数组
			mmap_array = self.mmap_manager.create_or_load(
				self.config['feature_mmap_dir'],
				f"{self.config['feature_mmap_dir']}/locks",
				key,
				None,  # 形状由MMapManager从元数据获取
				None  # 类型由MMapManager从元数据获取
			)
			
			if mmap_array is not None:
				# 转换为PyTorch张量并加载到合适的GPU
				device = self._get_backend_device(engine)
				features = torch.tensor(mmap_array, device=device)
				
				# 设置特征
				self._set_backend_features(engine, tier, features)
				return True
		except Exception as e:
			self._log_warning(f"Error loading tier feature: {e}")
		
		return False
	
	def _get_backend_device(self, engine):
		"""获取后端阶段的设备"""
		if hasattr(engine, 'pipeline') and hasattr(engine.pipeline, 'stages'):
			backend_stage = engine.pipeline.stages[3]
			return backend_stage.device
		else:
			# 尝试从参数获取设备
			try:
				return next(engine.parameters()).device
			except:
				# 默认返回CUDA:0
				return torch.device('cuda:0')
	
	def _cleanup_evaluation_resources(self):
		"""清理评估过程中的临时资源"""
		if self.mmap_manager and self.config['feature_mmap_enabled']:
			# 删除所有临时特征文件
			for tier in self.active_tiers:
				key = f"tier{tier}_e{self.current_epoch}_b{self.batch_count}"
				try:
					self.mmap_manager.remove_mmap(
						self.config['feature_mmap_dir'],
						f"{self.config['feature_mmap_dir']}/locks",
						key
					)
				except Exception as e:
					self._log_warning(f"Error removing mmap file: {e}")
		
		# 清空特征缓存
		self.active_tiers.clear()
		self.tier_metrics.clear()
		
		# 强制执行垃圾回收
		import gc
		gc.collect()
		torch.cuda.empty_cache()
	
	def _compute_metrics(self, pred, target):
		"""计算评估指标"""
		from utils.metrics import SegmentationMetrics
		
		# 二值化预测
		binary_pred = (pred > 0.5).float()
		
		# 计算指标
		metrics = SegmentationMetrics.evaluate_all(binary_pred, target, include_advanced=False)
		
		return metrics