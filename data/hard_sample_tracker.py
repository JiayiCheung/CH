import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
import json
import os
import threading
import time


class HardSampleTracker:
	"""硬样本跟踪器，管理和更新难度图"""
	
	def __init__(self, base_dir="difficulty_maps", alpha=0.7, device='cpu', logger=None):
		"""
		初始化硬样本跟踪器

		参数:
			base_dir: 存储难度图的目录
			alpha: 历史信息权重 (0-1)
			device: 计算设备
			logger: 日志记录器实例
		"""
		self.base_dir = Path(base_dir)
		self.alpha = alpha
		self.device = device
		self.logger = logger
		self.current_epoch = 0
		
		# 确保目录存在
		self.base_dir.mkdir(exist_ok=True, parents=True)
		
		# 加载案例维度信息
		self.case_dims = self._load_metadata()
		
		# 更新队列与处理线程
		self.update_queue = []
		self.queue_lock = threading.Lock()
		self.is_processing = False
		
		# 内存缓存
		self.difficulty_cache = {}
		self.cache_access_times = {}
		self.max_cache_size = 5  # 最多缓存5个案例
		
		if self.logger:
			self.logger.log_info(f"初始化硬样本跟踪器: {base_dir}")
	
	def _load_metadata(self):
		"""从元数据文件加载案例维度信息"""
		metadata_path = self.base_dir / "difficulty_metadata.json"
		if metadata_path.exists():
			try:
				with open(metadata_path, 'r') as f:
					return json.load(f)
			except Exception as e:
				if self.logger:
					self.logger.log_warning(f"加载元数据错误: {e}")
				return {}
		return {}
	
	def _save_metadata(self):
		"""保存案例维度信息到元数据文件"""
		metadata_path = self.base_dir / "difficulty_metadata.json"
		try:
			# 直接写入文件
			with open(metadata_path, 'w') as f:
				json.dump(self.case_dims, f)
		except Exception as e:
			if self.logger:
				self.logger.log_warning(f"保存元数据错误: {e}")
	
	def initialize_case(self, case_id, shape):
		"""
		初始化案例的难度图

		参数:
			case_id: 案例ID
			shape: 数据形状 [D, H, W]

		返回:
			是否成功初始化
		"""
		# 记录维度信息
		self.case_dims[case_id] = list(shape)
		self._save_metadata()
		
		# 创建难度图文件
		difficulty_path = self.base_dir / f"{case_id}.npy"
		
		if difficulty_path.exists():
			# 文件已存在，返回已加载的难度图
			return True
		
		try:
			# 创建填充值为0.5的难度图
			difficulty_map = np.full(shape, 0.5, dtype=np.float32)
			np.save(difficulty_path, difficulty_map)
			
			# 添加到缓存
			self._update_cache(case_id, difficulty_map)
			
			if self.logger:
				self.logger.log_info(f"初始化案例 {case_id} 难度图, 形状: {shape}")
			
			return True
		except Exception as e:
			if self.logger:
				self.logger.log_warning(f"创建难度图错误: {e}")
			return False
	
	def get_difficulty_map(self, case_id):
		"""
		获取案例的难度图

		参数:
			case_id: 案例ID

		返回:
			难度图数组
		"""
		if case_id not in self.case_dims:
			if self.logger:
				self.logger.log_warning(f"案例 {case_id} 未初始化")
			return None
		
		# 检查缓存
		if case_id in self.difficulty_cache:
			# 更新访问时间
			self.cache_access_times[case_id] = time.time()
			return self.difficulty_cache[case_id]
		
		# 从文件加载
		difficulty_path = self.base_dir / f"{case_id}.npy"
		
		if not difficulty_path.exists():
			# 文件不存在，创建默认难度图
			shape = tuple(self.case_dims[case_id])
			difficulty_map = np.full(shape, 0.5, dtype=np.float32)
			np.save(difficulty_path, difficulty_map)
		else:
			try:
				# 加载难度图
				difficulty_map = np.load(difficulty_path)
			except Exception as e:
				if self.logger:
					self.logger.log_warning(f"加载难度图错误: {e}")
				shape = tuple(self.case_dims[case_id])
				difficulty_map = np.full(shape, 0.5, dtype=np.float32)
		
		# 添加到缓存
		self._update_cache(case_id, difficulty_map)
		
		return difficulty_map
	
	def _update_cache(self, case_id, difficulty_map):
		"""更新内存缓存，淘汰最旧访问的条目"""
		# 记录访问时间
		self.cache_access_times[case_id] = time.time()
		
		# 添加到缓存
		self.difficulty_cache[case_id] = difficulty_map
		
		# 如果缓存已满，移除最旧的项
		if len(self.difficulty_cache) > self.max_cache_size:
			# 找出最旧访问的键
			oldest_key = min(self.cache_access_times, key=self.cache_access_times.get)
			# 删除最旧的条目
			del self.difficulty_cache[oldest_key]
			del self.cache_access_times[oldest_key]
	
	def update_difficulty(self, case_id, prediction, target):
		"""
		基于分割结果更新难度图

		参数:
			case_id: 案例ID
			prediction: 模型预测 [C, D, H, W]
			target: 真实标签 [D, H, W]

		返回:
			更新后的难度图
		"""
		# 确保数据在CPU上
		if torch.is_tensor(prediction):
			prediction = prediction.detach().cpu().numpy()
		if torch.is_tensor(target):
			target = target.detach().cpu().numpy()
		
		# 获取难度图
		difficulty_map = self.get_difficulty_map(case_id)
		if difficulty_map is None:
			if self.logger:
				self.logger.log_warning(f"无法更新难度图，案例 {case_id} 未初始化")
			return None
		
		# 计算局部性能图
		performance_map = self.compute_performance_map(prediction, target)
		
		# 计算新难度
		difficulty = 1.0 - performance_map
		
		# 更新难度图 (使用指数移动平均)
		difficulty_map = self.alpha * difficulty_map + (1 - self.alpha) * difficulty
		
		# 确保值在有效范围内
		np.clip(difficulty_map, 0.01, 0.99, out=difficulty_map)
		
		# 保存难度图
		difficulty_path = self.base_dir / f"{case_id}.npy"
		np.save(difficulty_path, difficulty_map)
		
		# 更新缓存
		self._update_cache(case_id, difficulty_map)
		
		if self.logger:
			self.logger.log_info(f"更新难度图 {case_id}，平均难度: {np.mean(difficulty_map):.3f}")
		
		return difficulty_map
	
	def compute_performance_map(self, prediction, target):
		"""
		计算局部性能图

		参数:
			prediction: 模型预测 [C, D, H, W]
			target: 真实标签 [D, H, W]

		返回:
			性能图 [D, H, W]，值域[0,1]，值越高表示性能越好
		"""
		# 确保输入形状正确
		if prediction.ndim == 4 and prediction.shape[0] == 1:
			# 单通道情况
			prediction = prediction[0]  # [D, H, W]
		
		# 二值化预测
		if prediction.dtype == np.float32 or prediction.dtype == np.float64:
			pred_binary = (prediction > 0.5).astype(np.float32)
		else:
			pred_binary = (prediction > 0).astype(np.float32)
		
		# 确保目标是二值的
		if target.dtype == np.float32 or target.dtype == np.float64:
			target_binary = (target > 0.5).astype(np.float32)
		else:
			target_binary = (target > 0).astype(np.float32)
		
		# 创建性能图
		performance_map = np.zeros_like(target_binary, dtype=np.float32)
		
		try:
			# 使用局部Dice系数作为性能度量
			# 定义局部窗口大小 (小一点以节省内存)
			window_size = min(12, *target.shape)
			
			# 使用卷积计算局部统计量
			# 转换为PyTorch张量以使用F.conv3d
			pred_tensor = torch.from_numpy(pred_binary).unsqueeze(0).unsqueeze(0).to(self.device)
			target_tensor = torch.from_numpy(target_binary).unsqueeze(0).unsqueeze(0).to(self.device)
			
			# 创建卷积核 (全1)
			kernel = torch.ones((1, 1, window_size, window_size, window_size), device=self.device)
			
			# 计算局部统计量
			pred_local_sum = F.conv3d(
				pred_tensor, kernel, padding=window_size // 2
			).squeeze().cpu().numpy()
			
			target_local_sum = F.conv3d(
				target_tensor, kernel, padding=window_size // 2
			).squeeze().cpu().numpy()
			
			intersection = F.conv3d(
				pred_tensor * target_tensor, kernel, padding=window_size // 2
			).squeeze().cpu().numpy()
			
			# 释放GPU内存
			del pred_tensor, target_tensor, kernel
			torch.cuda.empty_cache()
			
			# 计算局部Dice系数
			smooth = 1e-5
			local_dice = (2.0 * intersection + smooth) / (
					pred_local_sum + target_local_sum + smooth
			)
			
			# 使用局部Dice作为性能度量
			performance_map = local_dice
		
		except Exception as e:
			if self.logger:
				self.logger.log_warning(f"计算局部性能错误: {e}，使用全局Dice")
			
			# 出错时使用全局性能
			# 计算全局Dice
			intersection = np.sum(pred_binary * target_binary)
			dice = (2.0 * intersection) / (np.sum(pred_binary) + np.sum(target_binary) + 1e-5)
			
			# 使用全局Dice填充性能图
			performance_map.fill(dice)
		
		return performance_map
	
	def sync_difficulty_maps(self):
		"""同步所有难度图到磁盘（为了兼容性）"""
		# 处理所有挂起的更新
		with self.queue_lock:
			if not self.update_queue:
				return
			
			# 复制队列并清空
			updates = self.update_queue.copy()
			self.update_queue = []
		
		# 处理每个更新
		for case_id, prediction, target in updates:
			self.update_difficulty(case_id, prediction, target)
		
		if self.logger:
			self.logger.log_info("同步难度图到磁盘")
	
	def close(self):
		"""关闭硬样本跟踪器（为了兼容性）"""
		# 同步所有难度图
		self.sync_difficulty_maps()
		
		# 清空缓存
		self.difficulty_cache.clear()
		if hasattr(self, 'cache_access_times'):
			self.cache_access_times.clear()
		
		if self.logger:
			self.logger.log_info("关闭硬样本跟踪器")
	
	# 新增方法，但不影响原有API
	def set_epoch(self, epoch):
		"""设置当前epoch"""
		self.current_epoch = epoch
		if self.logger:
			self.logger.log_info(f"设置当前epoch: {epoch}")