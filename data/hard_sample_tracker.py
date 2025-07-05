import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
import json

from .mmap_utils import MMapManager


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
		
		# 确保目录存在
		self.base_dir.mkdir(exist_ok=True, parents=True)
		
		# 缓存难度图和维度信息
		self.case_dims = self._load_metadata()
	
	def _load_metadata(self):
		"""从元数据文件加载案例维度信息"""
		metadata_path = self.base_dir / "difficulty_metadata.json"
		if metadata_path.exists():
			try:
				with open(metadata_path, 'r') as f:
					return json.load(f)
			except Exception as e:
				if self.logger:
					self.logger.log_warning(f"Error loading metadata: {e}")
				return {}
		return {}
	
	def _save_metadata(self):
		"""保存案例维度信息到元数据文件"""
		metadata_path = self.base_dir / "difficulty_metadata.json"
		temp_path = metadata_path.with_suffix('.tmp')
		
		try:
			# 写入临时文件
			with open(temp_path, 'w') as f:
				json.dump(self.case_dims, f)
			
			# 原子重命名
			import os
			os.replace(temp_path, metadata_path)
		except Exception as e:
			if self.logger:
				self.logger.log_warning(f"Error saving metadata: {e}")
	
	def initialize_case(self, case_id, shape):
		"""
		初始化案例的难度图

		参数:
			case_id: 案例ID
			shape: 数据形状 [D, H, W]
		"""
		# 记录案例维度
		self.case_dims[case_id] = list(shape)  # 转换为列表以便JSON序列化
		self._save_metadata()
		
		# 创建或加载难度图
		difficulty_map = MMapManager.create_or_load(
			self.base_dir,
			self.base_dir,
			case_id,
			shape,
			initial_value=0.5
		)
		
		if self.logger:
			self.logger.log_info(f"Initialized difficulty map for {case_id}, shape: {shape}")
		
		return difficulty_map is not None
	
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
				self.logger.log_warning(f"Case {case_id} not initialized")
			return None
		
		# 获取形状
		shape = tuple(self.case_dims[case_id])
		
		# 使用静态方法获取难度图
		difficulty_map = MMapManager.create_or_load(
			self.base_dir,
			self.base_dir,  # 使用相同目录作为lock_dir
			case_id,
			shape
		)
		
		return difficulty_map
	
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
		# 获取难度图
		difficulty_map = self.get_difficulty_map(case_id)
		if difficulty_map is None:
			if self.logger:
				self.logger.log_warning(f"Cannot update difficulty for {case_id}: not initialized")
			return None
		
		# 确保张量在同一设备上
		if torch.is_tensor(prediction):
			prediction = prediction.detach().cpu().numpy()
		if torch.is_tensor(target):
			target = target.detach().cpu().numpy()
		
		# 计算性能图 (1-performance 即为难度)
		performance_map = self.compute_performance_map(prediction, target)
		
		# 计算新难度
		difficulty = 1.0 - performance_map
		
		# 更新难度图 (使用指数移动平均)
		difficulty_map[:] = self.alpha * difficulty_map + (1 - self.alpha) * difficulty
		
		# 确保值在有效范围内
		np.clip(difficulty_map, 0.01, 0.99, out=difficulty_map)
		
		# 同步到磁盘
		MMapManager.sync_to_disk(difficulty_map)
		
		if self.logger:
			self.logger.log_info(f"Updated difficulty map for {case_id}, "
			                     f"avg difficulty: {np.mean(difficulty_map):.3f}")
		
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
		
		# 二值化预测 (如果是概率)
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
			# 定义局部窗口大小
			window_size = min(16, *target.shape)
			
			# 使用卷积计算局部统计量
			# 转换为PyTorch张量以使用F.conv3d
			pred_tensor = torch.from_numpy(pred_binary).unsqueeze(0).unsqueeze(0)
			target_tensor = torch.from_numpy(target_binary).unsqueeze(0).unsqueeze(0)
			
			# 创建卷积核 (全1)
			kernel = torch.ones((1, 1, window_size, window_size, window_size))
			
			# 计算局部统计量
			pred_local_sum = F.conv3d(
				pred_tensor, kernel, padding=window_size // 2
			).squeeze().numpy()
			
			target_local_sum = F.conv3d(
				target_tensor, kernel, padding=window_size // 2
			).squeeze().numpy()
			
			intersection = F.conv3d(
				pred_tensor * target_tensor, kernel, padding=window_size // 2
			).squeeze().numpy()
			
			# 计算局部Dice系数
			smooth = 1e-5
			local_dice = (2.0 * intersection + smooth) / (
					pred_local_sum + target_local_sum + smooth
			)
			
			# 使用局部Dice作为性能度量
			performance_map = local_dice
		except Exception as e:
			if self.logger:
				self.logger.log_warning(f"Error computing local performance: {e}")
			# 出错时使用全局性能
			# 计算全局Dice
			intersection = np.sum(pred_binary * target_binary)
			dice = (2.0 * intersection) / (np.sum(pred_binary) + np.sum(target_binary) + 1e-5)
			
			# 使用全局Dice填充性能图
			performance_map.fill(dice)
		
		return performance_map
	
	def sync_difficulty_maps(self):
		"""同步所有难度图到磁盘（为了兼容性）"""
		# 在新设计中，每次更新都会同步到磁盘，此方法保留仅为兼容性
		if self.logger:
			self.logger.log_info("Syncing difficulty maps to disk")
	
	def close(self):
		"""关闭硬样本跟踪器（为了兼容性）"""
		# 在新设计中不需要显式关闭，此方法保留仅为兼容性
		if self.logger:
			self.logger.log_info("Closing hard sample tracker")