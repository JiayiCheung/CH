import numpy as np
from scipy import ndimage
from skimage import morphology
import torch
import torch.nn.functional as F
import time

from utils.logger import Logger  # 统一使用项目的Logger


class ImportanceSampler:
	"""血管重要性采样器，支持GPU加速"""
	
	def __init__(self, device='cuda', logger=None):
		"""
		初始化重要性采样器

		参数:
			device: 计算设备，默认使用CUDA加速
			logger: 日志记录器实例，应为utils.Logger的实例
		"""
		self.device = device if torch.cuda.is_available() and 'cuda' in device else 'cpu'
		self.logger = logger
		
		# 缓存，避免重复计算
		self.importance_cache = {}
		self.max_cache_entries = 5
		self.cache_timestamps = {}
	
	def compute_importance_map(self, label_data, difficulty_map=None):
		"""
		计算重要性图，结合几何特征和难度信息，使用GPU加速关键计算

		参数:
			label_data: 分割标签数据 [D, H, W]
			difficulty_map: 可选的难度图 [D, H, W]

		返回:
			重要性图 [D, H, W]
		"""
		# 生成缓存键
		cache_key = hash(label_data.tobytes()) if hasattr(label_data, 'tobytes') else None
		if cache_key is not None and cache_key in self.importance_cache:
			# 更新时间戳
			self.cache_timestamps[cache_key] = time.time()
			return self.importance_cache[cache_key]
		
		# 确保输入是二值的
		binary_mask = label_data > 0
		
		# 初始化重要性图
		importance_map = np.zeros_like(binary_mask, dtype=np.float32)
		
		# 如果没有前景，返回均匀重要性
		if np.sum(binary_mask) == 0:
			importance_map.fill(1.0)
			
			# 缓存结果
			if cache_key is not None:
				self._update_cache(cache_key, importance_map)
			
			return importance_map
		
		try:
			# 1. 提取骨架 (使用CPU，因为skeletonize没有GPU实现)
			skeleton = morphology.skeletonize(binary_mask)
			
			# 2. 计算分支点与几何特征 (使用GPU加速)
			geometry_importance = self._compute_geometry_importance_gpu(binary_mask, skeleton)
			
			# 3. 集成难度信息 (如果有)
			if difficulty_map is not None:
				# 结合几何重要性和难度图，难度占比40%
				importance_map = 0.6 * geometry_importance + 0.4 * difficulty_map
			else:
				importance_map = geometry_importance
			
			# 确保重要性非负，并有最小值
			importance_map = np.maximum(importance_map, 0.1)
			
			# 确保骨架点必然被采样
			importance_map[skeleton] = np.maximum(importance_map[skeleton], 0.5)
			
			# 归一化重要性图
			if np.max(importance_map) > 0:
				importance_map = importance_map / np.max(importance_map)
			else:
				importance_map.fill(1.0)
		
		except Exception as e:
			if self.logger:
				self.logger.log_warning(f"GPU计算重要性图失败: {e}，回退到均匀分布")
			importance_map.fill(1.0)
		
		# 缓存结果
		if cache_key is not None:
			self._update_cache(cache_key, importance_map)
		
		if self.logger:
			self.logger.log_info(f"Computed importance map, avg importance: {np.mean(importance_map):.3f}")
		
		return importance_map
	
	def _update_cache(self, key, value):
		"""更新缓存，使用LRU策略"""
		# 添加新条目和时间戳
		self.importance_cache[key] = value
		self.cache_timestamps[key] = time.time()
		
		# 如果缓存超出限制，删除最旧的条目
		if len(self.importance_cache) > self.max_cache_entries:
			oldest_key = min(self.cache_timestamps, key=self.cache_timestamps.get)
			del self.importance_cache[oldest_key]
			del self.cache_timestamps[oldest_key]
	
	def _compute_geometry_importance_gpu(self, binary_mask, skeleton):
		"""
		使用GPU计算几何重要性

		参数:
			binary_mask: 二值掩码
			skeleton: 骨架

		返回:
			几何重要性图
		"""
		try:
			# 将数据转移到GPU
			if not torch.is_tensor(binary_mask):
				mask_tensor = torch.tensor(binary_mask, dtype=torch.float32, device=self.device)
			else:
				mask_tensor = binary_mask.to(self.device)
			
			if not torch.is_tensor(skeleton):
				skel_tensor = torch.tensor(skeleton, dtype=torch.float32, device=self.device)
			else:
				skel_tensor = skeleton.to(self.device)
			
			# 1. 计算分支点 (使用3D卷积)
			kernel_size = 3
			padding = kernel_size // 2
			
			if skeleton.ndim == 3:  # 3D骨架
				kernel = torch.ones((1, 1, kernel_size, kernel_size, kernel_size), device=self.device)
				kernel[0, 0, 1, 1, 1] = 0  # 中心点不算邻居
				
				skel_tensor = skel_tensor.unsqueeze(0).unsqueeze(0)  # 添加批次和通道维度
				
				# 使用卷积计算每个点的邻居数
				neighbors = F.conv3d(skel_tensor, kernel, padding=padding).squeeze()
				
				# 分支点 = 有3个或更多邻居的骨架点
				branch_points = (neighbors >= 3) & skel_tensor.squeeze().bool()
				
				# 扩大分支点影响范围
				dist_from_branch = self._compute_distance_gpu(~branch_points)
				branch_map = torch.exp(-dist_from_branch / 5.0)  # 指数衰减
			
			else:  # 2D骨架
				kernel = torch.ones((1, 1, kernel_size, kernel_size), device=self.device)
				kernel[0, 0, 1, 1] = 0
				
				skel_tensor = skel_tensor.unsqueeze(0).unsqueeze(0)
				
				neighbors = F.conv2d(skel_tensor, kernel, padding=padding).squeeze()
				branch_points = (neighbors >= 3) & skel_tensor.squeeze().bool()
				
				dist_from_branch = self._compute_distance_gpu(~branch_points)
				branch_map = torch.exp(-dist_from_branch / 5.0)
			
			# 2. 计算距离变换 - 获取半径
			dist_transform = self._compute_distance_gpu(mask_tensor)
			
			# 3. 集成几何特征 - 使用分支点图作为主要特征
			# 分支点占40%，距离变换影响占60%
			geometry_importance = 0.4 * branch_map + 0.6 * torch.exp(-dist_transform / torch.max(dist_transform) * 10)
			
			# 转回CPU
			result = geometry_importance.cpu().numpy()
			
			# 清理GPU内存
			torch.cuda.empty_cache()
			
			return result
		
		except Exception as e:
			if self.logger:
				self.logger.log_warning(f"GPU计算几何重要性失败: {e}，回退到CPU版本")
			
			# 回退到CPU版本
			return self._compute_geometry_importance_cpu(binary_mask, skeleton)
	
	def _compute_distance_gpu(self, mask):
		"""
		GPU加速的距离变换近似计算
		使用迭代扩张方法近似计算距离图

		参数:
			mask: 二值掩码 (torch.Tensor)

		返回:
			距离图 (torch.Tensor)
		"""
		# 初始化距离图
		dist = torch.zeros_like(mask, dtype=torch.float32)
		dist[~mask] = float('inf')
		
		# 创建卷积核
		if mask.dim() == 3:  # 3D
			kernel = torch.ones((1, 1, 3, 3, 3), device=self.device)
			kernel[0, 0, 1, 1, 1] = 0
			mask = mask.unsqueeze(0).unsqueeze(0)
			padding = 1
		else:  # 2D
			kernel = torch.ones((1, 1, 3, 3), device=self.device)
			kernel[0, 0, 1, 1] = 0
			mask = mask.unsqueeze(0).unsqueeze(0)
			padding = 1
		
		# 迭代传播距离值 (简化的距离变换)
		max_iter = 10  # 限制迭代次数以保证性能
		for i in range(max_iter):
			# 使用卷积传播距离值
			if mask.dim() == 5:  # 3D
				dist_prop = F.conv3d(dist.unsqueeze(0).unsqueeze(0), kernel, padding=padding).squeeze()
			else:  # 2D
				dist_prop = F.conv2d(dist.unsqueeze(0).unsqueeze(0), kernel, padding=padding).squeeze()
			
			# 更新距离图
			dist = torch.minimum(dist, dist_prop + 1)
		
		return dist
	
	def _compute_geometry_importance_cpu(self, binary_mask, skeleton):
		"""CPU版本的几何重要性计算，作为备选"""
		# 1. 计算分支点
		branch_points = self.detect_branch_points(skeleton)
		
		# 2. 扩大分支点影响范围
		branch_map = ndimage.distance_transform_edt(~branch_points)
		branch_map = np.exp(-branch_map / 5.0)  # 指数衰减
		
		# 3. 计算距离变换
		dist_transform = ndimage.distance_transform_edt(binary_mask)
		
		# 4. 集成几何特征
		geometry_importance = 0.4 * branch_map + 0.6 * np.exp(-dist_transform / np.max(dist_transform) * 10)
		
		return geometry_importance
	
	def detect_branch_points(self, skeleton):
		"""
		检测骨架中的分支点 (CPU版本，作为备选)

		参数:
			skeleton: 骨架二值图

		返回:
			分支点掩码
		"""
		# 初始化结果
		branch_points = np.zeros_like(skeleton)
		
		# 跳过空骨架
		if np.sum(skeleton) == 0:
			return branch_points
		
		# 遍历骨架上的每一点
		if skeleton.ndim == 3:  # 3D骨架
			# 创建3D卷积核，计算邻居数
			kernel = np.ones((3, 3, 3), dtype=np.uint8)
			kernel[1, 1, 1] = 0  # 中心点不算邻居
			
			# 使用卷积计算每个点的邻居数
			neighbors = ndimage.convolve(
				skeleton.astype(np.uint8),
				kernel,
				mode='constant',
				cval=0
			)
			
			# 选择有3个或更多邻居的点作为分支点
			branch_points = (neighbors >= 3) & skeleton
		else:  # 2D骨架
			# 创建2D卷积核
			kernel = np.ones((3, 3), dtype=np.uint8)
			kernel[1, 1] = 0
			
			# 使用卷积计算每个点的邻居数
			neighbors = ndimage.convolve(
				skeleton.astype(np.uint8),
				kernel,
				mode='constant',
				cval=0
			)
			
			# 选择有3个或更多邻居的点作为分支点
			branch_points = (neighbors >= 3) & skeleton
		
		if self.logger:
			self.logger.log_info(f"Detected {np.sum(branch_points)} branch points")
		
		return branch_points
	
	def importance_based_sampling(self, coords, importance_values, max_samples):
		"""
		基于重要性进行加权采样

		参数:
			coords: 坐标点数组 [[z,y,x], ...]
			importance_values: 对应的重要性值数组
			max_samples: 最大采样数量

		返回:
			采样后的坐标数组
		"""
		# 如果点数少于最大采样数，全部返回
		if len(coords) <= max_samples:
			return coords
		
		# 确保重要性值是正的
		imp_values = np.maximum(importance_values, 1e-6)
		
		# 归一化为概率
		probs = imp_values / np.sum(imp_values)
		
		# 加权随机采样
		try:
			indices = np.random.choice(
				len(coords), size=max_samples, replace=False, p=probs
			)
			sampled_coords = coords[indices]
			
			if self.logger:
				self.logger.log_info(f"Sampled {len(sampled_coords)} points using importance sampling")
			
			return sampled_coords
		except Exception as e:
			if self.logger:
				self.logger.log_warning(f"Error in importance sampling: {e}, using random sampling")
			# 出错时退化为随机采样
			indices = np.random.choice(len(coords), size=max_samples, replace=False)
			return coords[indices]