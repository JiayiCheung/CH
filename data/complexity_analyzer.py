import numpy as np
from scipy import ndimage
from skimage import morphology
import torch
import torch.nn.functional as F
import time

from utils.logger import Logger  # 统一使用项目的Logger


class ComplexityAnalyzer:
	"""血管结构复杂度分析器，支持GPU加速"""
	
	def __init__(self, device='cuda', logger=None):
		"""
		初始化复杂度分析器

		参数:
			device: 计算设备，默认使用CUDA加速
			logger: 日志记录器实例，应为utils.Logger的实例
		"""
		self.device = device if torch.cuda.is_available() and 'cuda' in device else 'cpu'
		self.logger = logger
		
		# 缓存，避免重复计算
		self.complexity_cache = {}
		self.max_cache_entries = 5
		self.cache_timestamps = {}
	
	def compute_complexity(self, label_data):
		"""
		计算血管分割标签的复杂度分数，使用GPU加速关键计算

		参数:
			label_data: 分割标签数据 [D, H, W]

		返回:
			复杂度分数 (0.5-2.0)
		"""
		# 生成缓存键
		cache_key = hash(label_data.tobytes()) if hasattr(label_data, 'tobytes') else None
		if cache_key is not None and cache_key in self.complexity_cache:
			# 更新时间戳
			self.cache_timestamps[cache_key] = time.time()
			return self.complexity_cache[cache_key]
		
		# 确保数据是二值的
		binary_mask = label_data > 0
		
		# 1. 计算前景体素比例
		foreground_ratio = np.mean(binary_mask)
		if foreground_ratio < 1e-6:
			result = 0.5  # 极少或没有前景
			
			# 缓存结果
			if cache_key is not None:
				self._update_cache(cache_key, result)
			
			return result
		
		try:
			# 尝试使用GPU加速计算复杂性指标
			complexity_features = self._compute_complexity_features_gpu(binary_mask)
			
			# 提取各项指标
			branch_ratio = complexity_features.get('branch_ratio', 0)
			boundary_ratio = complexity_features.get('boundary_ratio', 0)
			component_complexity = complexity_features.get('component_complexity', 0)
			euler_complexity = complexity_features.get('euler_complexity', 0)
		
		except Exception as e:
			if self.logger:
				self.logger.log_warning(f"GPU计算复杂度特征失败: {e}，回退到CPU版本")
			
			# 回退到CPU版本计算特征
			# 2. 提取骨架和分支点
			try:
				skeleton = morphology.skeletonize(binary_mask)
				if np.sum(skeleton) < 10:  # 骨架太小
					branch_ratio = 0
				else:
					branch_points = self.detect_branch_points(skeleton)
					branch_ratio = np.sum(branch_points) / max(np.sum(skeleton), 1)
			except Exception as e:
				if self.logger:
					self.logger.log_warning(f"Error computing skeleton: {e}")
				branch_ratio = 0
			
			# 3. 计算边界复杂度
			try:
				eroded = ndimage.binary_erosion(binary_mask)
				boundary = binary_mask & (~eroded)
				boundary_ratio = np.sum(boundary) / max(np.sum(binary_mask), 1)
			except Exception as e:
				if self.logger:
					self.logger.log_warning(f"Error computing boundary: {e}")
				boundary_ratio = 0
			
			# 4. 计算连通分量
			try:
				labels, num_components = ndimage.label(binary_mask)
				component_complexity = min(num_components / 10, 1.0)  # 归一化
			except Exception as e:
				if self.logger:
					self.logger.log_warning(f"Error computing components: {e}")
				component_complexity = 0
			
			# 5. 计算欧拉数 (连通数-孔洞数)
			try:
				euler_number = self.compute_euler_number(binary_mask)
				euler_complexity = min(abs(euler_number) / 10, 1.0)  # 归一化
			except Exception as e:
				if self.logger:
					self.logger.log_warning(f"Error computing Euler number: {e}")
				euler_complexity = 0
		
		# 组合不同指标，加权计算总复杂度
		complexity = (
				0.2 * np.log1p(foreground_ratio * 1000) +  # 前景比例 (对数缩放)
				0.3 * branch_ratio * 10 +  # 分支比例 (放大)
				0.2 * boundary_ratio * 5 +  # 边界比例
				0.15 * component_complexity +  # 连通分量复杂度
				0.15 * euler_complexity  # 欧拉特征复杂度
		)
		
		# 归一化到目标范围 [0.5, 2.0]
		normalized = 0.5 + 1.5 * min(complexity, 1.0)
		
		# 缓存结果
		if cache_key is not None:
			self._update_cache(cache_key, normalized)
		
		if self.logger:
			self.logger.log_info(f"Complexity: {normalized:.3f} (fg={foreground_ratio:.3f}, "
			                     f"br={branch_ratio:.3f}, bound={boundary_ratio:.3f})")
		
		return normalized
	
	def _update_cache(self, key, value):
		"""更新缓存，使用LRU策略"""
		# 添加新条目和时间戳
		self.complexity_cache[key] = value
		self.cache_timestamps[key] = time.time()
		
		# 如果缓存超出限制，删除最旧的条目
		if len(self.complexity_cache) > self.max_cache_entries:
			oldest_key = min(self.cache_timestamps, key=self.cache_timestamps.get)
			del self.complexity_cache[oldest_key]
			del self.cache_timestamps[oldest_key]
	
	def _compute_complexity_features_gpu(self, binary_mask):
		"""
		使用GPU计算复杂度特征

		参数:
			binary_mask: 二值掩码

		返回:
			包含各种复杂度特征的字典
		"""
		# 将数据转移到GPU
		if not torch.is_tensor(binary_mask):
			mask_tensor = torch.tensor(binary_mask, dtype=torch.float32, device=self.device)
		else:
			mask_tensor = binary_mask.to(self.device).float()
		
		features = {}
		
		# 1. 计算骨架 (使用CPU，因为skeletonize没有GPU实现)
		skeleton_cpu = morphology.skeletonize(binary_mask)
		if np.sum(skeleton_cpu) < 10:  # 骨架太小
			features['branch_ratio'] = 0
		else:
			# 转移骨架到GPU
			skeleton_tensor = torch.tensor(skeleton_cpu, dtype=torch.float32, device=self.device)
			
			# 2. 计算分支点
			kernel_size = 3
			padding = kernel_size // 2
			
			if binary_mask.ndim == 3:  # 3D
				kernel = torch.ones((1, 1, kernel_size, kernel_size, kernel_size), device=self.device)
				kernel[0, 0, 1, 1, 1] = 0  # 中心点不算邻居
				
				skel_tensor = skeleton_tensor.unsqueeze(0).unsqueeze(0)  # 添加批次和通道维度
				
				# 使用卷积计算每个点的邻居数
				neighbors = F.conv3d(skel_tensor, kernel, padding=padding).squeeze()
				
				# 分支点 = 有3个或更多邻居的骨架点
				branch_points = (neighbors >= 3) & skel_tensor.squeeze().bool()
				
				# 计算分支比例
				branch_ratio = torch.sum(branch_points.float()) / max(torch.sum(skeleton_tensor), 1)
				features['branch_ratio'] = branch_ratio.item()
			
			else:  # 2D
				kernel = torch.ones((1, 1, kernel_size, kernel_size), device=self.device)
				kernel[0, 0, 1, 1] = 0
				
				skel_tensor = skeleton_tensor.unsqueeze(0).unsqueeze(0)
				
				neighbors = F.conv2d(skel_tensor, kernel, padding=padding).squeeze()
				branch_points = (neighbors >= 3) & skel_tensor.squeeze().bool()
				
				branch_ratio = torch.sum(branch_points.float()) / max(torch.sum(skeleton_tensor), 1)
				features['branch_ratio'] = branch_ratio.item()
		
		# 3. 计算边界复杂度
		if binary_mask.ndim == 3:  # 3D
			# 创建3D核进行腐蚀
			kernel = torch.ones((1, 1, 3, 3, 3), device=self.device)
			mask_tensor_expanded = mask_tensor.unsqueeze(0).unsqueeze(0)
			
			# 使用3D卷积近似腐蚀
			eroded = (F.conv3d(mask_tensor_expanded, kernel, padding=1) >= 27).float().squeeze()
			
			# 计算边界
			boundary = mask_tensor - eroded
			boundary = torch.clamp(boundary, 0, 1)
			
			# 计算边界比例
			boundary_ratio = torch.sum(boundary) / max(torch.sum(mask_tensor), 1)
			features['boundary_ratio'] = boundary_ratio.item()
		
		else:  # 2D
			kernel = torch.ones((1, 1, 3, 3), device=self.device)
			mask_tensor_expanded = mask_tensor.unsqueeze(0).unsqueeze(0)
			
			eroded = (F.conv2d(mask_tensor_expanded, kernel, padding=1) >= 9).float().squeeze()
			boundary = mask_tensor - eroded
			boundary = torch.clamp(boundary, 0, 1)
			
			boundary_ratio = torch.sum(boundary) / max(torch.sum(mask_tensor), 1)
			features['boundary_ratio'] = boundary_ratio.item()
		
		# 4. 连通分量和欧拉数需要在CPU上计算
		mask_cpu = binary_mask
		labels, num_components = ndimage.label(mask_cpu)
		features['component_complexity'] = min(num_components / 10, 1.0)
		
		euler_number = self.compute_euler_number(mask_cpu)
		features['euler_complexity'] = min(abs(euler_number) / 10, 1.0)
		
		# 清理GPU内存
		torch.cuda.empty_cache()
		
		return features
	
	def detect_branch_points(self, skeleton):
		"""
		检测骨架中的分支点 (CPU版本)

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
		
		return branch_points
	
	def compute_euler_number(self, binary_mask):
		"""
		计算二值图像的欧拉数

		参数:
			binary_mask: 二值掩码

		返回:
			欧拉数
		"""
		if binary_mask.ndim == 3:
			# 3D欧拉数计算比较复杂，这里用一个近似方法
			# 连通分量数 - 孔洞数 (近似)
			labels, num_components = ndimage.label(binary_mask)
			
			# 反转掩码，计算"孔洞"
			inv_mask = ~binary_mask
			# 移除边界连通的背景
			padded = np.pad(inv_mask, 1, mode='constant', constant_values=1)
			eroded = ndimage.binary_erosion(padded)[1:-1, 1:-1, 1:-1]
			# 标记孔洞
			labels_inv, num_holes = ndimage.label(eroded)
			
			return num_components - num_holes
		else:
			# 2D欧拉数 = 连通分量数 - 孔洞数
			labels, num_components = ndimage.label(binary_mask)
			
			# 反转掩码，计算孔洞
			inv_mask = ~binary_mask
			# 移除边界连通的背景
			padded = np.pad(inv_mask, 1, mode='constant', constant_values=1)
			eroded = ndimage.binary_erosion(padded)[1:-1, 1:-1]
			# 标记孔洞
			labels_inv, num_holes = ndimage.label(eroded)
			
			return num_components - num_holes
	
	def adjust_sampling_density(self, complexity, base_tier1, base_tier2):
		"""
		基于复杂度调整采样密度

		参数:
			complexity: 复杂度分数 (0.5-2.0)
			base_tier1: 基础Tier-1采样数
			base_tier2: 基础Tier-2采样数

		返回:
			调整后的(tier1_samples, tier2_samples)
		"""
		# 使用复杂度调整采样数量
		tier1_samples = int(base_tier1 * complexity)
		tier2_samples = int(base_tier2 * complexity)
		
		# 确保最小采样数量
		tier1_samples = max(tier1_samples, 3)
		tier2_samples = max(tier2_samples, 5)
		
		# 确保最大采样数量(防止内存爆炸)
		tier1_samples = min(tier1_samples, base_tier1 * 3)
		tier2_samples = min(tier2_samples, base_tier2 * 3)
		
		if self.logger:
			self.logger.log_info(f"Adjusted sampling: tier1={tier1_samples}, tier2={tier2_samples} "
			                     f"(complexity={complexity:.2f})")
		
		return tier1_samples, tier2_samples