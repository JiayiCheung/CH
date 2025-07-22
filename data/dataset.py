# data/dataset.py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import nibabel as nib
import random
from tqdm import tqdm
from pathlib import Path
import pickle

from .processing import CTPreprocessor
from .tier_sampling import TierSampler
from .mmap_utils import MMapManager


class LiverVesselDataset(Dataset):
	"""肝脏血管分割数据集，支持智能采样和缓存优化"""
	
	def __init__(self, image_dir, label_dir, tier=None, transform=None,
	             preprocess=True, max_cases=None, random_sampling=True,
	             enable_smart_sampling=True, sampling_params=None,
	             hard_sample_tracker=None, difficulty_maps_dir="difficulty_maps",
	             cache_dir="dataset_preprocessing_cache",
	             logger=None):
		"""
		初始化数据集

		参数:
			image_dir: 图像目录路径
			label_dir: 标签目录路径
			tier: 只加载特定tier的数据，如果为None则加载所有tier
			transform: 数据增强变换
			preprocess: 是否进行预处理
			max_cases: 最大加载病例数
			random_sampling: 是否随机采样病例
			enable_smart_sampling: 是否启用智能采样
			sampling_params: 采样参数字典，优先级高于sampling_scheduler
			hard_sample_tracker: 硬样本跟踪器
			difficulty_maps_dir: 难度图目录
			cache_dir: 缓存目录
			logger: 日志记录器实例
		"""
		self.image_dir = Path(image_dir)
		self.label_dir = Path(label_dir)
		self.tier = tier
		self.transform = transform
		self.preprocess = preprocess
		self.enable_smart_sampling = enable_smart_sampling
		self.cache_dir = cache_dir
		self.logger = logger
		
		# 确保缓存目录存在
		MMapManager.ensure_dir_exists(self.cache_dir)
		
		# 初始化预处理器和采样器
		self.preprocessor = CTPreprocessor(logger=logger) if preprocess else None
		self.sampler = TierSampler(logger=logger)
		
		# 初始化采样参数
		self.sampling_params = sampling_params or {
			'enabled': enable_smart_sampling,
			'tier1_samples': 10,
			'tier2_samples': 20,
			'importance_weight': 0.0,
			'hard_mining_weight': 0.0
		}
		
		# 设置硬样本跟踪器
		self.hard_sample_tracker = hard_sample_tracker
		if hard_sample_tracker is None and enable_smart_sampling:
			# 延迟导入避免循环引用
			from .hard_sample_tracker import HardSampleTracker
			self.hard_sample_tracker = HardSampleTracker(difficulty_maps_dir, logger=logger)
		
		# 加载图像和标签文件列表
		self.image_files, self.label_files = self._load_file_list(max_cases, random_sampling)
		
		# 执行初始采样（带缓存优化）
		self.patches, self.case_patches = self._initial_sampling()
		
		# 打印初始统计
		if logger:
			logger.log_info(f"初始化完成，数据集包含 {len(self.patches)} 个样本")
	
	def _load_file_list(self, max_cases, random_sampling):
		"""加载文件列表"""
		image_files = sorted(list(self.image_dir.glob("*.nii.gz")))
		
		# 随机采样病例
		if random_sampling and max_cases and len(image_files) > max_cases:
			selected_images = random.sample(image_files, max_cases)
		else:
			selected_images = image_files[:max_cases] if max_cases else image_files
		
		# 查找对应的标签文件
		label_files = []
		valid_images = []
		
		for image_path in selected_images:
			case_id = image_path.stem
			
			# 查找标签文件
			label_path = self.label_dir / f"{case_id}.gz"
			if not label_path.exists():
				alt_path = self.label_dir / f"{case_id}.nii.gz"
				if alt_path.exists():
					label_path = alt_path
				else:
					if self.logger:
						self.logger.log_warning(f"错误：标签文件不存在，跳过 {case_id}")
					else:
						print(f"错误：标签文件不存在，跳过 {case_id}")
					continue
			
			valid_images.append(image_path)
			label_files.append(label_path)
		
		return valid_images, label_files
	
	def _initial_sampling(self):
		"""执行初始采样 - 融合智能采样和缓存优化"""
		# 设置采样参数
		self.sampler.set_sampling_params(self.sampling_params)
		
		all_patches = []
		case_patches = {}
		
		for i, (image_path, label_path) in enumerate(tqdm(
				zip(self.image_files, self.label_files),
				desc="Processing cases with cache optimization",
				total=len(self.image_files),
				disable=self.logger is None
		)):
			case_id = MMapManager.get_case_id_from_path(image_path)
			
			# 尝试加载缓存
			cached_patches = self._load_cached_case_data(case_id, i)
			if cached_patches is not None:
				all_patches.extend(cached_patches)
				case_patches[case_id] = cached_patches
				if self.logger:
					self.logger.log_info(f"Loaded cached data for {case_id}: {len(cached_patches)} patches")
				continue
			
			# 缓存未命中，执行完整处理
			if self.logger:
				self.logger.log_info(f"Cache miss for {case_id}, processing from scratch...")
			
			patches = self._process_case_from_scratch(image_path, label_path, case_id, i)
			if patches is None:
				continue
			
			# 保存到缓存（失败则直接报错）
			self._save_case_to_cache(case_id, patches)
			
			# 添加到结果
			all_patches.extend(patches)
			case_patches[case_id] = patches
			
			# 打印统计信息
			tier_counts = {}
			for t in range(3):
				tier_counts[t] = sum(1 for p in patches if p['tier'] == t)
			
			if self.logger:
				self.logger.log_info(f"Processed and cached {case_id}: {len(patches)} patches "
				                     f"(tier0:{tier_counts.get(0, 0)}, "
				                     f"tier1:{tier_counts.get(1, 0)}, "
				                     f"tier2:{tier_counts.get(2, 0)})")
		
		if self.logger:
			self.logger.log_info(
				f"Initial sampling completed: {len(all_patches)} patches from {len(case_patches)} cases")
		
		return all_patches, case_patches
	
	def _load_cached_case_data(self, case_id, case_index):
		"""加载缓存的case数据，失败返回None"""
		try:
			# 定义缓存键
			image_key = f"{case_id}_processed_image"
			label_key = f"{case_id}_processed_label"
			patches_key = f"{case_id}_patches_data"
			
			# 加载图像数据
			image_mmap = MMapManager.create_or_load(self.cache_dir, image_key, (512, 512, 300), np.float32, 0.0)
			if image_mmap is None or image_mmap.sum() == 0:
				return None
			
			# 加载标签数据
			label_mmap = MMapManager.create_or_load(self.cache_dir, label_key, (512, 512, 300), np.uint8, 0)
			if label_mmap is None or label_mmap.sum() == 0:
				return None
			
			# 加载patches信息
			patches_mmap = MMapManager.create_or_load(self.cache_dir, patches_key, (10240000000,), np.uint8, 0)
			if patches_mmap is None:
				return None
			
			# 解析patches数据
			patches_bytes = patches_mmap[patches_mmap != 0].tobytes()
			if len(patches_bytes) == 0:
				return None
			
			# 反序列化patches
			patches_str = patches_bytes.decode('utf-8', errors='ignore').rstrip('\x00')
			if not patches_str:
				return None
			
			patches = pickle.loads(patches_str.encode('latin1'))
			if not isinstance(patches, list) or len(patches) == 0:
				return None
			
			# 为每个patch添加缓存数据和智能采样相关信息
			cached_image_data = np.array(image_mmap, copy=True)
			cached_label_data = np.array(label_mmap, copy=True)
			
			for patch in patches:
				patch['case_id'] = case_id
				patch['case_index'] = case_index
				patch['cached'] = True
				patch['cached_image_data'] = cached_image_data
				patch['cached_label_data'] = cached_label_data
				# 添加智能采样需要的ID字段
				patch['id'] = case_id
			
			# 根据tier过滤
			if self.tier is not None:
				patches = [p for p in patches if p['tier'] == self.tier]
			
			return patches
		
		except Exception as e:
			if self.logger:
				self.logger.log_warning(f"Failed to load cached data for {case_id}: {e}")
			return None
	
	def _process_case_from_scratch(self, image_path, label_path, case_id, case_index):
		"""从头处理case，包含完整的智能采样功能"""
		try:
			# 加载原始数据
			image_data = nib.load(str(image_path)).get_fdata()
			label_data = nib.load(str(label_path)).get_fdata()
			
			if image_data is None or label_data is None:
				if self.logger:
					self.logger.log_error(f"Failed to load data for {case_id}")
				return None
			
			# 执行预处理
			processed_image_data = image_data
			processed_label_data = label_data
			liver_mask = None
			
			if self.preprocess:
				processed_image_data = self.preprocessor.normalize(image_data)
				liver_mask = self.preprocessor.extract_liver_roi(processed_image_data)
			
			# 获取难度图(如果启用智能采样)
			difficulty_map = None
			if self.enable_smart_sampling and self.sampling_params['enabled'] and self.hard_sample_tracker is not None:
				# 初始化难度图
				if case_id not in self.hard_sample_tracker.case_dims:
					self.hard_sample_tracker.initialize_case(case_id, label_data.shape)
				
				# 获取难度图
				difficulty_map = self.hard_sample_tracker.get_difficulty_map(case_id)
			
			# 应用三级采样
			patch_list = self.sampler.sample(
				processed_image_data, processed_label_data, liver_mask,
				difficulty_map=difficulty_map, case_id=case_id
			)
			
			# 为每个patch添加完整信息
			for patch_idx, patch in enumerate(patch_list):
				patch['case_id'] = case_id
				patch['case_index'] = case_index
				patch['patch_index'] = patch_idx
				patch['cached'] = False
				patch['processed_image_data'] = processed_image_data
				patch['processed_label_data'] = processed_label_data
				# 添加智能采样需要的ID字段
				patch['id'] = case_id
			
			return patch_list
		
		except Exception as e:
			if self.logger:
				self.logger.log_error(f"Failed to process case {case_id}: {e}")
			return None
	
	def _save_case_to_cache(self, case_id, patches):
		"""保存case到缓存，失败直接报错"""
		if len(patches) == 0:
			raise ValueError(f"No patches to cache for case {case_id}")
		
		try:
			# 获取预处理数据
			processed_image_data = patches[0]['processed_image_data']
			processed_label_data = patches[0]['processed_label_data']
			
			# 定义缓存键
			image_key = f"{case_id}_processed_image"
			label_key = f"{case_id}_processed_label"
			patches_key = f"{case_id}_patches_data"
			
			# 保存图像数据
			actual_shape = processed_image_data.shape
			image_mmap = MMapManager.create_or_load(self.cache_dir, image_key, actual_shape, np.float32, 0.0)
			if image_mmap is None:
				raise RuntimeError(f"Failed to create image cache for {case_id}")
			
			image_mmap[:] = processed_image_data
			MMapManager.sync_to_disk(image_mmap)
			
			# 保存标签数据
			label_mmap = MMapManager.create_or_load(self.cache_dir, label_key, actual_shape, np.uint8, 0)
			if label_mmap is None:
				raise RuntimeError(f"Failed to create label cache for {case_id}")
			
			label_mmap[:] = processed_label_data.astype(np.uint8)
			MMapManager.sync_to_disk(label_mmap)
			
			# 保存patches信息 (移除大数据引用)
			patches_to_cache = []
			for patch in patches:
				patch_copy = {k: v for k, v in patch.items()
				              if k not in ['processed_image_data', 'processed_label_data']}
				patches_to_cache.append(patch_copy)
			
			# 序列化并保存patches
			patches_str = pickle.dumps(patches_to_cache).decode('latin1')
			patches_bytes = patches_str.encode('utf-8')
			
			if len(patches_bytes) >= 10240000000:
				raise ValueError(f"Patches data too large for case {case_id}: {len(patches_bytes)} bytes")
			
			patches_mmap = MMapManager.create_or_load(self.cache_dir, patches_key, (10240000000,), np.uint8, 0)
			if patches_mmap is None:
				raise RuntimeError(f"Failed to create patches cache for {case_id}")
			
			patches_mmap[:len(patches_bytes)] = list(patches_bytes)
			MMapManager.sync_to_disk(patches_mmap)
			
			if self.logger:
				self.logger.log_info(f"Successfully cached case {case_id}")
		
		except Exception as e:
			if self.logger:
				self.logger.log_error(f"Failed to cache case {case_id}: {e}")
			raise RuntimeError(f"Cache operation failed for case {case_id}: {e}")
	
	def apply_sampling_params(self, params):
		"""
		应用新的采样参数 - 用于周期性更新采样策略

		参数:
			params: 新的采样参数字典

		返回:
			布尔值，指示是否需要重新采样
		"""
		# 检查参数是否发生显著变化
		significant_change = False
		
		# 检查是否切换了采样开关
		if self.sampling_params.get('enabled', False) != params.get('enabled', False):
			significant_change = True
		
		# 检查tier采样数量变化是否显著
		if (abs(self.sampling_params.get('tier1_samples', 10) - params.get('tier1_samples', 10)) > 3 or
				abs(self.sampling_params.get('tier2_samples', 30) - params.get('tier2_samples', 30)) > 5):
			significant_change = True
		
		# 检查权重变化是否显著
		if (abs(self.sampling_params.get('importance_weight', 0) - params.get('importance_weight', 0)) > 0.2 or
				abs(self.sampling_params.get('hard_mining_weight', 0) - params.get('hard_mining_weight', 0)) > 0.2):
			significant_change = True
		
		# 更新采样参数
		old_params = self.sampling_params.copy()
		self.sampling_params.update(params)
		
		# 如果参数发生显著变化，应用到采样器
		if significant_change:
			self.sampler.set_sampling_params(self.sampling_params)
			if self.logger:
				self.logger.log_info(f"采样参数已更新: {old_params} -> {self.sampling_params}")
		
		return significant_change
	
	def update_sampling(self):
		"""
		根据当前采样参数重新采样

		返回:
			布尔值，指示是否成功更新
		"""
		if not self.sampling_params.get('enabled', False):
			if self.logger:
				self.logger.log_info("智能采样已禁用，跳过更新")
			return False
		
		if self.logger:
			self.logger.log_info("开始更新采样...")
		
		# 保存旧的样本数量，用于比较
		old_count = len(self.patches)
		
		# 重新采样
		self.patches, self.case_patches = self._initial_sampling()
		
		# 记录变化
		if self.logger:
			self.logger.log_info(f"采样已更新: {old_count} -> {len(self.patches)} 个样本")
		
		return True
	
	def update_difficulty_maps(self, model, device):
		"""
		使用当前模型更新难度图

		参数:
			model: 当前模型
			device: 计算设备

		返回:
			布尔值，指示是否成功更新
		"""
		if not self.enable_smart_sampling or self.hard_sample_tracker is None:
			return False
		
		if not self.sampling_params.get('enabled', False) or self.sampling_params.get('hard_mining_weight', 0) <= 0:
			if self.logger:
				self.logger.log_info("难样本挖掘已禁用，跳过更新难度图")
			return False
		
		if self.logger:
			self.logger.log_info("更新难度图...")
		
		# 设置模型为评估模式
		model.eval()
		
		with torch.no_grad():
			for case_id, patches in self.case_patches.items():
				# 只处理有足够多patches的案例
				if len(patches) < 5:
					continue
				
				# 获取Tier-0 patch (全局视图)
				tier0_patches = [p for p in patches if p['tier'] == 0]
				if not tier0_patches:
					continue
				
				# 使用Tier-0进行预测
				patch = tier0_patches[0]
				
				# 获取图像数据（优先使用缓存）
				if patch.get('cached', False) and 'cached_image_data' in patch:
					image_data = patch['cached_image_data']
				else:
					image_data = patch['image']
				
				# 获取标签数据
				if patch.get('cached', False) and 'cached_label_data' in patch:
					label_data = patch['cached_label_data']
				else:
					label_data = patch['label']
				
				image = torch.from_numpy(image_data).float().unsqueeze(0).unsqueeze(0).to(device)
				label = torch.from_numpy(label_data).float().to(device)
				
				# 设置tier
				if hasattr(model, 'set_tier'):
					model.set_tier(0)
				
				# 前向传播
				pred = model(image)
				
				# 更新难度图
				self.hard_sample_tracker.update_difficulty(case_id, pred.squeeze().cpu().numpy(), label.cpu().numpy())
		
		# 同步难度图到磁盘
		self.hard_sample_tracker.sync_difficulty_maps()
		
		if self.logger:
			self.logger.log_info("难度图已更新")
		
		return True
	
	def __len__(self):
		"""返回数据集大小"""
		return len(self.patches)
	
	def __getitem__(self, idx):
		"""获取数据集中的一个样本"""
		patch = self.patches[idx]
		
		# 获取图像和标签（优先使用缓存数据）
		if patch.get('cached', False):
			if 'cached_image_data' in patch and 'cached_label_data' in patch:
				# 使用缓存的完整数据，需要根据patch坐标提取对应区域
				full_image = patch['cached_image_data']
				full_label = patch['cached_label_data']
				
				# 提取patch区域（假设patch中包含坐标信息）
				if 'bbox' in patch:
					bbox = patch['bbox']
					image = full_image[bbox[0]:bbox[1], bbox[2]:bbox[3], bbox[4]:bbox[5]]
					label = full_label[bbox[0]:bbox[1], bbox[2]:bbox[3], bbox[4]:bbox[5]]
				else:
					# 如果没有bbox信息，使用完整图像
					image = full_image
					label = full_label
			else:
				# fallback到patch自带的数据
				image = patch['image']
				label = patch['label']
		else:
			# 使用patch自带的数据
			image = patch['image']
			label = patch['label']
		
		image = image.astype(np.float32)
		label = label.astype(np.float32)
		
		# 应用变换
		if self.transform:
			transformed = self.transform(image=image, mask=label)
			image = transformed['image']
			label = transformed['mask']
		
		# 转换为PyTorch张量
		image_tensor = torch.from_numpy(image).unsqueeze(0)  # 添加通道维度
		label_tensor = torch.from_numpy(label)
		
		return {
			'image': image_tensor,
			'label': label_tensor,
			'tier': patch['tier'],
			'case_id': patch['id']
		}