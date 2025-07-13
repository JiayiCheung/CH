#dataset.py

"""
肝脏血管分割数据集

"""
import numpy as np
import torch
from torch.utils.data import Dataset
import nibabel as nib
import random
from tqdm import tqdm
from pathlib import Path

from .processing import CTPreprocessor
from .tier_sampling import TierSampler


class LiverVesselDataset(Dataset):
	"""肝脏血管分割数据集，支持智能采样"""
	
	def __init__(self, image_dir, label_dir, tier=None, transform=None, preprocess=True,
	             max_cases=None, random_sampling=True, enable_smart_sampling=False,
	             sampling_params=None, logger=None, config=None):
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
			sampling_params: 采样参数字典
			logger: 日志记录器实例
			config: 配置字典
		"""
		self.image_dir = Path(image_dir)
		self.label_dir = Path(label_dir)
		self.tier = tier
		self.transform = transform
		self.preprocess = preprocess
		self.enable_smart_sampling = enable_smart_sampling
		self.logger = logger
		self.config = config or {}
		
		# 初始化预处理器和采样器
		self.sampler = TierSampler(logger=logger)
		
		if preprocess:
			preprocessing_config = self.config.get('preprocessing', {})
			self.preprocessor = CTPreprocessor(
				clip_percentiles=preprocessing_config.get('clip_percentiles', (0.5, 99.5)),
				roi_threshold=preprocessing_config.get('roi_threshold', 0.8),
				roi_percentile=preprocessing_config.get('roi_percentile', 99.8),
				use_largest_cc=preprocessing_config.get('use_largest_cc', True),
				device='cuda' if torch.cuda.is_available() else 'cpu',
				logger=logger
			)
		
		# 初始化采样参数
		self.sampling_params = sampling_params or {
			'enabled': enable_smart_sampling,
			'tier1_samples': 10,
			'tier2_samples': 30,
			'importance_weight': 0.0,
			'hard_mining_weight': 0.0
		}
		
		# 加载图像和标签文件列表
		self.image_files, self.label_files = self._load_file_list(max_cases, random_sampling)
		
		# 初始采样
		self.patches, self.case_patches = self._initial_sampling()
		
		# 打印初始统计
		if logger:
			logger.log_info(f"Dataset initialized with {len(self.patches)} patches from {len(self.image_files)} cases")
		else:
			print(f"Dataset initialized with {len(self.patches)} patches from {len(self.image_files)} cases")
	
	def _load_file_list(self, max_cases, random_sampling):
		"""加载文件列表"""
		image_files = sorted(list(self.image_dir.glob("*.nii.gz")))
		
		if not image_files:
			raise ValueError(f"No .nii.gz files found in {self.image_dir}")
		
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
			if case_id.endswith('.nii'):
				case_id = case_id[:-4]
			
			# 查找标签文件
			label_path = self.label_dir / f"{case_id}.nii.gz"
			if not label_path.exists():
				# 尝试其他可能的命名格式
				alt_path = self.label_dir / f"{case_id}_seg.nii.gz"
				if alt_path.exists():
					label_path = alt_path
				else:
					if self.logger:
						self.logger.log_warning(f"Label file not found for {case_id}, skipping")
					else:
						print(f"Warning: Label file not found for {case_id}, skipping")
					continue
			
			valid_images.append(image_path)
			label_files.append(label_path)
		
		if not valid_images:
			raise ValueError("No valid image-label pairs found")
		
		return valid_images, label_files
	
	def _initial_sampling(self):
		"""执行初始采样"""
		# 设置采样参数
		self.sampler.set_sampling_params(self.sampling_params)
		
		all_patches = []
		case_patches = {}
		
		for i, (image_path, label_path) in enumerate(tqdm(
				zip(self.image_files, self.label_files),
				desc="Initial sampling",
				total=len(self.image_files),
				disable=self.logger is None  # 如果有logger则禁用tqdm
		)):
			try:
				# 加载图像和标签
				image_data = self._load_volume(image_path)
				label_data = self._load_volume(label_path)
				
				if image_data is None or label_data is None:
					continue
				
				# 预处理
				if self.preprocess:
					image_data = self.preprocessor.normalize(image_data)
					image_data, roi_mask = self.preprocessor.extract_roi(image_data, label_data)
				
				# 采样
				case_id = image_path.stem
				if case_id.endswith('.nii'):
					case_id = case_id[:-4]
				
				patches = self.sampler.sample_patches(
					image_data, label_data, case_id, self.tier
				)
				
				# 添加案例信息到每个patch
				for patch in patches:
					patch['case_id'] = case_id
					patch['case_index'] = i
				
				all_patches.extend(patches)
				case_patches[case_id] = patches
			
			except Exception as e:
				if self.logger:
					self.logger.log_error(f"Error processing {image_path}: {e}")
				else:
					print(f"Error processing {image_path}: {e}")
				continue
		
		if self.logger:
			self.logger.log_info(f"Initial sampling completed: {len(all_patches)} patches")
		else:
			print(f"Initial sampling completed: {len(all_patches)} patches")
		
		return all_patches, case_patches
	
	def _load_volume(self, file_path):
		"""加载3D体积数据"""
		try:
			nib_img = nib.load(str(file_path))
			volume = nib_img.get_fdata()
			
			# 确保数据类型
			if volume.dtype != np.float32:
				volume = volume.astype(np.float32)
			
			return volume
		except Exception as e:
			if self.logger:
				self.logger.log_error(f"Error loading {file_path}: {e}")
			else:
				print(f"Error loading {file_path}: {e}")
			return None
	
	def apply_sampling_params(self, params):
		"""
		应用新的采样参数

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
		tier1_diff = abs(self.sampling_params.get('tier1_samples', 10) - params.get('tier1_samples', 10))
		tier2_diff = abs(self.sampling_params.get('tier2_samples', 30) - params.get('tier2_samples', 30))
		
		if tier1_diff > 3 or tier2_diff > 5:
			significant_change = True
		
		# 检查权重变化是否显著
		importance_diff = abs(self.sampling_params.get('importance_weight', 0) - params.get('importance_weight', 0))
		hard_mining_diff = abs(self.sampling_params.get('hard_mining_weight', 0) - params.get('hard_mining_weight', 0))
		
		if importance_diff > 0.2 or hard_mining_diff > 0.2:
			significant_change = True
		
		# 更新采样参数
		old_params = self.sampling_params.copy()
		self.sampling_params.update(params)
		
		# 如果参数发生显著变化，重新应用到采样器
		if significant_change:
			self.sampler.set_sampling_params(self.sampling_params)
			if self.logger:
				self.logger.log_info(f"Sampling params updated: {old_params} -> {self.sampling_params}")
		
		return significant_change
	
	def update_sampling(self):
		"""
		根据当前采样参数重新采样

		返回:
			布尔值，指示是否成功更新
		"""
		if not self.sampling_params.get('enabled', False):
			if self.logger:
				self.logger.log_info("Smart sampling disabled, skipping update")
			return False
		
		if self.logger:
			self.logger.log_info("Updating sampling...")
		
		# 保存旧的样本数量
		old_count = len(self.patches)
		
		# 重新采样
		self.patches, self.case_patches = self._initial_sampling()
		
		# 记录变化
		if self.logger:
			self.logger.log_info(f"Sampling updated: {old_count} -> {len(self.patches)} patches")
		
		return True
	
	def __len__(self):
		"""返回数据集大小"""
		return len(self.patches)
	
	def __getitem__(self, idx):
		"""获取数据项"""
		if idx >= len(self.patches):
			raise IndexError(f"Index {idx} out of range for dataset of size {len(self.patches)}")
		
		patch = self.patches[idx]
		
		# 获取数据
		image_patch = patch['image'].copy()
		label_patch = patch['label'].copy()
		tier = patch['tier']
		
		# 数据增强
		if self.transform is not None:
			# 应用变换
			sample = self.transform({
				'image': image_patch,
				'label': label_patch
			})
			image_patch = sample['image']
			label_patch = sample['label']
		
		# 转换为tensor
		if not isinstance(image_patch, torch.Tensor):
			image_patch = torch.from_numpy(image_patch).float()
		if not isinstance(label_patch, torch.Tensor):
			label_patch = torch.from_numpy(label_patch).float()
		
		# 确保维度正确
		if image_patch.dim() == 3:
			image_patch = image_patch.unsqueeze(0)  # 添加通道维度
		if label_patch.dim() == 3:
			label_patch = label_patch.unsqueeze(0)
		
		return {
			'image': image_patch,
			'label': label_patch,
			'tier': tier,
			'case_id': patch.get('case_id', ''),
			'patch_id': f"{patch.get('case_id', '')}_{idx}"
		}
	
	def get_case_info(self):
		"""获取案例信息"""
		info = {
			'total_cases': len(self.image_files),
			'total_patches': len(self.patches),
			'patches_per_case': len(self.patches) / len(self.image_files) if self.image_files else 0,
			'sampling_enabled': self.sampling_params.get('enabled', False),
			'current_sampling_params': self.sampling_params.copy()
		}
		
		# 统计每个tier的patch数量
		tier_counts = {}
		for patch in self.patches:
			tier = patch['tier']
			tier_counts[tier] = tier_counts.get(tier, 0) + 1
		
		info['tier_distribution'] = tier_counts
		
		return info