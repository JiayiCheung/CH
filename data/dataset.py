import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import nibabel as nib
import random
from tqdm import tqdm
from pathlib import Path

from .processing import CTPreprocessor
from .tier_sampling import TierSampler
from .hard_sample_tracker import HardSampleTracker
from utils.sampling_scheduler import SamplingScheduler


class LiverVesselDataset(Dataset):
	"""肝脏血管分割数据集，支持智能采样和硬样本挖掘"""
	
	def __init__(self, image_dir, label_dir, tier=None, transform=None,
	             preprocess=True, max_cases=None, random_sampling=True,
	             enable_smart_sampling=True, sampling_scheduler=None,
	             hard_sample_tracker=None, difficulty_maps_dir="difficulty_maps",
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
			sampling_scheduler: 采样调度器
			hard_sample_tracker: 硬样本跟踪器
			difficulty_maps_dir: 难度图目录
			logger: 日志记录器实例
		"""
		self.image_dir = Path(image_dir)
		self.label_dir = Path(label_dir)
		self.tier = tier
		self.transform = transform
		self.preprocess = preprocess
		self.enable_smart_sampling = enable_smart_sampling
		self.logger = logger
		
		# 初始化预处理器和采样器
		self.preprocessor = CTPreprocessor() if preprocess else None
		self.sampler = TierSampler(logger=logger)
		
		# 设置采样调度器
		self.sampling_scheduler = sampling_scheduler
		if sampling_scheduler is None and enable_smart_sampling:
			self.sampling_scheduler = SamplingScheduler(logger=logger)
		
		# 设置硬样本跟踪器
		self.hard_sample_tracker = hard_sample_tracker
		if hard_sample_tracker is None and enable_smart_sampling:
			self.hard_sample_tracker = HardSampleTracker(difficulty_maps_dir, logger=logger)
		
		# 加载数据
		self.patches, self.case_patches = self._load_data(max_cases, random_sampling)
	
	def _load_data(self, max_cases, random_sampling):
		"""加载数据，应用三级采样"""
		image_files = sorted(list(self.image_dir.glob("*.nii.gz")))
		
		# 随机采样病例
		if random_sampling and max_cases and len(image_files) > max_cases:
			selected_cases = random.sample(image_files, max_cases)
		else:
			selected_cases = image_files[:max_cases] if max_cases else image_files
		
		all_patches = []
		case_patches = {}  # 记录每个案例的patches
		
		for image_path in tqdm(selected_cases, desc="Loading cases"):
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
			
			# 加载原始数据
			image_data = nib.load(str(image_path)).get_fdata()
			label_data = nib.load(str(label_path)).get_fdata()
			
			# 预处理
			if self.preprocess:
				image_data = self.preprocessor.normalize(image_data)
				liver_mask = self.preprocessor.extract_liver_roi(image_data)
			else:
				liver_mask = None
			
			# 获取难度图(如果启用智能采样)
			difficulty_map = None
			if self.enable_smart_sampling and self.hard_sample_tracker is not None:
				# 初始化难度图
				if case_id not in self.hard_sample_tracker.case_dims:
					self.hard_sample_tracker.initialize_case(case_id, label_data.shape)
				
				# 获取难度图
				difficulty_map = self.hard_sample_tracker.get_difficulty_map(case_id)
			
			# 设置采样参数
			if self.enable_smart_sampling and self.sampling_scheduler is not None:
				# 获取采样参数
				sampling_params = self.sampling_scheduler.get_tier_sampling_params()
				# 设置到采样器
				self.sampler.set_sampling_params(sampling_params)
			
			# 应用三级采样
			patch_list = self.sampler.sample(
				image_data, label_data, liver_mask,
				difficulty_map=difficulty_map, case_id=case_id
			)
			
			# 添加案例ID
			for patch in patch_list:
				patch['id'] = case_id
			
			# 根据tier过滤
			if self.tier is not None:
				patch_list = [p for p in patch_list if p['tier'] == self.tier]
			
			all_patches.extend(patch_list)
			case_patches[case_id] = patch_list
			
			# 打印统计信息
			tier_counts = {}
			for t in range(3):
				tier_counts[t] = sum(1 for p in patch_list if p['tier'] == t)
			
			if self.logger:
				self.logger.log_info(f"Case {case_id}: {len(patch_list)} patches "
				                     f"(tier0:{tier_counts.get(0, 0)}, "
				                     f"tier1:{tier_counts.get(1, 0)}, "
				                     f"tier2:{tier_counts.get(2, 0)})")
			else:
				print(f"Case {case_id}: {len(patch_list)} patches "
				      f"(tier0:{tier_counts.get(0, 0)}, "
				      f"tier1:{tier_counts.get(1, 0)}, "
				      f"tier2:{tier_counts.get(2, 0)})")
		
		if self.logger:
			self.logger.log_info(f"Total patches: {len(all_patches)}")
		else:
			print(f"Total patches: {len(all_patches)}")
		return all_patches, case_patches
	
	def update_difficulty_maps(self, model, device):
		"""
		使用当前模型更新难度图

		参数:
			model: 当前模型
			device: 计算设备
		"""
		if not self.enable_smart_sampling or self.hard_sample_tracker is None:
			return
		
		if self.logger:
			self.logger.log_info("Updating difficulty maps...")
		
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
				image = torch.from_numpy(patch['image']).float().unsqueeze(0).unsqueeze(0).to(device)
				label = torch.from_numpy(patch['label']).float().to(device)
				
				# 设置tier
				model.set_tier(0)
				
				# 前向传播
				pred = model(image)
				
				# 更新难度图
				self.hard_sample_tracker.update_difficulty(case_id, pred.squeeze(), label)
		
		# 同步难度图到磁盘
		self.hard_sample_tracker.sync_difficulty_maps()
		if self.logger:
			self.logger.log_info("Difficulty maps updated")
	
	def __len__(self):
		"""返回数据集大小"""
		return len(self.patches)
	
	def __getitem__(self, idx):
		"""获取数据集中的一个样本"""
		patch = self.patches[idx]
		
		# 获取图像和标签
		image = patch['image'].astype(np.float32)
		label = patch['label'].astype(np.float32)
		
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