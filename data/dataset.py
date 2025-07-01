import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import nibabel as nib
from pathlib import Path
import random
from tqdm import tqdm

from .preprocessing import CTPreprocessor
from .tier_sampling import TierSampler


class LiverVesselDataset(Dataset):
	"""肝脏血管分割数据集"""
	
	def __init__(self, image_dir, label_dir, tier=None, transform=None,
	             preprocess=True, max_cases=None, random_sampling=True):
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
		"""
		self.image_dir = Path(image_dir)
		self.label_dir = Path(label_dir)
		self.tier = tier
		self.transform = transform
		self.preprocess = preprocess
		
		# 初始化预处理器和采样器
		self.preprocessor = CTPreprocessor() if preprocess else None
		self.sampler = TierSampler()
		
		# 加载数据
		self.patches = self._load_data(max_cases, random_sampling)
	
	def _load_data(self, max_cases, random_sampling):
		"""加载数据，应用三级采样"""
		image_files = sorted(list(self.image_dir.glob("*.nii.gz")))
		
		# 随机采样病例
		if random_sampling and max_cases and len(image_files) > max_cases:
			selected_cases = random.sample(image_files, max_cases)
		else:
			selected_cases = image_files[:max_cases] if max_cases else image_files
		
		all_patches = []
		for image_path in tqdm(selected_cases, desc="Loading cases"):
			case_id = image_path.stem
			
			# 查找标签文件
			label_path = self.label_dir / f"{case_id}.gz"
			if not label_path.exists():
				alt_path = self.label_dir / f"{case_id}.nii.gz"
				if alt_path.exists():
					label_path = alt_path
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
			
			# 应用三级采样
			patch_list = self.sampler.sample(image_data, label_data, liver_mask)
			
			# 添加病例ID
			for patch in patch_list:
				patch['id'] = case_id
			
			# 根据tier过滤
			if self.tier is not None:
				patch_list = [p for p in patch_list if p['tier'] == self.tier]
			
			all_patches.extend(patch_list)
			
			# 打印统计信息
			tier_counts = {}
			for t in range(3):
				tier_counts[t] = sum(1 for p in patch_list if p['tier'] == t)
			
			print(f"Case {case_id}: {len(patch_list)} patches "
			      f"(tier0:{tier_counts.get(0, 0)}, "
			      f"tier1:{tier_counts.get(1, 0)}, "
			      f"tier2:{tier_counts.get(2, 0)})")
		
		print(f"Total patches: {len(all_patches)}")
		return all_patches
	
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