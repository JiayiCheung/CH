import numpy as np
from scipy.ndimage import rotate
import torch


class Compose:
	"""组合多个变换"""
	
	def __init__(self, transforms):
		self.transforms = transforms
	
	def __call__(self, volume, mask=None):
		if mask is not None:
			for t in self.transforms:
				volume, mask = t(volume, mask)
			return volume, mask
		else:
			for t in self.transforms:
				volume = t(volume)
			return volume


class RandomRotation3D:
	"""随机3D旋转变换"""
	
	def __init__(self, max_angle=15, axes=((0, 1), (0, 2), (1, 2))):
		self.max_angle = max_angle
		self.axes = axes
	
	def __call__(self, volume, mask=None):
		# 随机选择旋转轴
		axis = self.axes[np.random.choice(len(self.axes))]
		
		# 随机旋转角度
		angle = np.random.uniform(-self.max_angle, self.max_angle)
		
		# 应用旋转
		volume = rotate(volume, angle, axes=axis, reshape=False, order=1, mode='constant', cval=0)
		
		if mask is not None:
			mask = rotate(mask, angle, axes=axis, reshape=False, order=0, mode='constant', cval=0)
			return volume, mask
		return volume


class RandomFlip:
	"""随机翻转变换"""
	
	def __init__(self, axes=(0, 1, 2)):
		self.axes = axes
	
	def __call__(self, volume, mask=None):
		for axis in self.axes:
			if np.random.random() > 0.5:
				volume = np.flip(volume, axis=axis).copy()
				if mask is not None:
					mask = np.flip(mask, axis=axis).copy()
		
		if mask is not None:
			return volume, mask
		return volume


class RandomIntensityShift:
	"""随机强度偏移变换"""
	
	def __init__(self, shift_range=(-0.1, 0.1)):
		self.shift_range = shift_range
	
	def __call__(self, volume, mask=None):
		shift = np.random.uniform(self.shift_range[0], self.shift_range[1])
		volume = volume + shift
		volume = np.clip(volume, 0, 1)
		
		if mask is not None:
			return volume, mask
		return volume


class RandomIntensityScale:
	"""随机强度缩放变换"""
	
	def __init__(self, scale_range=(0.9, 1.1)):
		self.scale_range = scale_range
	
	def __call__(self, volume, mask=None):
		scale = np.random.uniform(self.scale_range[0], self.scale_range[1])
		volume = volume * scale
		volume = np.clip(volume, 0, 1)
		
		if mask is not None:
			return volume, mask
		return volume


class RandomGaussianNoise:
	"""随机高斯噪声变换"""
	
	def __init__(self, mean=0, std=0.01):
		self.mean = mean
		self.std = std
	
	def __call__(self, volume, mask=None):
		noise = np.random.normal(self.mean, self.std, volume.shape)
		volume = volume + noise
		volume = np.clip(volume, 0, 1)
		
		if mask is not None:
			return volume, mask
		return volume


class NormalizeIntensity:
	"""强度标准化变换"""
	
	def __init__(self, method='percentile'):
		self.method = method
	
	def __call__(self, volume, mask=None):
		if self.method == 'percentile':
			p_low = np.percentile(volume, 0.5)
			p_high = np.percentile(volume, 99.5)
			volume = (volume - p_low) / (p_high - p_low + 1e-8)
			volume = np.clip(volume, 0, 1)
		elif self.method == 'minmax':
			min_val = volume.min()
			max_val = volume.max()
			volume = (volume - min_val) / (max_val - min_val + 1e-8)
		elif self.method == 'zscore':
			mean = volume.mean()
			std = volume.std()
			volume = (volume - mean) / (std + 1e-8)
		
		if mask is not None:
			return volume, mask
		return volume


def get_training_transforms(config):
	"""获取训练数据增强变换"""
	return Compose([
		NormalizeIntensity(),
		RandomRotation3D(max_angle=config['rotation_max_angle']),
		RandomFlip(),
		RandomIntensityScale(scale_range=config['intensity_scale_range']),
		RandomIntensityShift(shift_range=config['intensity_shift_range']),
		RandomGaussianNoise(std=config['gaussian_noise_std'])
	])


def get_validation_transforms():
	"""获取验证数据变换"""
	return Compose([
		NormalizeIntensity()
	])