#!/usr/bin/env python3
"""
内存映射管理工具
"""
import os
import numpy as np
import filelock
from pathlib import Path
import json
import time

class MMapManager:
	"""
	内存映射管理器 - 纯静态方法实现
	"""
	
	@staticmethod
	def ensure_dir_exists(dir_path):
		"""确保目录存在"""
		Path(dir_path).mkdir(exist_ok=True, parents=True)
	
	@staticmethod
	def get_case_id_from_path(file_path):
		"""从文件路径提取case_id"""
		file_name = Path(file_path).stem
		# 处理.nii.gz的双重后缀
		if file_name.endswith('.nii'):
			file_name = file_name[:-4]
		
		# 处理hepaticvessel_XXX格式
		if "hepaticvessel_" in file_name:
			return file_name
		return file_name
	
	@staticmethod
	def get_map_path(base_dir, case_id):
		"""获取映射文件路径"""
		return Path(base_dir) / f"{case_id}.mmap"
	
	@staticmethod
	def get_lock_path(base_dir, case_id):
		"""获取锁文件路径"""
		return Path(base_dir) / f"{case_id}.lock"
	
	@staticmethod
	def get_meta_path(base_dir):
		"""获取元数据文件路径"""
		return Path(base_dir) / "mmap_metadata.json"
	
	@staticmethod
	def save_metadata(base_dir, metadata):
		"""保存元数据（原子操作）"""
		meta_path = MMapManager.get_meta_path(base_dir)
		temp_path = meta_path.with_suffix('.tmp')
		
		# 确保目录存在
		MMapManager.ensure_dir_exists(base_dir)
		
		# 写入临时文件
		with open(temp_path, 'w') as f:
			json.dump(metadata, f, indent=2)
		
		# 原子重命名
		os.replace(temp_path, meta_path)
	
	@staticmethod
	def load_metadata(base_dir):
		"""加载元数据"""
		meta_path = MMapManager.get_meta_path(base_dir)
		if meta_path.exists():
			try:
				with open(meta_path, 'r') as f:
					return json.load(f)
			except (json.JSONDecodeError, IOError) as e:
				print(f"Warning: Error loading metadata: {e}")
				return {}
		return {}
	
	@staticmethod
	def create_or_load(base_dir, case_id, shape, dtype=np.float32, initial_value=0.5, timeout=30):
		"""
		创建或加载内存映射（简化版本）

		参数:
			base_dir: 存储内存映射文件的目录
			case_id: 案例ID
			shape: 数据形状
			dtype: 数据类型
			initial_value: 初始值
			timeout: 锁超时时间

		返回:
			内存映射数组或None（失败时）
		"""
		# 确保目录存在
		MMapManager.ensure_dir_exists(base_dir)
		
		map_path = MMapManager.get_map_path(base_dir, case_id)
		lock_path = MMapManager.get_lock_path(base_dir, case_id)
		
		# 创建文件锁
		lock = filelock.FileLock(str(lock_path), timeout=timeout)
		
		try:
			with lock:
				# 如果文件已存在，尝试加载
				if map_path.exists():
					try:
						mmap_array = np.memmap(
							str(map_path),
							dtype=dtype,
							mode='r+',
							shape=shape
						)
						return mmap_array
					except (ValueError, OSError) as e:
						print(f"Warning: Error loading existing mmap for {case_id}: {e}")
						# 删除损坏的文件
						if map_path.exists():
							os.remove(str(map_path))
				
				# 创建新的内存映射文件
				mmap_array = np.memmap(
					str(map_path),
					dtype=dtype,
					mode='w+',
					shape=shape
				)
				
				# 初始化数据
				mmap_array.fill(initial_value)
				mmap_array.flush()
				
				# 更新元数据
				metadata = MMapManager.load_metadata(base_dir)
				metadata[case_id] = {
					"shape": list(shape),
					"dtype": str(dtype),
					"created_at": time.time(),
					"file_size": os.path.getsize(str(map_path))
				}
				MMapManager.save_metadata(base_dir, metadata)
				
				return mmap_array
		
		except filelock.Timeout:
			print(f"Warning: Lock timeout for {case_id}")
			return None
		except Exception as e:
			print(f"Error in create_or_load for {case_id}: {e}")
			return None
	
	@staticmethod
	def sync_to_disk(mmap_array):
		"""同步映射到磁盘"""
		if mmap_array is not None:
			try:
				mmap_array.flush()
				return True
			except Exception as e:
				print(f"Error syncing mmap to disk: {e}")
				return False
		return False
	
	@staticmethod
	def remove_mmap(base_dir, case_id, timeout=10):
		"""移除内存映射文件"""
		map_path = MMapManager.get_map_path(base_dir, case_id)
		lock_path = MMapManager.get_lock_path(base_dir, case_id)
		
		# 创建文件锁
		lock = filelock.FileLock(str(lock_path), timeout=timeout)
		
		try:
			with lock:
				# 删除映射文件
				if map_path.exists():
					os.remove(str(map_path))
				
				# 更新元数据
				metadata = MMapManager.load_metadata(base_dir)
				if case_id in metadata:
					del metadata[case_id]
					MMapManager.save_metadata(base_dir, metadata)
				
				# 删除锁文件
				if lock_path.exists():
					os.remove(str(lock_path))
				
				return True
		except Exception as e:
			print(f"Error removing mmap for {case_id}: {e}")
			return False
	
	@staticmethod
	def list_available_cases(base_dir):
		"""列出所有可用的案例"""
		metadata = MMapManager.load_metadata(base_dir)
		return list(metadata.keys())
	
	@staticmethod
	def get_case_info(base_dir, case_id):
		"""获取案例信息"""
		metadata = MMapManager.load_metadata(base_dir)
		return metadata.get(case_id, {})
	
	@staticmethod
	def cleanup_orphaned_files(base_dir):
		"""清理孤儿文件（映射文件存在但元数据中没有记录）"""
		if not Path(base_dir).exists():
			return
		
		metadata = MMapManager.load_metadata(base_dir)
		metadata_cases = set(metadata.keys())
		
		# 查找所有.mmap文件
		mmap_files = list(Path(base_dir).glob("*.mmap"))
		
		orphaned_count = 0
		for mmap_file in mmap_files:
			case_id = mmap_file.stem
			if case_id not in metadata_cases:
				try:
					os.remove(str(mmap_file))
					orphaned_count += 1
					print(f"Removed orphaned file: {mmap_file}")
				except Exception as e:
					print(f"Error removing orphaned file {mmap_file}: {e}")
		
		# 清理孤儿锁文件
		lock_files = list(Path(base_dir).glob("*.lock"))
		for lock_file in lock_files:
			case_id = lock_file.stem
			map_path = MMapManager.get_map_path(base_dir, case_id)
			if not map_path.exists():
				try:
					os.remove(str(lock_file))
					print(f"Removed orphaned lock: {lock_file}")
				except Exception as e:
					print(f"Error removing orphaned lock {lock_file}: {e}")
		
		if orphaned_count > 0:
			print(f"Cleaned up {orphaned_count} orphaned files")
	
	@staticmethod
	def get_storage_stats(base_dir):
		"""获取存储统计信息"""
		if not Path(base_dir).exists():
			return {"total_cases": 0, "total_size": 0}
		
		metadata = MMapManager.load_metadata(base_dir)
		
		total_size = 0
		valid_cases = 0
		
		for case_id, info in metadata.items():
			map_path = MMapManager.get_map_path(base_dir, case_id)
			if map_path.exists():
				total_size += info.get("file_size", 0)
				valid_cases += 1
		
		return {
			"total_cases": valid_cases,
			"total_size": total_size,
			"total_size_mb": total_size / (1024 * 1024),
			"metadata_cases": len(metadata),
			"base_dir": str(base_dir)
		}

# 向后兼容的便捷函数
def create_mmap_manager(base_dir="difficulty_maps"):
	"""
	向后兼容的工厂函数
	返回的是类，使用静态方法
	"""
	import warnings
	warnings.warn(
		"create_mmap_manager is deprecated. Use MMapManager static methods directly.",
		DeprecationWarning,
		stacklevel=2
	)
	return MMapManager