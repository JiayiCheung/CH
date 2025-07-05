import os
import numpy as np
import filelock
from pathlib import Path
import json


class MMapManager:
	"""
	内存映射管理器 - 工厂模式实现
	使用静态方法替代实例方法，避免序列化问题
	支持Task08_HepaticVessel数据集的命名规则
	"""
	
	@staticmethod
	def ensure_dir_exists(dir_path):
		"""确保目录存在"""
		Path(dir_path).mkdir(exist_ok=True, parents=True)
	
	@staticmethod
	def get_case_id_from_path(file_path):
		"""从文件路径提取case_id"""
		file_name = Path(file_path).stem
		# 处理可能的命名格式如 "hepaticvessel_458.nii.gz"
		if "hepaticvessel_" in file_name:
			return file_name  # 直接使用文件名作为case_id
		return file_name
	
	@staticmethod
	def get_map_path(base_dir, case_id):
		"""获取映射文件路径"""
		return Path(base_dir) / f"{case_id}.mmap"
	
	@staticmethod
	def get_lock_path(lock_dir, case_id):
		"""获取锁文件路径"""
		return Path(lock_dir) / f"{case_id}.lock"
	
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
			json.dump(metadata, f)
		
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
			except Exception as e:
				print(f"Error loading metadata: {e}")
				return {}
		return {}
	
	@staticmethod
	def create_or_load(base_dir, lock_dir, case_id, shape, dtype=np.float32, initial_value=0.5):
		"""
		创建或加载内存映射（静态工厂方法）

		参数:
			base_dir: 存储内存映射文件的目录
			lock_dir: 存储文件锁的目录
			case_id: 案例ID
			shape: 数据形状
			dtype: 数据类型
			initial_value: 初始值

		返回:
			内存映射数组
		"""
		# 确保目录存在
		MMapManager.ensure_dir_exists(base_dir)
		MMapManager.ensure_dir_exists(lock_dir)
		
		map_path = MMapManager.get_map_path(base_dir, case_id)
		lock_path = MMapManager.get_lock_path(lock_dir, case_id)
		
		# 每次调用创建新的锁对象（局部变量）
		# 使用超时参数避免无限等待
		lock = filelock.FileLock(str(lock_path), timeout=30)
		
		try:
			with lock:
				# 如果文件已存在，直接加载
				if map_path.exists():
					try:
						mmap_array = np.memmap(
							str(map_path),
							dtype=dtype,
							mode='r+',
							shape=shape
						)
						return mmap_array
					except Exception as e:
						print(f"Error loading mmap for {case_id}: {e}. Recreating...")
						# 如果加载失败，删除并重新创建
						if map_path.exists():
							os.remove(str(map_path))
				
				# 创建新映射
				mmap_array = np.memmap(
					str(map_path),
					dtype=dtype,
					mode='w+',
					shape=shape
				)
				mmap_array.fill(initial_value)
				mmap_array.flush()
				
				# 更新元数据
				metadata = MMapManager.load_metadata(base_dir)
				metadata[case_id] = {
					"shape": list(shape),
					"dtype": str(dtype),
					"created_at": str(Path(map_path).stat().st_mtime)
				}
				MMapManager.save_metadata(base_dir, metadata)
				
				return mmap_array
		
		except filelock.Timeout:
			print(f"Lock timeout for {case_id}, returning None")
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
	def remove_mmap(base_dir, lock_dir, case_id):
		"""移除内存映射文件"""
		map_path = MMapManager.get_map_path(base_dir, case_id)
		lock_path = MMapManager.get_lock_path(lock_dir, case_id)
		
		# 使用锁确保安全删除
		lock = filelock.FileLock(str(lock_path), timeout=10)
		
		try:
			with lock:
				if map_path.exists():
					os.remove(str(map_path))
				
				# 更新元数据
				metadata = MMapManager.load_metadata(base_dir)
				if case_id in metadata:
					del metadata[case_id]
					MMapManager.save_metadata(base_dir, metadata)
				
				return True
		except Exception as e:
			print(f"Error removing mmap for {case_id}: {e}")
			return False
	
	@staticmethod
	def list_available_cases(base_dir):
		"""列出所有可用的案例"""
		metadata = MMapManager.load_metadata(base_dir)
		return list(metadata.keys())
	
	# 为原始MMapManager提供兼容包装的静态方法
	
	@staticmethod
	def create_manager(base_dir="difficulty_maps", lock_dir=None):
		"""
		创建兼容旧API的MMapManager包装对象
		此方法仅用于兼容旧代码，不推荐在新代码中使用
		"""
		return MMapManagerLegacyWrapper(base_dir, lock_dir)


# 向后兼容包装器，用于支持旧代码
class MMapManagerLegacyWrapper:
	"""兼容原始MMapManager API的包装器"""
	
	def __init__(self, base_dir="difficulty_maps", lock_dir=None):
		"""初始化包装器"""
		self.base_dir = base_dir
		self.lock_dir = lock_dir if lock_dir else base_dir
		# 确保目录存在
		MMapManager.ensure_dir_exists(self.base_dir)
		MMapManager.ensure_dir_exists(self.lock_dir)
	
	# 模拟原始类的行为但不存储open_maps
	
	def get_map_path(self, case_id):
		"""获取映射文件路径"""
		return MMapManager.get_map_path(self.base_dir, case_id)
	
	def get_lock_path(self, case_id):
		"""获取锁文件路径"""
		return MMapManager.get_lock_path(self.lock_dir, case_id)
	
	def create_or_load(self, case_id, shape, dtype=np.float32, initial_value=0.5):
		"""创建或加载内存映射"""
		return MMapManager.create_or_load(
			self.base_dir, self.lock_dir, case_id, shape, dtype, initial_value
		)
	
	def sync_to_disk(self, case_id=None, mmap_array=None):
		"""同步映射到磁盘"""
		if mmap_array is not None:
			return MMapManager.sync_to_disk(mmap_array)
		return True  # 如果没有提供数组，假设成功
	
	def close(self, case_id=None):
		"""关闭映射（空操作，用于兼容性）"""
		pass  # 在新设计中不需要关闭操作
	
	def __getstate__(self):
		"""控制序列化行为"""
		state = self.__dict__.copy()
		# 确保所有属性都是可序列化的
		return state
	
	def __setstate__(self, state):
		"""控制反序列化行为"""
		self.__dict__.update(state)