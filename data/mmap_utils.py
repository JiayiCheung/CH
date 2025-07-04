import os
import numpy as np
import filelock
from pathlib import Path

class MMapManager:
    """内存映射管理器，处理难度图的创建、加载和同步"""
    
    def __init__(self, base_dir="difficulty_maps", lock_dir=None, logger=None):
        """
        初始化内存映射管理器
        
        参数:
            base_dir: 存储内存映射文件的目录
            lock_dir: 存储文件锁的目录，默认与base_dir相同
            logger: 日志记录器实例
        """
        self.base_dir = Path(base_dir)
        self.lock_dir = Path(lock_dir) if lock_dir else self.base_dir
        self.logger = logger
        
        # 创建必要的目录
        self.base_dir.mkdir(exist_ok=True, parents=True)
        self.lock_dir.mkdir(exist_ok=True, parents=True)
        
        # 跟踪打开的映射
        self.open_maps = {}
    
    def get_map_path(self, case_id):
        """获取案例对应的映射文件路径"""
        return self.base_dir / f"{case_id}.mmap"
    
    def get_lock_path(self, case_id):
        """获取案例对应的锁文件路径"""
        return self.lock_dir / f"{case_id}.lock"
    
    def create_or_load(self, case_id, shape, dtype=np.float32, initial_value=0.5):
        """
        创建或加载内存映射
        
        参数:
            case_id: 案例ID
            shape: 映射的形状
            dtype: 数据类型
            initial_value: 初始值
        
        返回:
            内存映射数组
        """
        map_path = self.get_map_path(case_id)
        lock_path = self.get_lock_path(case_id)
        
        # 使用文件锁确保线程/进程安全
        with filelock.FileLock(str(lock_path)):
            if map_path.exists():
                # 加载现有映射
                try:
                    mmap_array = np.memmap(
                        str(map_path), dtype=dtype, mode='r+',
                        shape=shape
                    )
                    if self.logger:
                        self.logger.log_info(f"Loaded existing mmap for {case_id}, shape: {shape}")
                except Exception as e:
                    # 文件可能损坏，重新创建
                    if self.logger:
                        self.logger.log_warning(f"Error loading mmap for {case_id}: {e}. Recreating...")
                    if map_path.exists():
                        os.remove(str(map_path))
                    mmap_array = self._create_new(str(map_path), shape, dtype, initial_value)
            else:
                # 创建新映射
                mmap_array = self._create_new(str(map_path), shape, dtype, initial_value)
        
        # 跟踪打开的映射
        self.open_maps[case_id] = mmap_array
        return mmap_array
    
    def _create_new(self, path, shape, dtype, initial_value):
        """创建新的内存映射并初始化"""
        mmap_array = np.memmap(path, dtype=dtype, mode='w+', shape=shape)
        mmap_array.fill(initial_value)
        mmap_array.flush()
        if self.logger:
            self.logger.log_info(f"Created new mmap at {path}, shape: {shape}")
        return mmap_array
    
    def sync_to_disk(self, case_id=None):
        """
        将内存映射同步到磁盘
        
        参数:
            case_id: 指定案例ID，如果为None则同步所有
        """
        if case_id is not None:
            if case_id in self.open_maps:
                self.open_maps[case_id].flush()
                if self.logger:
                    self.logger.log_info(f"Synced mmap for {case_id} to disk")
        else:
            # 同步所有打开的映射
            for case_id, mmap_array in self.open_maps.items():
                mmap_array.flush()
            if self.logger:
                self.logger.log_info(f"Synced all {len(self.open_maps)} mmaps to disk")
    
    def close(self, case_id=None):
        """
        关闭内存映射
        
        参数:
            case_id: 指定案例ID，如果为None则关闭所有
        """
        if case_id is not None:
            if case_id in self.open_maps:
                self.sync_to_disk(case_id)
                del self.open_maps[case_id]
                if self.logger:
                    self.logger.log_info(f"Closed mmap for {case_id}")
        else:
            # 关闭所有打开的映射
            self.sync_to_disk()
            self.open_maps.clear()
            if self.logger:
                self.logger.log_info("Closed all mmaps")