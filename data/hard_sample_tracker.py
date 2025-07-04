"""
硬样本跟踪模块，用于识别和跟踪难分割区域
"""
import numpy as np
import torch
import torch.nn.functional as F
import logging
from pathlib import Path
from .mmap_utils import MMapManager

logger = logging.getLogger(__name__)

class HardSampleTracker:
    """硬样本跟踪器，管理和更新难度图"""
    
    def __init__(self, base_dir="difficulty_maps", alpha=0.7, device='cpu'):
        """
        初始化硬样本跟踪器
        
        参数:
            base_dir: 存储难度图的目录
            alpha: 历史信息权重 (0-1)
            device: 计算设备
        """
        self.mmap_manager = MMapManager(base_dir)
        self.alpha = alpha
        self.device = device
        
        # 缓存难度图和维度信息
        self.case_dims = {}  # 存储案例的体素维度
    
    def initialize_case(self, case_id, shape):
        """
        初始化案例的难度图
        
        参数:
            case_id: 案例ID
            shape: 数据形状 [D, H, W]
        """
        # 记录案例维度
        self.case_dims[case_id] = shape
        
        # 创建或加载难度图
        self.mmap_manager.create_or_load(case_id, shape, initial_value=0.5)
        logger.debug(f"Initialized difficulty map for {case_id}, shape: {shape}")
    
    def get_difficulty_map(self, case_id):
        """
        获取案例的难度图
        
        参数:
            case_id: 案例ID
            
        返回:
            难度图数组
        """
        if case_id not in self.case_dims:
            logger.warning(f"Case {case_id} not initialized")
            return None
        
        # 获取难度图
        difficulty_map = self.mmap_manager.create_or_load(
            case_id, self.case_dims[case_id]
        )
        
        return difficulty_map
    
    def update_difficulty(self, case_id, prediction, target):
        """
        基于分割结果更新难度图
        
        参数:
            case_id: 案例ID
            prediction: 模型预测 [C, D, H, W]
            target: 真实标签 [D, H, W]
            
        返回:
            更新后的难度图
        """
        # 获取难度图
        difficulty_map = self.get_difficulty_map(case_id)
        if difficulty_map is None:
            logger.warning(f"Cannot update difficulty for {case_id}: not initialized")
            return None
        
        # 确保张量在同一设备上
        if torch.is_tensor(prediction):
            prediction = prediction.detach().cpu().numpy()
        if torch.is_tensor(target):
            target = target.detach().cpu().numpy()
        
        # 计算性能图 (1-performance 即为难度)
        performance_map = self.compute_performance_map(prediction, target)
        
        # 计算新难度
        difficulty = 1.0 - performance_map
        
        # 更新难度图 (使用指数移动平均)
        difficulty_map[:] = self.alpha * difficulty_map + (1 - self.alpha) * difficulty
        
        # 确保值在有效范围内
        np.clip(difficulty_map, 0.01, 0.99, out=difficulty_map)
        
        # 同步到磁盘
        self.mmap_manager.sync_to_disk(case_id)
        
        logger.debug(f"Updated difficulty map for {case_id}, "
                    f"avg difficulty: {np.mean(difficulty_map):.3f}")
        
        return difficulty_map
    
    def compute_performance_map(self, prediction, target):
        """
        计算局部性能图
        
        参数:
            prediction: 模型预测 [C, D, H, W]
            target: 真实标签 [D, H, W]
            
        返回:
            性能图 [D, H, W]，值域[0,1]，值越高表示性能越好
        """
        # 确保输入形状正确
        if prediction.ndim == 4 and prediction.shape[0] == 1:
            # 单通道情况
            prediction = prediction[0]  # [D, H, W]
        
        # 二值化预测 (如果是概率)
        if prediction.dtype == np.float32 or prediction.dtype == np.float64:
            pred_binary = (prediction > 0.5).astype(np.float32)
        else:
            pred_binary = (prediction > 0).astype(np.float32)
        
        # 确保目标是二值的
        if target.dtype == np.float32 or target.dtype == np.float64:
            target_binary = (target > 0.5).astype(np.float32)
        else:
            target_binary = (target > 0).astype(np.float32)
        
        # 创建性能图
        performance_map = np.zeros_like(target_binary, dtype=np.float32)
        
        try:
            # 使用局部Dice系数作为性能度量
            # 定义局部窗口大小
            window_size = min(16, *target.shape)
            
            # 使用卷积计算局部统计量
            # 转换为PyTorch张量以使用F.conv3d
            pred_tensor = torch.from_numpy(pred_binary).unsqueeze(0).unsqueeze(0)
            target_tensor = torch.from_numpy(target_binary).unsqueeze(0).unsqueeze(0)
            
            # 创建卷积核 (全1)
            kernel = torch.ones((1, 1, window_size, window_size, window_size))
            
            # 计算局部统计量
            pred_local_sum = F.conv3d(
                pred_tensor, kernel, padding=window_size//2
            ).squeeze().numpy()
            
            target_local_sum = F.conv3d(
                target_tensor, kernel, padding=window_size//2
            ).squeeze().numpy()
            
            intersection = F.conv3d(
                pred_tensor * target_tensor, kernel, padding=window_size//2
            ).squeeze().numpy()
            
            # 计算局部Dice系数
            smooth = 1e-5
            local_dice = (2.0 * intersection + smooth) / (
                pred_local_sum + target_local_sum + smooth
            )
            
            # 使用局部Dice作为性能度量
            performance_map = local_dice
        except Exception as e:
            logger.warning(f"Error computing local performance: {e}")
            # 出错时使用全局性能
            # 计算全局Dice
            intersection = np.sum(pred_binary * target_binary)
            dice = (2.0 * intersection) / (np.sum(pred_binary) + np.sum(target_binary) + 1e-5)
            
            # 使用全局Dice填充性能图
            performance_map.fill(dice)
        
        return performance_map
    
    def sync_difficulty_maps(self):
        """同步所有难度图到磁盘"""
        self.mmap_manager.sync_to_disk()
    
    def close(self):
        """关闭硬样本跟踪器，释放资源"""
        self.mmap_manager.close()
        logger.debug("Closed hard sample tracker")