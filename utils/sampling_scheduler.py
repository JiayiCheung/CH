import numpy as np
from pathlib import Path

class SamplingScheduler:
    """采样调度器，控制训练过程中采样策略的变化"""
    
    def __init__(self,
                 base_tier1=10,
                 base_tier2=30,
                 max_tier1=20,
                 max_tier2=60,
                 warmup_epochs=5,
                 enable_hard_mining=True,
                 enable_adaptive_density=True,
                 enable_importance_sampling=True,
                 logger=None):
        """
        初始化采样调度器
        
        参数:
            base_tier1: 基础Tier-1采样数量
            base_tier2: 基础Tier-2采样数量
            max_tier1: 最大Tier-1采样数量
            max_tier2: 最大Tier-2采样数量
            warmup_epochs: 预热轮数，在此期间采用均匀采样
            enable_hard_mining: 是否启用硬样本挖掘
            enable_adaptive_density: 是否启用自适应采样密度
            enable_importance_sampling: 是否启用重要性采样
            logger: 日志记录器实例
        """
        self.base_tier1 = base_tier1
        self.base_tier2 = base_tier2
        self.max_tier1 = max_tier1
        self.max_tier2 = max_tier2
        self.warmup_epochs = warmup_epochs
        
        self.enable_hard_mining = enable_hard_mining
        self.enable_adaptive_density = enable_adaptive_density
        self.enable_importance_sampling = enable_importance_sampling
        self.logger = logger
        
        # 当前状态
        self.current_epoch = 0
        self.hard_mining_weight = 0.0
        self.importance_weight = 0.0
        
        if self.logger:
            self.logger.log_info(f"Initialized SamplingScheduler: "
                       f"base_tier1={base_tier1}, base_tier2={base_tier2}, "
                       f"warmup_epochs={warmup_epochs}")
    
    def update(self, epoch):
        """
        更新调度器状态
        
        参数:
            epoch: 当前训练轮数
        """
        self.current_epoch = epoch
        
        # 计算硬样本挖掘权重
        if self.enable_hard_mining:
            if epoch < self.warmup_epochs:
                self.hard_mining_weight = 0.0
            else:
                # 在预热后逐渐增加权重，最大到0.4
                progress = min(1.0, (epoch - self.warmup_epochs) / 10)
                self.hard_mining_weight = 0.4 * progress
        else:
            self.hard_mining_weight = 0.0
        
        # 计算重要性采样权重
        if self.enable_importance_sampling:
            if epoch < self.warmup_epochs:
                self.importance_weight = 0.0
            else:
                # 在预热后立即启用，权重为0.6
                self.importance_weight = 0.6
        else:
            self.importance_weight = 0.0
        
        if self.logger:
            self.logger.log_info(f"Updated sampling weights: hard_mining={self.hard_mining_weight:.2f}, "
                       f"importance={self.importance_weight:.2f}")
    
    def get_tier_sampling_params(self, complexity=1.0):
        """
        获取当前的采样参数
        
        参数:
            complexity: 案例复杂度
        
        返回:
            采样参数字典
        """
        # 基础采样数量
        tier1_samples = self.base_tier1
        tier2_samples = self.base_tier2
        
        # 应用自适应密度
        if self.enable_adaptive_density and self.current_epoch >= self.warmup_epochs:
            # 调整采样数量
            tier1_scale = 1.0 + (complexity - 1.0) * 0.5  # 调整范围为[0.5, 1.5]
            tier2_scale = 1.0 + (complexity - 1.0) * 1.0  # 调整范围为[0, 2.0]
            
            tier1_samples = int(tier1_samples * tier1_scale)
            tier2_samples = int(tier2_samples * tier2_scale)
            
            # 确保采样数量在合理范围内
            tier1_samples = max(3, min(tier1_samples, self.max_tier1))
            tier2_samples = max(5, min(tier2_samples, self.max_tier2))
        
        return {
            'tier1_samples': tier1_samples,
            'tier2_samples': tier2_samples,
            'hard_mining_weight': self.hard_mining_weight,
            'importance_weight': self.importance_weight
        }
    
    def get_progress_stats(self):
        """
        获取当前进度统计
        
        返回:
            进度统计字典
        """
        return {
            'epoch': self.current_epoch,
            'warmup_progress': min(1.0, self.current_epoch / self.warmup_epochs),
            'hard_mining_weight': self.hard_mining_weight,
            'importance_weight': self.importance_weight
        }