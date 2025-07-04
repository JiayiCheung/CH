import numpy as np
from scipy import ndimage
from skimage import morphology

class ComplexityAnalyzer:
    """血管结构复杂度分析器"""
    
    def __init__(self, logger=None):
        """初始化复杂度分析器"""
        self.logger = logger
    
    def compute_complexity(self, label_data):
        """
        计算血管分割标签的复杂度分数
        
        参数:
            label_data: 分割标签数据 [D, H, W]
        
        返回:
            复杂度分数 (0.5-2.0)
        """
        # 确保数据是二值的
        binary_mask = label_data > 0
        
        # 1. 计算前景体素比例
        foreground_ratio = np.mean(binary_mask)
        if foreground_ratio < 1e-6:
            return 0.5  # 极少或没有前景
        
        # 2. 提取骨架和分支点
        try:
            skeleton = morphology.skeletonize(binary_mask)
            if np.sum(skeleton) < 10:  # 骨架太小
                return 0.5
                
            branch_points = self.detect_branch_points(skeleton)
            branch_ratio = np.sum(branch_points) / max(np.sum(skeleton), 1)
        except Exception as e:
            if self.logger:
                self.logger.log_warning(f"Error computing skeleton: {e}")
            branch_ratio = 0
        
        # 3. 计算边界复杂度
        try:
            eroded = ndimage.binary_erosion(binary_mask)
            boundary = binary_mask & (~eroded)
            boundary_ratio = np.sum(boundary) / max(np.sum(binary_mask), 1)
        except Exception as e:
            if self.logger:
                self.logger.log_warning(f"Error computing boundary: {e}")
            boundary_ratio = 0
        
        # 4. 计算连通分量
        try:
            labels, num_components = ndimage.label(binary_mask)
            component_complexity = min(num_components / 10, 1.0)  # 归一化
        except Exception as e:
            if self.logger:
                self.logger.log_warning(f"Error computing components: {e}")
            component_complexity = 0
        
        # 5. 计算欧拉数 (连通数-孔洞数)
        try:
            euler_number = self.compute_euler_number(binary_mask)
            euler_complexity = min(abs(euler_number) / 10, 1.0)  # 归一化
        except Exception as e:
            if self.logger:
                self.logger.log_warning(f"Error computing Euler number: {e}")
            euler_complexity = 0
        
        # 组合不同指标，加权计算总复杂度
        complexity = (
            0.2 * np.log1p(foreground_ratio * 1000) +  # 前景比例 (对数缩放)
            0.3 * branch_ratio * 10 +                  # 分支比例 (放大)
            0.2 * boundary_ratio * 5 +                 # 边界比例
            0.15 * component_complexity +              # 连通分量复杂度
            0.15 * euler_complexity                    # 欧拉特征复杂度
        )
        
        # 归一化到目标范围 [0.5, 2.0]
        normalized = 0.5 + 1.5 * min(complexity, 1.0)
        
        if self.logger:
            self.logger.log_info(f"Complexity: {normalized:.3f} (fg={foreground_ratio:.3f}, "
                          f"br={branch_ratio:.3f}, bound={boundary_ratio:.3f})")
        
        return normalized
    
    def detect_branch_points(self, skeleton):
        """
        检测骨架中的分支点
        
        参数:
            skeleton: 骨架二值图

        返回:
            分支点掩码
        """
        # 初始化结果
        branch_points = np.zeros_like(skeleton)
        
        # 跳过空骨架
        if np.sum(skeleton) == 0:
            return branch_points
        
        # 遍历骨架上的每一点
        if skeleton.ndim == 3:  # 3D骨架
            # 创建3D卷积核，计算邻居数
            kernel = np.ones((3, 3, 3), dtype=np.uint8)
            kernel[1, 1, 1] = 0  # 中心点不算邻居
            
            # 使用卷积计算每个点的邻居数
            neighbors = ndimage.convolve(
                skeleton.astype(np.uint8),
                kernel,
                mode='constant',
                cval=0
            )
            
            # 选择有3个或更多邻居的点作为分支点
            branch_points = (neighbors >= 3) & skeleton
        else:  # 2D骨架
            # 创建2D卷积核
            kernel = np.ones((3, 3), dtype=np.uint8)
            kernel[1, 1] = 0
            
            # 使用卷积计算每个点的邻居数
            neighbors = ndimage.convolve(
                skeleton.astype(np.uint8),
                kernel,
                mode='constant',
                cval=0
            )
            
            # 选择有3个或更多邻居的点作为分支点
            branch_points = (neighbors >= 3) & skeleton
        
        return branch_points
    
    def compute_euler_number(self, binary_mask):
        """
        计算二值图像的欧拉数
        
        参数:
            binary_mask: 二值掩码

        返回:
            欧拉数
        """
        if binary_mask.ndim == 3:
            # 3D欧拉数计算比较复杂，这里用一个近似方法
            # 连通分量数 - 孔洞数 (近似)
            labels, num_components = ndimage.label(binary_mask)
            
            # 反转掩码，计算"孔洞"
            inv_mask = ~binary_mask
            # 移除边界连通的背景
            padded = np.pad(inv_mask, 1, mode='constant', constant_values=1)
            eroded = ndimage.binary_erosion(padded)[1:-1, 1:-1, 1:-1]
            # 标记孔洞
            labels_inv, num_holes = ndimage.label(eroded)
            
            return num_components - num_holes
        else:
            # 2D欧拉数 = 连通分量数 - 孔洞数
            labels, num_components = ndimage.label(binary_mask)
            
            # 反转掩码，计算孔洞
            inv_mask = ~binary_mask
            # 移除边界连通的背景
            padded = np.pad(inv_mask, 1, mode='constant', constant_values=1)
            eroded = ndimage.binary_erosion(padded)[1:-1, 1:-1]
            # 标记孔洞
            labels_inv, num_holes = ndimage.label(eroded)
            
            return num_components - num_holes
    
    def adjust_sampling_density(self, complexity, base_tier1, base_tier2):
        """
        基于复杂度调整采样密度
        
        参数:
            complexity: 复杂度分数 (0.5-2.0)
            base_tier1: 基础Tier-1采样数
            base_tier2: 基础Tier-2采样数
        
        返回:
            调整后的(tier1_samples, tier2_samples)
        """
        # 使用复杂度调整采样数量
        tier1_samples = int(base_tier1 * complexity)
        tier2_samples = int(base_tier2 * complexity)
        
        # 确保最小采样数量
        tier1_samples = max(tier1_samples, 3)
        tier2_samples = max(tier2_samples, 5)
        
        # 确保最大采样数量(防止内存爆炸)
        tier1_samples = min(tier1_samples, base_tier1 * 3)
        tier2_samples = min(tier2_samples, base_tier2 * 3)
        
        if self.logger:
            self.logger.log_info(f"Adjusted sampling: tier1={tier1_samples}, tier2={tier2_samples} "
                         f"(complexity={complexity:.2f})")
        
        return tier1_samples, tier2_samples