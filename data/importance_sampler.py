import numpy as np
from scipy import ndimage
from skimage import morphology, measure

class ImportanceSampler:
    """血管重要性采样器"""
    
    def __init__(self, logger=None):
        """初始化重要性采样器"""
        self.logger = logger
    
    def compute_importance_map(self, label_data, difficulty_map=None):
        """
        计算重要性图，结合几何特征和难度信息
        
        参数:
            label_data: 分割标签数据 [D, H, W]
            difficulty_map: 可选的难度图 [D, H, W]
        
        返回:
            重要性图 [D, H, W]
        """
        # 确保输入是二值的
        binary_mask = label_data > 0
        
        # 初始化重要性图
        importance_map = np.zeros_like(binary_mask, dtype=np.float32)
        
        # 如果没有前景，返回均匀重要性
        if np.sum(binary_mask) == 0:
            importance_map.fill(1.0)
            return importance_map
            
        # 1. 提取骨架
        try:
            skeleton = morphology.skeletonize(binary_mask)
        except Exception as e:
            if self.logger:
                self.logger.log_warning(f"Error skeletonizing: {e}")
            # 如果骨架提取失败，使用原始掩码
            skeleton = binary_mask
        
        # 2. 计算分支点重要性
        branch_map = np.zeros_like(importance_map)
        try:
            branch_points = self.detect_branch_points(skeleton)
            # 扩大分支点影响范围
            branch_map = ndimage.distance_transform_edt(~branch_points)
            branch_map = np.exp(-branch_map / 5.0)  # 指数衰减
        except Exception as e:
            if self.logger:
                self.logger.log_warning(f"Error computing branch importance: {e}")
        
        # 3. 计算曲率重要性
        curvature_map = np.zeros_like(importance_map)
        try:
            curvature = self.compute_curvature(skeleton)
            # 归一化曲率
            curvature_norm = np.zeros_like(skeleton, dtype=np.float32)
            curvature_norm[skeleton] = curvature
            # 扩大曲率影响范围
            curvature_map = ndimage.distance_transform_edt(~skeleton)
            curvature_map = np.exp(-curvature_map / 3.0)  # 指数衰减
            # 加权曲率
            curvature_map = curvature_map * np.maximum(curvature_norm, 0.2)
        except Exception as e:
            if self.logger:
                self.logger.log_warning(f"Error computing curvature importance: {e}")
        
        # 4. 计算半径变化重要性
        radius_map = np.zeros_like(importance_map)
        try:
            # 距离变换获取半径
            dist = ndimage.distance_transform_edt(binary_mask)
            # 计算骨架上的半径
            skeleton_radius = np.zeros_like(skeleton, dtype=np.float32)
            skeleton_radius[skeleton] = dist[skeleton]
            # 计算半径梯度
            radius_gradient = self.compute_radius_gradient(skeleton, skeleton_radius)
            # 归一化半径梯度
            radius_norm = np.zeros_like(skeleton, dtype=np.float32)
            radius_norm[skeleton] = radius_gradient
            # 扩大半径梯度影响范围
            radius_map = ndimage.distance_transform_edt(~skeleton)
            radius_map = np.exp(-radius_map / 3.0)  # 指数衰减
            # 加权半径梯度
            radius_map = radius_map * np.maximum(radius_norm, 0.2)
        except Exception as e:
            if self.logger:
                self.logger.log_warning(f"Error computing radius importance: {e}")
        
        # 5. 集成几何特征
        # 权重分配: 分支点 40%, 曲率 30%, 半径变化 30%
        geometry_importance = (
            0.4 * branch_map +
            0.3 * curvature_map +
            0.3 * radius_map
        )
        
        # 6. 集成难度信息 (如果有)
        if difficulty_map is not None:
            # 结合几何重要性和难度图，难度占比40%
            importance_map = 0.6 * geometry_importance + 0.4 * difficulty_map
        else:
            importance_map = geometry_importance
        
        # 确保重要性非负，并有最小值
        importance_map = np.maximum(importance_map, 0.1)
        
        # 7. 确保骨架点必然被采样
        importance_map[skeleton] = np.maximum(importance_map[skeleton], 0.5)
        
        # 8. 归一化重要性图
        if np.max(importance_map) > 0:
            importance_map = importance_map / np.max(importance_map)
        else:
            importance_map.fill(1.0)
        
        return importance_map
    
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
    
    def compute_curvature(self, skeleton):
        """
        计算骨架的曲率
        
        参数:
            skeleton: 骨架二值图

        返回:
            骨架点的曲率值数组
        """
        # 跳过空骨架
        if np.sum(skeleton) == 0:
            return np.array([])
        
        # 提取骨架点坐标
        coords = np.array(np.where(skeleton)).T
        
        # 如果点太少，返回默认曲率
        if len(coords) < 5:
            return np.ones(len(coords))
        
        # 计算每个点的曲率
        curvature = np.ones(len(coords))
        
        try:
            # 对于3D骨架，使用主曲率近似
            if skeleton.ndim == 3:
                # 使用结构张量分析局部曲率
                for i, (z, y, x) in enumerate(coords):
                    # 获取局部邻域
                    z_min, z_max = max(0, z-2), min(skeleton.shape[0], z+3)
                    y_min, y_max = max(0, y-2), min(skeleton.shape[1], y+3)
                    x_min, x_max = max(0, x-2), min(skeleton.shape[2], x+3)
                    
                    neighborhood = skeleton[z_min:z_max, y_min:y_max, x_min:x_max]
                    
                    # 如果邻域太小，跳过
                    if neighborhood.size <= 1:
                        continue
                    
                    # 计算局部协方差矩阵
                    local_coords = np.array(np.where(neighborhood)).T
                    if len(local_coords) < 4:
                        continue
                    
                    # 中心化坐标
                    centered = local_coords - np.mean(local_coords, axis=0)
                    
                    # 计算协方差矩阵
                    cov = np.cov(centered.T)
                    
                    # 计算特征值
                    try:
                        eigvals = np.linalg.eigvalsh(cov)
                        # 使用特征值比作为曲率度量
                        if eigvals[0] > 1e-6:
                            curvature[i] = eigvals[-1] / eigvals[0]
                        else:
                            curvature[i] = 1.0
                    except np.linalg.LinAlgError:
                        curvature[i] = 1.0
            else:
                # 2D曲率计算
                # 使用图像处理库提取线段
                labeled_skeleton, num_segments = ndimage.label(skeleton)
                
                for segment_id in range(1, num_segments + 1):
                    # 提取当前线段
                    segment = labeled_skeleton == segment_id
                    segment_coords = np.array(np.where(segment)).T
                    
                    # 跳过太短的线段
                    if len(segment_coords) < 5:
                        continue
                    
                    # 使用多项式拟合计算曲率
                    try:
                        # 排序坐标
                        segment_coords = self._sort_coords(segment_coords)
                        # 使用参数化拟合
                        t = np.arange(len(segment_coords))
                        # 拟合y坐标
                        poly_y = np.polyfit(t, segment_coords[:, 0], 3)
                        # 拟合x坐标
                        poly_x = np.polyfit(t, segment_coords[:, 1], 3)
                        
                        # 计算导数
                        poly_y_d1 = np.polyder(poly_y, 1)
                        poly_x_d1 = np.polyder(poly_x, 1)
                        poly_y_d2 = np.polyder(poly_y, 2)
                        poly_x_d2 = np.polyder(poly_x, 2)
                        
                        # 计算曲率: |x'y'' - y'x''| / (x'^2 + y'^2)^(3/2)
                        for idx, t_val in enumerate(t):
                            x_d1 = np.polyval(poly_x_d1, t_val)
                            y_d1 = np.polyval(poly_y_d1, t_val)
                            x_d2 = np.polyval(poly_x_d2, t_val)
                            y_d2 = np.polyval(poly_y_d2, t_val)
                            
                            num = abs(x_d1 * y_d2 - y_d1 * x_d2)
                            denom = (x_d1**2 + y_d1**2)**(1.5)
                            
                            if denom > 1e-6:
                                k = num / denom
                            else:
                                k = 0
                            
                            # 找到对应的索引
                            coord = tuple(segment_coords[idx])
                            orig_idx = np.where((coords == coord).all(axis=1))[0]
                            if len(orig_idx) > 0:
                                curvature[orig_idx[0]] = k
                    except Exception as e:
                        if self.logger:
                            self.logger.log_warning(f"Error computing 2D curvature: {e}")
        except Exception as e:
            if self.logger:
                self.logger.log_warning(f"Error in curvature computation: {e}")
            # 出错时返回默认曲率
            curvature = np.ones(len(coords))
        
        # 归一化曲率到[0,1]范围
        if len(curvature) > 0:
            max_curve = np.max(curvature)
            if max_curve > 0:
                curvature = curvature / max_curve
        
        return curvature
    
    def _sort_coords(self, coords):
        """
        排序坐标点，使其沿线段顺序排列
        
        参数:
            coords: 坐标点数组 [[y1,x1], [y2,x2], ...]
            
        返回:
            排序后的坐标数组
        """
        # 如果点太少，直接返回
        if len(coords) < 3:
            return coords
            
        # 使用最小生成树或最近邻排序
        # 这里使用简单的最近邻方法
        ordered = [coords[0]]
        remaining = coords[1:].copy()
        
        while len(remaining) > 0:
            last = ordered[-1]
            # 计算到所有剩余点的距离
            dists = np.sum((remaining - last)**2, axis=1)
            # 找到最近的点
            idx = np.argmin(dists)
            # 添加到有序列表
            ordered.append(remaining[idx])
            # 从剩余列表中移除
            remaining = np.delete(remaining, idx, axis=0)
        
        return np.array(ordered)
    
    def compute_radius_gradient(self, skeleton, radius):
        """
        计算骨架上的半径梯度
        
        参数:
            skeleton: 骨架二值图
            radius: 骨架上的半径值
            
        返回:
            骨架上的半径梯度
        """
        # 初始化梯度图
        gradient = np.zeros_like(skeleton, dtype=np.float32)
        
        # 如果骨架为空，返回空梯度
        if np.sum(skeleton) == 0:
            return gradient
        
        try:
            # 使用Sobel算子计算梯度
            if skeleton.ndim == 3:
                # 将半径值填充到3D体积
                radius_volume = np.zeros_like(skeleton, dtype=np.float32)
                radius_volume[skeleton] = radius[skeleton]
                
                # 使用高斯平滑减少噪声
                radius_smooth = ndimage.gaussian_filter(radius_volume, sigma=1.0)
                
                # 计算梯度幅度
                grad_z = ndimage.sobel(radius_smooth, axis=0, mode='reflect')
                grad_y = ndimage.sobel(radius_smooth, axis=1, mode='reflect')
                grad_x = ndimage.sobel(radius_smooth, axis=2, mode='reflect')
                
                # 计算梯度幅度
                gradient = np.sqrt(grad_z**2 + grad_y**2 + grad_x**2)
            else:
                # 2D情况
                radius_image = np.zeros_like(skeleton, dtype=np.float32)
                radius_image[skeleton] = radius[skeleton]
                
                # 使用高斯平滑
                radius_smooth = ndimage.gaussian_filter(radius_image, sigma=1.0)
                
                # 计算梯度
                grad_y = ndimage.sobel(radius_smooth, axis=0, mode='reflect')
                grad_x = ndimage.sobel(radius_smooth, axis=1, mode='reflect')
                
                # 计算梯度幅度
                gradient = np.sqrt(grad_y**2 + grad_x**2)
            
            # 只保留骨架上的梯度
            gradient = gradient * skeleton
            
            # 归一化梯度
            max_grad = np.max(gradient)
            if max_grad > 0:
                gradient = gradient / max_grad
        except Exception as e:
            if self.logger:
                self.logger.log_warning(f"Error computing radius gradient: {e}")
        
        return gradient
    
    def importance_based_sampling(self, coords, importance_values, max_samples):
        """
        基于重要性进行加权采样
        
        参数:
            coords: 坐标点数组 [[z,y,x], ...]
            importance_values: 对应的重要性值数组
            max_samples: 最大采样数量
            
        返回:
            采样后的坐标数组
        """
        # 如果点数少于最大采样数，全部返回
        if len(coords) <= max_samples:
            return coords
        
        # 确保重要性值是正的
        imp_values = np.maximum(importance_values, 1e-6)
        
        # 归一化为概率
        probs = imp_values / np.sum(imp_values)
        
        # 加权随机采样
        try:
            indices = np.random.choice(
                len(coords), size=max_samples, replace=False, p=probs
            )
            return coords[indices]
        except Exception as e:
            if self.logger:
                self.logger.log_warning(f"Error in importance sampling: {e}, using random sampling")
            # 出错时退化为随机采样
            indices = np.random.choice(len(coords), size=max_samples, replace=False)
            return coords[indices]