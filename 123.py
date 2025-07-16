#!/usr/bin/env python3
"""
测试修正后的柱坐标映射
验证r=0奇异性处理是否正确
"""

import torch
import numpy as np
from models.ch_branch.cylindrical_mapping import CylindricalMapping


def create_test_volume():
	"""创建测试用的3D体积 - 模拟中心血管"""
	size = 32
	volume = torch.zeros(1, 1, size, size, size)
	
	# 创建一个沿z轴变化的中心血管
	center = size // 2
	for z in range(size):
		# 血管半径沿z轴a变化
		radius = 3 + 2 * np.sin(z * np.pi / size)
		
		for i in range(size):
			for j in range(size):
				dist = np.sqrt((i - center) ** 2 + (j - center) ** 2)
				if dist <= radius:
					# 中心血管强度沿z轴变化
					intensity = 1.0 - 0.3 * (z / size)
					volume[0, 0, z, i, j] = intensity
	
	return volume


def test_r0_handling():
	"""测试r=0处理的正确性"""
	print("🔧 测试柱坐标映射的r=0修正...")
	
	# 创建测试数据
	test_vol = create_test_volume()
	print(f"✅ 创建测试体积: {test_vol.shape}")
	
	# 创建映射器
	mapper = CylindricalMapping(r_samples=16, theta_samples=36, z_samples=32)
	
	# 测试正向映射
	cyl_vol = mapper.cartesian_to_cylindrical(test_vol)
	print(f"✅ 柱坐标映射完成: {cyl_vol.shape}")
	
	# 检查r=0处的处理
	B, C, R, T, Z = cyl_vol.shape
	
	# 验证r=0层的物理正确性
	r0_layer = cyl_vol[0, 0, 0, :, :]  # [T, Z]
	
	print(f"📊 r=0层分析:")
	print(f"   形状: {r0_layer.shape}")
	
	# 检查每个z切片的theta一致性
	theta_consistency_errors = []
	for z in range(Z):
		z_slice = r0_layer[:, z]  # 所有theta在z处的值
		std_dev = z_slice.std().item()
		theta_consistency_errors.append(std_dev)
	
	max_inconsistency = max(theta_consistency_errors)
	avg_inconsistency = np.mean(theta_consistency_errors)
	
	print(f"   theta一致性 - 最大偏差: {max_inconsistency:.6f}")
	print(f"   theta一致性 - 平均偏差: {avg_inconsistency:.6f}")
	
	# 检查z方向的变化保持
	z_center_values = r0_layer[0, :].cpu().numpy()  # 第一个theta的所有z值
	z_variation = np.std(z_center_values)
	
	print(f"   z方向变化保持: {z_variation:.6f}")
	
	# 测试逆映射
	reconstructed = mapper.cylindrical_to_cartesian(cyl_vol, test_vol.shape[2:])
	print(f"✅ 逆映射完成: {reconstructed.shape}")
	
	# 计算重构误差
	mse = torch.nn.functional.mse_loss(reconstructed, test_vol).item()
	print(f"📈 重构MSE误差: {mse:.6f}")
	
	# 判断测试结果
	success = True
	if max_inconsistency > 1e-5:
		print("❌ 警告: r=0处theta一致性较差")
		success = False
	
	if z_variation < 1e-3:
		print("❌ 警告: z方向变化丢失")
		success = False
	
	if mse > 0.1:
		print("❌ 警告: 重构误差过大")
		success = False
	
	if success:
		print("🎉 所有测试通过！r=0修正工作正常")
	else:
		print("⚠️  部分测试未通过，需要进一步调试")
	
	return success


def test_integration_with_ch_branch():
	"""测试与CH分支的集成"""
	print("\n🔗 测试与CH分支的集成...")
	
	try:
		from models.ch_branch import CHBranch
		
		# 创建CH分支
		ch_branch = CHBranch(
			max_n=2, max_k=3, max_l=4,
			cylindrical_dims=(16, 36, 16)  # 小一点测试更快
		)
		
		# 创建测试输入
		test_input = torch.randn(1, 1, 32, 32, 32)
		
		# 前向传播测试
		output = ch_branch(test_input)
		print(f"✅ CH分支集成测试通过: {test_input.shape} -> {output.shape}")
		
		return True
	
	except Exception as e:
		print(f"❌ CH分支集成测试失败: {e}")
		return False


if __name__ == "__main__":
	print("🚀 开始测试柱坐标映射修正...")
	
	# 基础功能测试
	basic_success = test_r0_handling()
	
	# 集成测试
	integration_success = test_integration_with_ch_branch()
	
	print(f"\n📋 测试总结:")
	print(f"   基础功能: {'✅ 通过' if basic_success else '❌ 失败'}")
	print(f"   集成测试: {'✅ 通过' if integration_success else '❌ 失败'}")
	
	if basic_success and integration_success:
		print("🎉 所有测试完成！可以继续下一步开发")
	else:
		print("⚠️  存在问题，建议先调试再继续")