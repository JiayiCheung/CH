#!/usr/bin/env python3
"""
统一训练入口 - 肝脏血管分割
简洁版本：找到分布式训练脚本并用torchrun启动
"""
import os
import sys
import subprocess
import argparse
from pathlib import Path


def parse_args():
	"""解析命令行参数"""
	parser = argparse.ArgumentParser(description='Liver Vessel Segmentation Training')
	
	# 数据参数
	parser.add_argument('--image_dir', type=str, required=True)
	parser.add_argument('--label_dir', type=str, required=True)
	parser.add_argument('--output_dir', type=str, default='./output')
	parser.add_argument('--config', type=str, default='configs/default.yaml')
	
	# 训练参数
	parser.add_argument('--batch_size', type=int, default=1)
	parser.add_argument('--epochs', type=int, default=100)
	parser.add_argument('--lr', type=float, default=1e-4)
	parser.add_argument('--num_workers', type=int, default=4)
	parser.add_argument('--resume', type=str)
	parser.add_argument('--amp', action='store_true')
	parser.add_argument('--val_interval', type=int, default=5)
	
	# 分布式参数（可选，主要由SLURM环境变量提供）
	parser.add_argument('--nodes', type=int, default=1)
	parser.add_argument('--gpus_per_node', type=int, default=4)
	parser.add_argument('--node_rank', type=int, default=0)
	
	return parser.parse_args()


def find_training_script():
	"""找到分布式训练脚本"""
	script_dir = Path(__file__).parent
	train_script = script_dir / "scripts" / "distributed" / "distributed_train.py"
	
	if not train_script.exists():
		raise FileNotFoundError(f"Training script not found: {train_script}")
	
	return train_script


def build_command(args):
	"""构建torchrun命令"""
	train_script = find_training_script()
	
	# 使用现代torchrun，让它自己处理环境变量
	cmd = [
		sys.executable, "-m", "torch.distributed.run",
		"--nproc_per_node", str(args.gpus_per_node),
		"--nnodes", str(args.nodes),
		"--node_rank", str(args.node_rank),
		str(train_script)
	]
	
	# 添加训练参数
	cmd.extend([
		"--image_dir", args.image_dir,
		"--label_dir", args.label_dir,
		"--output_dir", args.output_dir,
		"--config", args.config,
		"--batch_size", str(args.batch_size),
		"--epochs", str(args.epochs),
		"--lr", str(args.lr),
		"--num_workers", str(args.num_workers),
		"--val_interval", str(args.val_interval)
	])
	
	if args.resume:
		cmd.extend(["--resume", args.resume])
	
	
	
	return cmd


def main():
	"""主函数"""
	try:
		args = parse_args()
		
		# 基本验证
		if not Path(args.image_dir).exists():
			raise FileNotFoundError(f"Image directory not found: {args.image_dir}")
		if not Path(args.label_dir).exists():
			raise FileNotFoundError(f"Label directory not found: {args.label_dir}")
		
		# 构建并执行命令
		cmd = build_command(args)
		
		print("🚀 Starting distributed training...")
		print(f"Command: {' '.join(cmd)}")
		
		# 直接执行，让torchrun处理分布式细节
		result = subprocess.run(cmd, check=True)
		
		print("✅ Training completed successfully!")
	
	except FileNotFoundError as e:
		print(f"❌ {e}")
		sys.exit(1)
	except subprocess.CalledProcessError as e:
		print(f"❌ Training failed with exit code: {e.returncode}")
		sys.exit(e.returncode)
	except KeyboardInterrupt:
		print("⚠️ Training interrupted")
		sys.exit(1)


if __name__ == "__main__":
	main()