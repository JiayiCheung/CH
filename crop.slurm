#!/bin/bash
#SBATCH --job-name=Vessel_Preprocessing
#SBATCH --output=Vessel_Preprocessing_%j.out
#SBATCH --error=Vessel_Preprocessing_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00

module purge
module load miniforge3/24.1 compilers/gcc/13.2.0 compilers/cuda/11.8 cudnn/8.8.1.3_cuda11.x
source activate vessel

cd /home/bingxing2/home/scx7as9/run/CSUqx/CH

# 创建预处理输出目录
mkdir -p data/preprocess




python -m data.crop \
  --images_dir /home/bingxing2/home/scx7as9/run/CSUqx/DataBase/Task08_HepaticVessel/imagesTr \
  --labels_dir /home/bingxing2/home/scx7as9/run/CSUqx/DataBase/Task08_HepaticVessel/labelsTr \
  --output_dir /home/bingxing2/home/scx7as9/run/CSUqx/CH/data/preprocess \
  --num_workers 4