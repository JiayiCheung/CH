#!/bin/bash
#SBATCH --job-name=TA-CHNet_training
#SBATCH --output=TA-CHNet_training_%j.out
#SBATCH --error=TA-CHNet_training_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --exclusive
#SBATCH --cpus-per-task=32
#SBATCH --time=48:00:00

module purge
module load miniforge3/24.1 compilers/gcc/13.2.0 compilers/cuda/11.8 cudnn/8.8.1.3_cuda11.x
source activate vessel

export CUDA_VISIBLE_DEVICES=0,1,2,3

cd /home/bingxing2/home/scx7as9/run/CSUqx/CH

torchrun \
    --nproc_per_node=4 \
    train.py \
    --config=config.yaml