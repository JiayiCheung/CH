#!/bin/bash
#SBATCH -x paraai-n32-h-01-agent-1,paraai-n32-h-01-agent-4,paraai-n32-h-01-agent-8,paraai-n32-h-01-agent-16,paraai-n32-h-01-agent-17,paraai-n32-h-01-agent-25,paraai-n32-h-01-agent-27,paraai-n32-h-01-agent-28,paraai-n32-h-01-agent-29,paraai-n32-h-01-agent-30,paraai-n32-h-01-agent-31
#SBATCH --job-name=lsfno_training
#SBATCH --output=lsfno_training_%j.out
#SBATCH --error=lsfno_training_%j.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32

# 先加载模块
module purge
module load miniforge3/24.1 compilers/gcc/13.2.0 compilers/cuda/11.8 cudnn/8.8.1.3_cuda11.x

# 激活conda环境 (只使用一种方法)
source activate vessel

# 在激活环境后设置环境变量
export PYTHONNOUSERSITE=1
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=16

# 诊断信息
echo "PYTHONPATH: $PYTHONPATH"
echo "CONDA_PREFIX: $CONDA_PREFIX"
which python

mkdir -p /home/bingxing2/home/scx7as9/run/CSUqx/CH/scripts/vessel_output/test_run

# 启动训练
python train.py \
  --image_dir /home/bingxing2/home/scx7as9/run/CSUqx/DataBase/Task08_HepaticVessel/imagesTr \
  --label_dir /home/bingxing2/home/scx7as9/run/CSUqx/DataBase/Task08_HepaticVessel/labelsTr \
  --output_dir /home/bingxing2/home/scx7as9/run/CSUqx/CH/scripts/vessel_output/test_run \
  --config /home/bingxing2/home/scx7as9/run/CSUqx/CH/configs/default.yaml \
  --batch_size 1 \
  --epochs 10 \
  --lr 1e-4 \
  --val_interval 1 --local_rank 0