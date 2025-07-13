#!/bin/bash
#SBATCH -x paraai-n32-h-01-agent-1,paraai-n32-h-01-agent-4,paraai-n32-h-01-agent-8,paraai-n32-h-01-agent-16,paraai-n32-h-01-agent-17,paraai-n32-h-01-agent-25,paraai-n32-h-01-agent-27,paraai-n32-h-01-agent-28,paraai-n32-h-01-agent-29,paraai-n32-h-01-agent-30,paraai-n32-h-01-agent-31
#SBATCH --job-name=TA-CHNet_training
#SBATCH --output=TA-CHNet_training_%j.out
#SBATCH --error=TA-CHNet_training_%j.err
#SBATCH --nodes=2                # 请求2个计算节点
#SBATCH --ntasks-per-node=4      # 每节点4个任务
#SBATCH --gres=gpu:4             # 每节点4个GPU
#SBATCH --cpus-per-task=8        # 每任务8个CPU核心
#SBATCH --qos=gpugpu             # 多节点GPU作业必须添加此参数
#SBATCH --time=48:00:00          # 最大运行时间48小时

# 加载环境
module purge
module load miniforge3/24.1 compilers/gcc/13.2.0 compilers/cuda/11.8 cudnn/8.8.1.3_cuda11.x
source activate vessel

# 分布式环境变量
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=$((29000 + RANDOM % 1000))
export WORLD_SIZE=$((SLURM_NNODES * SLURM_NTASKS_PER_NODE))
export PYTHONNOUSERSITE=1
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=8


# 性能优化
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=0
export NCCL_SOCKET_IFNAME=ib0

# 路径设置
PROJECT_ROOT="/home/bingxing2/home/scx7as9/run/CSUqx/CH"
DATA_ROOT="/home/bingxing2/home/scx7as9/run/CSUqx/DataBase/Task08_HepaticVessel"
OUTPUT_DIR="/home/bingxing2/home/scx7as9/run/CSUqx/CH/vessel_output/run_${SLURM_JOB_ID}"

mkdir -p ${OUTPUT_DIR}
cd ${PROJECT_ROOT}

# 启动训练
srun --export=ALL python scripts/distributed/distributed_train.py \
  --image_dir ${DATA_ROOT}/imagesTr \
  --label_dir ${DATA_ROOT}/labelsTr \
  --output_dir ${OUTPUT_DIR} \
  --config configs/default.yaml \
  --epochs 20 \
  --num_workers 8 \
  --val_interval 5 \
