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

# 加载必要模块
module purge
module load miniforge3/24.1 compilers/gcc/13.2.0 compilers/cuda/11.8 cudnn/8.8.1.3_cuda11.x

# 激活conda环境
source activate vessel

# 设置分布式训练环境变量
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=29501
export WORLD_SIZE=$SLURM_NTASKS
export PYTHONNOUSERSITE=1
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=8

# 设置NCCL通信参数
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=0
export NCCL_SOCKET_IFNAME=^docker0,lo

# 打印作业信息
echo "作业ID: $SLURM_JOB_ID"
echo "开始时间: $(date)"
echo "节点列表: $(scontrol show hostnames $SLURM_JOB_NODELIST)"
echo "主节点: $MASTER_ADDR"
echo "GPU总数: $((SLURM_NNODES * 4))"
echo "进程总数: $WORLD_SIZE"

# 创建输出目录
OUTPUT_DIR="/home/bingxing2/home/scx7as9/run/CSUqx/CH/scripts/distributed/vessel_output/test_run"
mkdir -p ${OUTPUT_DIR}

# 创建检查点目录
CHECKPOINT_DIR="${OUTPUT_DIR}/checkpoints"
mkdir -p ${CHECKPOINT_DIR}



# 使用srun启动分布式训练
srun --export=ALL \
  bash -c "
    export RANK=\$SLURM_PROCID
    export LOCAL_RANK=\$SLURM_LOCALID
    python distributed_train.py \
      --image_dir /home/bingxing2/home/scx7as9/run/CSUqx/DataBase/Task08_HepaticVessel/imagesTr \
      --label_dir /home/bingxing2/home/scx7as9/run/CSUqx/DataBase/Task08_HepaticVessel/labelsTr \
      --output_dir $OUTPUT_DIR \
      --config /home/bingxing2/home/scx7as9/run/CSUqx/CH/configs/pipeline.yaml \
      --epochs 10 \
      --lr 1e-4 \
      --num_workers 8 \
      --val_interval 1
  "



