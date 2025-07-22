#!/bin/bash
#SBATCH -x paraai-n32-h-01-agent-1,paraai-n32-h-01-agent-4,paraai-n32-h-01-agent-8,paraai-n32-h-01-agent-16,paraai-n32-h-01-agent-17,paraai-n32-h-01-agent-25,paraai-n32-h-01-agent-27,paraai-n32-h-01-agent-28,paraai-n32-h-01-agent-29,paraai-n32-h-01-agent-30,paraai-n32-h-01-agent-31
#SBATCH --job-name=TA-CHNet_training
#SBATCH --output=TA-CHNet_training_%j.out
#SBATCH --error=TA-CHNet_training_%j.err
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4             # 每节点仍然申请4个GPU
#SBATCH --cpus-per-task=8        # 每任务8个CPU核心
#SBATCH --qos=gpugpu             # 多节点GPU作业必须添加此参数
#SBATCH --time=48:00:00          # 最大运行时间48小时

echo "=========================================="
echo "🚀 TA-CHNet 肝脏血管分割模型分布式训练"
echo "=========================================="
echo "作业ID: $SLURM_JOB_ID"
echo "开始时间: $(date)"

# =============================================================================
# 环境设置
# =============================================================================
echo "📦 加载环境模块..."
module purge
module load miniforge3/24.1 compilers/gcc/13.2.0 compilers/cuda/11.8 cudnn/8.8.1.3_cuda11.x

echo "🐍 激活conda环境..."
source activate vessel

echo "Python版本: $(python --version)"
echo "PyTorch版本: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA版本: $(python -c 'import torch; print(torch.version.cuda)')"

# =============================================================================
# 分布式环境变量设置 (基于测试结果优化)
# =============================================================================
echo "⚙️ 设置分布式环境变量..."

export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=29501
export WORLD_SIZE=$SLURM_NTASKS
export PYTHONNOUSERSITE=1
export WORLD_SIZE=8


# CPU线程控制
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8

# =============================================================================
# NCCL网络配置 (使用测试验证的成功配置)
# =============================================================================
echo "🌐 配置NCCL网络通信（使用100G RoCE高速网络）..."

# ✅ 按N32-H分区手册要求配置
export NCCL_DEBUG=WARN
export NCCL_IB_DISABLE=0                    # 启用InfiniBand接口访问RoCE
export NCCL_IB_HCA=mlx5_0,mlx5_2           # 手册指定的Mellanox网卡
export NCCL_IB_GID_INDEX=3                 # 手册指定的GID索引
export NCCL_TIMEOUT=1800                   # 30分钟超时

# RoCE RDMA优化设置
export NCCL_NET_GDR_LEVEL=2                # 启用GPU Direct RDMA
export NCCL_IB_TIMEOUT=23                  # IB接口超时设置
export NCCL_TREE_THRESHOLD=0               # 强制使用树形算法
export NCCL_P2P_LEVEL=NVL                  # 节点内NVLink通信


# 内存优化
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
export CUDA_LAUNCH_BLOCKING=0



echo "网络配置信息:"
echo "  MASTER_ADDR: $MASTER_ADDR"
echo "  MASTER_PORT: $MASTER_PORT"
echo "  WORLD_SIZE: $WORLD_SIZE"
echo "  NCCL_IB_DISABLE: $NCCL_IB_DISABLE"
echo "  NCCL_IB_HCA: $NCCL_IB_HCA"
echo "  NCCL_IB_GID_INDEX: $NCCL_IB_GID_INDEX"
echo "  NCCL_NET: $NCCL_NET"

# =============================================================================
# 路径和目录设置
# =============================================================================
echo "📁 设置项目路径..."

PROJECT_ROOT="/home/bingxing2/home/scx7as9/run/CSUqx/CH"
DATA_ROOT="/home/bingxing2/home/scx7as9/run/CSUqx/DataBase/Task08_HepaticVessel"
OUTPUT_DIR="/home/bingxing2/home/scx7as9/run/CSUqx/CH/scripts/distributed/vessel_output/run_${SLURM_JOB_ID}"
CHECKPOINT_DIR="${OUTPUT_DIR}/checkpoints"
LOG_DIR="${OUTPUT_DIR}/logs"

# 创建必要目录
mkdir -p ${OUTPUT_DIR}
mkdir -p ${CHECKPOINT_DIR}
mkdir -p ${LOG_DIR}

echo "目录设置:"
echo "  项目根目录: $PROJECT_ROOT"
echo "  数据目录: $DATA_ROOT"
echo "  输出目录: $OUTPUT_DIR"
echo "  检查点目录: $CHECKPOINT_DIR"

cd $PROJECT_ROOT

# =============================================================================
# 硬件信息收集
# =============================================================================
echo "💻 硬件信息:"
echo "节点列表: $(scontrol show hostnames $SLURM_JOB_NODELIST)"
echo "主节点: $MASTER_ADDR"
echo "节点数: $SLURM_NNODES"
echo "每节点GPU数: 4"
echo "GPU总数: $((SLURM_NNODES * 4))"
echo "进程总数: $WORLD_SIZE"

# 检查GPU状态
echo "🔍 GPU状态检查..."
srun --export=ALL --ntasks=2 --ntasks-per-node=1 \
    bash -c "echo \"=== 节点: \$(hostname) ===\"; nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv,noheader"

# =============================================================================
# 训练配置检查
# =============================================================================
echo "📋 训练配置检查..."

CONFIG_FILE="/home/bingxing2/home/scx7as9/run/CSUqx/CH/configs/pipeline.yaml"
TRAIN_SCRIPT="$PROJECT_ROOT/scripts/distributed/distributed_train.py"

# 检查关键文件是否存在
if [ ! -f "$CONFIG_FILE" ]; then
    echo "❌ 配置文件不存在: $CONFIG_FILE"
    exit 1
fi

if [ ! -f "$TRAIN_SCRIPT" ]; then
    echo "❌ 训练脚本不存在: $TRAIN_SCRIPT"
    exit 1
fi

if [ ! -d "$DATA_ROOT/imagesTr" ]; then
    echo "❌ 训练图像目录不存在: $DATA_ROOT/imagesTr"
    exit 1
fi

if [ ! -d "$DATA_ROOT/labelsTr" ]; then
    echo "❌ 训练标签目录不存在: $DATA_ROOT/labelsTr"
    exit 1
fi

echo "✅ 所有必要文件检查通过"



# =============================================================================
# 启动分布式训练
# =============================================================================
echo "=========================================="
echo "🎯 启动TA-CHNet分布式训练"
echo "=========================================="


# 启动训练
srun --export=ALL \
  bash -c "
    export RANK=\$SLURM_PROCID
    export LOCAL_RANK=\$SLURM_LOCALID
    echo \"[启动] 节点: \$(hostname), Global Rank: \$RANK, Local Rank: \$LOCAL_RANK\"

    # 切换到项目目录
    cd $PROJECT_ROOT

    # 启动训练
    python scripts/distributed/distributed_train.py \
      --image_dir ${DATA_ROOT}/imagesTr \
      --label_dir ${DATA_ROOT}/labelsTr \
      --output_dir ${OUTPUT_DIR} \
      --config ${CONFIG_FILE} \
      --epochs 20 \
      --lr 1e-4 \
      --num_workers 8 \
      --val_interval 3 \
      --save_interval 5 \
      --log_interval 50 \
      --amp \
      2>&1 | tee ${LOG_DIR}/rank_\${RANK}.log
  "

TRAIN_EXIT_CODE=$?

# =============================================================================
# 训练完成后处理
# =============================================================================
echo "=========================================="
echo "📊 训练完成后处理"
echo "=========================================="

echo "结束时间: $(date)"
echo "训练退出码: $TRAIN_EXIT_CODE"

# 收集训练统计信息
if [ -d "${LOG_DIR}" ]; then
    echo "📈 收集训练日志..."

    # 合并所有rank的日志
    cat ${LOG_DIR}/rank_*.log > ${LOG_DIR}/combined_training.log 2>/dev/null || true

    # 统计训练时间
    TRAIN_TIME=$(grep -h "训练耗时" ${LOG_DIR}/rank_*.log | head -1 2>/dev/null || echo "未知")
    echo "总训练时间: $TRAIN_TIME"

    # 检查最终模型
    if [ -f "${CHECKPOINT_DIR}/final_model.pth" ]; then
        echo "✅ 最终模型已保存: ${CHECKPOINT_DIR}/final_model.pth"
        ls -lh ${CHECKPOINT_DIR}/final_model.pth
    else
        echo "⚠️ 未找到最终模型文件"
    fi

    # 检查最佳模型
    if [ -f "${CHECKPOINT_DIR}/best_model.pth" ]; then
        echo "✅ 最佳模型已保存: ${CHECKPOINT_DIR}/best_model.pth"
        ls -lh ${CHECKPOINT_DIR}/best_model.pth
    fi
fi

# GPU状态最终检查
echo "🔧 最终GPU状态:"
srun --export=ALL --ntasks=1 --ntasks-per-node=1 \
    nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader 2>/dev/null || true

if [ $TRAIN_EXIT_CODE -eq 0 ]; then
    echo "🎉 TA-CHNet训练成功完成！"
    echo "📁 结果保存在: ${OUTPUT_DIR}"
    echo "📋 日志文件: ${LOG_DIR}/combined_training.log"
    echo "💾 模型文件: ${CHECKPOINT_DIR}/"
else
    echo "❌ 训练过程中出现错误 (退出码: $TRAIN_EXIT_CODE)"
    echo "🔍 请查看错误日志: ${LOG_DIR}/"
fi

echo "=========================================="
echo "作业完成时间: $(date)"
echo "=========================================="

exit $TRAIN_EXIT_CODE