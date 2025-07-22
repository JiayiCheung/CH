#!/bin/bash
#SBATCH -x paraai-n32-h-01-agent-1,paraai-n32-h-01-agent-4,paraai-n32-h-01-agent-8,paraai-n32-h-01-agent-16,paraai-n32-h-01-agent-17,paraai-n32-h-01-agent-25,paraai-n32-h-01-agent-27,paraai-n32-h-01-agent-28,paraai-n32-h-01-agent-29,paraai-n32-h-01-agent-30,paraai-n32-h-01-agent-31
#SBATCH --job-name=DIST_TEST_FIXED_NET
#SBATCH --output=dist_test_fixed_%j.out
#SBATCH --error=dist_test_fixed_%j.err
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=8
#SBATCH --qos=gpugpu
#SBATCH --time=1:00:00

echo "=========================================="
echo "🔧 修正网络配置的分布式测试"
echo "=========================================="
echo "作业ID: $SLURM_JOB_ID"
echo "开始时间: $(date)"

# 加载环境
module purge
module load miniforge3/24.1 compilers/gcc/13.2.0 compilers/cuda/11.8 cudnn/8.8.1.3_cuda11.x
source activate vessel

echo "Python版本: $(python --version)"
echo "PyTorch版本: $(python -c 'import torch; print(torch.__version__)')"

# =============================================================================
# 网络接口检查和修正
# =============================================================================
echo "=========================================="
echo "🔍 网络接口检查"
echo "=========================================="

echo "检查所有节点的网络接口:"
srun --export=ALL bash -c "
    echo \"=== 节点: \$(hostname) ===\"
    echo \"所有网络接口:\"
    ip addr show | grep -E '^[0-9]+:' | cut -d: -f2 | tr -d ' '
    echo \"活动接口状态:\"
    ip addr show | grep -E 'inet ' | head -5
    echo \"InfiniBand接口检查:\"
    ls /sys/class/net/ | grep -E '^(ib|mlx)' || echo '  无InfiniBand接口'
    echo \"以太网接口检查:\"
    ls /sys/class/net/ | grep -E '^(eth|en)' || echo '  无以太网接口'
    echo \"----------------------------------------\"
"

# =============================================================================
# 分布式环境变量设置（多种网络配置尝试）
# =============================================================================
echo "=========================================="
echo "⚙️ 分布式环境配置（智能网络检测）"
echo "=========================================="

export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=29501
export WORLD_SIZE=$SLURM_NTASKS
export PYTHONNOUSERSITE=1
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=8

# 项目路径
PROJECT_ROOT="/home/bingxing2/home/scx7as9/run/CSUqx/CH"
SCRIPT_DIR="${PROJECT_ROOT}/scripts/distributed"
cd $PROJECT_ROOT

echo "基础配置:"
echo "  MASTER_ADDR: $MASTER_ADDR"
echo "  MASTER_PORT: $MASTER_PORT"
echo "  WORLD_SIZE: $WORLD_SIZE"

# =============================================================================
# 尝试多种网络配置
# =============================================================================

# 尝试1：使用以太网接口（最兼容）
echo ""
echo "🧪 尝试1：使用以太网接口配置"
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=1                    # 禁用InfiniBand
export NCCL_SOCKET_IFNAME="^lo,docker0"     # 排除回环和docker接口，使用以太网
export NCCL_TIMEOUT=1800

echo "网络配置1:"
echo "  NCCL_IB_DISABLE: $NCCL_IB_DISABLE"
echo "  NCCL_SOCKET_IFNAME: $NCCL_SOCKET_IFNAME"

echo "开始测试..."
srun --export=ALL \
    bash -c "
        export RANK=\$SLURM_PROCID
        export LOCAL_RANK=\$SLURM_LOCALID
        echo \"[配置1] 主机: \$(hostname), Rank: \$RANK, Local: \$LOCAL_RANK\"
        cd $SCRIPT_DIR
        timeout 300 python 123.py 2>/dev/null
    "

EXIT_CODE1=$?
echo "配置1测试结果: 退出码 $EXIT_CODE1"

if [ $EXIT_CODE1 -eq 0 ]; then
    echo "✅ 配置1成功！使用以太网接口"
    echo "=========================================="
    echo "🎉 分布式测试成功！"
    echo "✅ 建议在训练中使用以下NCCL配置:"
    echo "   export NCCL_IB_DISABLE=1"
    echo "   export NCCL_SOCKET_IFNAME=\"^lo,docker0\""
    echo "=========================================="
    exit 0
fi

# 尝试2：自动检测网络接口
echo ""
echo "🧪 尝试2：自动网络接口检测"
export NCCL_IB_DISABLE=0
unset NCCL_SOCKET_IFNAME  # 让NCCL自动选择

echo "网络配置2:"
echo "  NCCL_IB_DISABLE: $NCCL_IB_DISABLE"
echo "  NCCL_SOCKET_IFNAME: 自动检测"

srun --export=ALL \
    bash -c "
        export RANK=\$SLURM_PROCID
        export LOCAL_RANK=\$SLURM_LOCALID
        echo \"[配置2] 主机: \$(hostname), Rank: \$RANK, Local: \$LOCAL_RANK\"
        cd $SCRIPT_DIR
        timeout 300 python 123.py 2>/dev/null
    "

EXIT_CODE2=$?
echo "配置2测试结果: 退出码 $EXIT_CODE2"

if [ $EXIT_CODE2 -eq 0 ]; then
    echo "✅ 配置2成功！自动网络检测"
    echo "=========================================="
    echo "🎉 分布式测试成功！"
    echo "✅ 建议在训练中使用以下NCCL配置:"
    echo "   export NCCL_IB_DISABLE=0"
    echo "   unset NCCL_SOCKET_IFNAME"
    echo "=========================================="
    exit 0
fi

# 尝试3：强制使用特定以太网接口
echo ""
echo "🧪 尝试3：强制使用特定接口"

# 检测常见的以太网接口名称
COMMON_INTERFACES="eth0 ens3 ens4 enp0s3 enp0s8"
for iface in $COMMON_INTERFACES; do
    echo "尝试接口: $iface"
    
    export NCCL_IB_DISABLE=1
    export NCCL_SOCKET_IFNAME="$iface"
    
    srun --export=ALL \
        bash -c "
            export RANK=\$SLURM_PROCID
            export LOCAL_RANK=\$SLURM_LOCALID
            if [ \$RANK -eq 0 ]; then
                echo \"[配置3-$iface] 主机: \$(hostname), Rank: \$RANK\"
            fi
            cd $SCRIPT_DIR
            timeout 180 python 123.py 2>/dev/null
        " 2>/dev/null

    EXIT_CODE3=$?
    
    if [ $EXIT_CODE3 -eq 0 ]; then
        echo "✅ 配置3成功！使用接口: $iface"
        echo "=========================================="
        echo "🎉 分布式测试成功！"
        echo "✅ 建议在训练中使用以下NCCL配置:"
        echo "   export NCCL_IB_DISABLE=1"
        echo "   export NCCL_SOCKET_IFNAME=\"$iface\""
        echo "=========================================="
        exit 0
    fi
done

# 尝试4：TCP网络（最后的备选方案）
echo ""
echo "🧪 尝试4：强制TCP网络通信"

export NCCL_IB_DISABLE=1
export NCCL_SOCKET_IFNAME="lo"  # 使用回环接口
export NCCL_P2P_DISABLE=1       # 禁用点对点通信
export NCCL_NET_GDR_LEVEL=0     # 禁用GPU直接RDMA

echo "网络配置4 (TCP):"
echo "  NCCL_IB_DISABLE: $NCCL_IB_DISABLE"
echo "  NCCL_SOCKET_IFNAME: $NCCL_SOCKET_IFNAME"
echo "  NCCL_P2P_DISABLE: $NCCL_P2P_DISABLE"

srun --export=ALL \
    bash -c "
        export RANK=\$SLURM_PROCID
        export LOCAL_RANK=\$SLURM_LOCALID
        echo \"[配置4] 主机: \$(hostname), Rank: \$RANK, Local: \$LOCAL_RANK\"
        cd $SCRIPT_DIR
        timeout 300 python 123.py
    "

EXIT_CODE4=$?

# =============================================================================
# 最终结果分析
# =============================================================================
echo "=========================================="
echo "📊 最终测试结果分析"
echo "=========================================="
echo "结束时间: $(date)"
echo "配置1 (以太网): 退出码 $EXIT_CODE1"
echo "配置2 (自动检测): 退出码 $EXIT_CODE2"  
echo "配置3 (特定接口): 测试了多个接口"
echo "配置4 (TCP): 退出码 $EXIT_CODE4"

if [ $EXIT_CODE4 -eq 0 ]; then
    echo ""
    echo "🎉 分布式测试最终成功！"
    echo "✅ 使用TCP网络配置:"
    echo "   export NCCL_IB_DISABLE=1"
    echo "   export NCCL_SOCKET_IFNAME=\"lo\""
    echo "   export NCCL_P2P_DISABLE=1"
    echo "   export NCCL_NET_GDR_LEVEL=0"
    exit 0
else
    echo ""
    echo "❌ 所有网络配置尝试均失败"
    echo "可能的问题:"
    echo "  1. 网络驱动问题"
    echo "  2. 防火墙阻止通信"
    echo "  3. NCCL版本兼容性问题"
    echo "  4. 系统级网络配置问题"
    echo ""
    echo "建议联系系统管理员检查网络配置"
    exit 1
fi