import argparse
import sys
import yaml
import torch
import torch.optim as optim
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader
from tqdm import tqdm
from pathlib import Path

# 添加项目根目录到路径
sys.path.append(str(Path(__file__).resolve().parent.parent))

from models import VesselSegmenter
from data import LiverVesselDataset
from losses import VesselSegmentationLoss
from utils import Logger
from distributed.engine import DistributedEngine
from data.sampling_manager import SamplingManager
from scripts.evaluation import EvaluationManager


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='Train Liver Vessel Segmentation Model')
    
    # 数据参数
    parser.add_argument('--image_dir', required=True, help='Path to image directory')
    parser.add_argument('--label_dir', required=True, help='Path to label directory')
    parser.add_argument('--output_dir', default='./output', help='Path to output directory')
    parser.add_argument('--config', default='configs/default.yaml', help='Path to config file')
    
    # 训练参数
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--amp', action='store_true', help='Use automatic mixed precision')
    
    # 分布式参数
    parser.add_argument('--gpus', type=str, default='0,1,2,3', help='GPU indices to use')
    
    # 模型参数
    parser.add_argument('--resume', help='Path to checkpoint to resume from')
    
    return parser.parse_args()


def train_epoch(engine, train_loader, criterion, optimizer, scaler, epoch, args,
               sampling_manager=None, evaluation_manager=None, logger=None):
    """
    训练一个epoch
    
    参数:
        engine: 分布式执行引擎
        train_loader: 训练数据加载器
        criterion: 损失函数
        optimizer: 优化器
        scaler: 梯度缩放器(用于混合精度)
        epoch: 当前epoch
        args: 命令行参数
        sampling_manager: 采样管理器
        evaluation_manager: 评估管理器
        logger: 日志记录器
    
    返回:
        平均损失
    """
    engine.train()
    running_loss = 0.0
    
    # 获取当前采样参数
    if sampling_manager:
        sampling_params = sampling_manager.get_sampling_params(epoch)
        # 将采样参数应用到数据集
        # train_loader.dataset.apply_sampling_params(sampling_params)
    
    # 创建进度条
    iterator = tqdm(train_loader, desc=f"Epoch {epoch}")
    
    for batch_idx, batch in enumerate(iterator):
        # 获取数据
        images = batch['image']
        labels = batch['label']
        tiers = batch['tier']
        
        # 清零梯度
        optimizer.zero_grad(set_to_none=True)
        
        # 处理每个样本
        outputs = []
        for j, tier in enumerate(tiers):
            # 设置当前tier
            engine.pipeline.set_tier(int(tier))
            
            # 前向传播
            with torch.cuda.amp.autocast(enabled=args.amp):
                outputs.append(engine.forward(images[j:j+1]))
        
        # 合并输出
        output = torch.cat(outputs)
        
        # 计算损失
        with torch.cuda.amp.autocast(enabled=args.amp):
            loss = criterion(output, labels)
        
        # 反向传播和优化
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        # 更新统计
        running_loss += loss.item()
        
        # 更新进度条
        iterator.set_postfix(loss=loss.item())
    
    # 计算平均损失
    avg_loss = running_loss / len(train_loader)
    
    # 记录日志
    if logger:
        logger.log_info(f"Epoch {epoch} - Average Loss: {avg_loss:.4f}")
        logger.log_metrics({'train_loss': avg_loss}, epoch)
    
    return avg_loss


def main(args):
    """主函数"""
    # 加载配置
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # 设置随机种子
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    
    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # 创建日志记录器
    logger = Logger(output_dir / 'logs')
    
    # 解析GPU列表
    gpus = [int(gpu) for gpu in args.gpus.split(',')]
    assert len(gpus) >= 4, "Need at least 4 GPUs for distributed execution"
    
    # 创建数据集
    train_dataset = LiverVesselDataset(args.image_dir, args.label_dir)
    val_dataset = LiverVesselDataset(args.image_dir, args.label_dir, random_sampling=False)
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers
    )
    
    # 创建模型
    model = VesselSegmenter(
        in_channels=1,
        out_channels=1,
        ch_params=config.get('ch_params'),
        tier_params=config.get('tier_params')
    )
    
    # 创建分布式执行引擎
    engine = DistributedEngine(model, gpus=gpus[:4], amp_enabled=args.amp)
    
    # 创建损失函数
    criterion = VesselSegmentationLoss(
        vessel_weight=config.get('vessel_weight', 10.0),
        tumor_weight=config.get('tumor_weight', 15.0),
        use_boundary=config.get('use_boundary', True)
    ).cuda(gpus[3])  # 损失计算在最后一个GPU上
    
    # 创建优化器
    optimizer = optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=config.get('weight_decay', 1e-5)
    )
    
    # 创建梯度缩放器
    scaler = GradScaler(enabled=args.amp)
    
    # 创建采样管理器
    sampling_manager = SamplingManager(config.get('smart_sampling', {}))
    
    # 创建评估管理器
    evaluation_manager = EvaluationManager(config.get('evaluation', {}))
    
    # 恢复检查点(如果有)
    start_epoch = 0
    if args.resume:
        checkpoint = torch.load(args.resume)
        engine.model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scaler.load_state_dict(checkpoint['scaler'])
        start_epoch = checkpoint['epoch'] + 1
        logger.log_info(f"Resumed from checkpoint: {args.resume} at epoch {start_epoch}")
    
    # 训练循环
    for epoch in range(start_epoch, args.epochs):
        # 更新采样策略
        if sampling_manager.should_update(epoch):
            sampling_manager.update_sampling_strategy(engine, train_dataset, epoch)
        
        # 训练一个epoch
        train_loss = train_epoch(
            engine, train_loader, criterion, optimizer, scaler, epoch, args,
            sampling_manager, evaluation_manager, logger
        )
        
        # 评估模型
        if evaluation_manager.should_evaluate(epoch):
            metrics = evaluation_manager.evaluate(engine, val_loader, epoch)
            
            # 记录评估指标
            if metrics and logger:
                logger.log_metrics(metrics, epoch, prefix='val_')
                logger.log_info(f"Epoch {epoch} - Validation Dice: {metrics['dice']:.4f}")
        
        # 保存检查点
        if epoch % 5 == 0 or epoch == args.epochs - 1:
            checkpoint = {
                'epoch': epoch,
                'model': engine.get_consolidated_model().state_dict(),
                'optimizer': optimizer.state_dict(),
                'scaler': scaler.state_dict(),
                'config': config
            }
            
            torch.save(checkpoint, output_dir / f'checkpoint_epoch_{epoch}.pt')
            logger.log_info(f"Checkpoint saved at epoch {epoch}")
    
    # 训练结束，保存最终模型
    final_model = engine.get_consolidated_model()
    torch.save(final_model.state_dict(), output_dir / 'final_model.pt')
    
    logger.log_info("Training completed!")
    
    return final_model


if __name__ == '__main__':
    args = parse_args()
    main(args)