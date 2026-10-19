import sys
import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import argparse
import yaml
from tqdm import tqdm
import numpy as np
import wandb  # 添加wandb导入
import logging
import random

# 添加项目根目录到Python路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.insert(0, project_root)

# 现在可以正常导入项目模块
from utils.label_mapping import (
    BINARY_CLASSES, 
    MULTICLASS_CLASSES, 
    DEFAULT_LABEL_SPACE, 
    get_multiclass_classes,
)
from utils import create_data_loaders
# 修改导入，直接使用load_config函数
from utils.preprocess import load_config
# 导入模型类
from models.cer_cancer_model import CervicalCancerModel
# 导入高级数据增强方法
from utils.advanced_augmentation import mixup_data, mixup_criterion, cutmix_data, cutmix_criterion


class FocalLoss(nn.Module):
    """Focal Loss implementation for handling class imbalance"""

    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = nn.functional.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)

        if self.alpha is not None:
            if isinstance(self.alpha, (float, int)):
                alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            else:
                alpha_t = self.alpha[targets]
            focal_loss = alpha_t * (1 - pt) ** self.gamma * ce_loss
        else:
            focal_loss = (1 - pt) ** self.gamma * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class EMA:
    """
    Exponential Moving Average implementation for model parameters
    """

    def __init__(self, model, decay=0.999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}

        # 初始化shadow weights
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        """更新EMA参数"""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def apply_shadow(self):
        """应用EMA权重到模型"""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data
                param.data = self.shadow[name]

    def restore(self):
        """恢复原始模型权重"""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data = self.backup[name]
        self.backup = {}


def compute_class_weights(train_loader):
    """计算类别权重以处理数据不平衡"""
    # 统计每个类别的样本数量
    class_counts = {}
    for _, labels in train_loader:
        for label in labels:
            label_item = label.item()
            class_counts[label_item] = class_counts.get(label_item, 0) + 1

    # 计算权重（反比）
    total_samples = sum(class_counts.values())
    num_classes = len(class_counts)
    class_weights = torch.zeros(num_classes)

    for class_id, count in class_counts.items():
        class_weights[class_id] = total_samples / (num_classes * count)

    return class_weights


def get_balanced_sampler(train_loader):
    """获取平衡采样器"""
    from torch.utils.data.sampler import WeightedRandomSampler

    # 统计每个类别的样本数量
    class_counts = {}
    for _, labels in train_loader:
        for label in labels:
            label_item = label.item()
            class_counts[label_item] = class_counts.get(label_item, 0) + 1

    # 计算每个样本的权重
    weights = []
    for _, labels in train_loader:
        for label in labels:
            label_item = label.item()
            weight = 1.0 / class_counts[label_item]
            weights.append(weight)

    sampler = WeightedRandomSampler(
        torch.DoubleTensor(weights),
        len(weights),
        replacement=True
    )

    return sampler

def save_classes_sidecar(weights_path: str, classification_mode: str, target_names, label_space=None):
    """把类别表随权重一起保存，供推理端对齐，消除训练/推理漂移"""
    sidecar = os.path.splitext(weights_path)[0] + ".classes.json"
    with open(sidecar, "w", encoding="utf-8") as f:
        json.dump(
            {"classification_mode": classification_mode, "label_space": label_space, "classes": list(target_names)},
            f, ensure_ascii=False, indent=2,
        )

def build_model_filename(config, classification_mode, label_space=None):
    """
    统一构造产物文件名，避免不同标签空间互相覆盖。

    - 二分类:  <name>_binary_model.pth
    - 多分类:  <name>_multiclass_<label_space>_model.pth
    """
    name = config['model']['name']
    if classification_mode == 'binary':
        return f"{name}_binary_model.pth"
    return f"{name}_multiclass_{label_space or DEFAULT_LABEL_SPACE}_model.pth"

def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler,
                num_epochs, device, log_dir, model_save_dir, early_stopping_patience=10,
                use_wandb=False, config=None, use_lr_warmup=False, scheduler_type='plateau',
                classification_mode='binary', use_mixup=False, mixup_alpha=0.2,
                use_cutmix=False, cutmix_alpha=1.0, ema=None, target_names=None, label_space=None):
    """训练模型"""
    writer = SummaryWriter(log_dir=log_dir)
    best_acc = 0.0
    best_loss = float('inf')
    patience_counter = 0
    train_losses = []
    val_losses = []
    val_accuracies = []

    # 学习率预热设置
    if use_lr_warmup:
        warmup_epochs = min(5, num_epochs // 10)
        warmup_scheduler = optim.lr_scheduler.LinearLR(
            optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_epochs
        )

    for epoch in range(num_epochs):
        # 应用学习率预热
        if use_lr_warmup and epoch < warmup_epochs:
            warmup_scheduler.step()

        # 训练阶段
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        for inputs, labels in tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs}'):
            inputs = inputs.to(device)
            labels = labels.to(device)

            # 应用MixUp或CutMix数据增强
            use_mix = False
            if use_mixup and use_cutmix:
                # 随机选择使用MixUp或CutMix
                use_mix = random.random() > 0.5
                if use_mix:  # 使用MixUp
                    inputs, labels_a, labels_b, lam = mixup_data(inputs, labels, mixup_alpha)
                else:  # 使用CutMix
                    inputs, labels_a, labels_b, lam = cutmix_data(inputs, labels, cutmix_alpha)
            elif use_mixup:
                use_mix = True
                inputs, labels_a, labels_b, lam = mixup_data(inputs, labels, mixup_alpha)
            elif use_cutmix:
                inputs, labels_a, labels_b, lam = cutmix_data(inputs, labels, cutmix_alpha)

            optimizer.zero_grad()

            outputs = model(inputs)

            # 根据是否使用增强技术选择损失函数
            if use_mixup or use_cutmix:
                loss = mixup_criterion(criterion, outputs, labels_a, labels_b, lam)
            else:
                loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            # 更新EMA
            if ema is not None:
                ema.update()

            running_loss += loss.item() * inputs.size(0)

            # 计算准确率时使用原始标签
            if use_mixup or use_cutmix:
                _, predicted = torch.max(outputs.data, 1)
                correct_train += (lam * predicted.eq(labels_a.data).sum().float() +
                                  (1 - lam) * predicted.eq(labels_b.data).sum().float()).item()
                total_train += labels.size(0)
            else:
                _, predicted = torch.max(outputs.data, 1)
                total_train += labels.size(0)
                correct_train += (predicted == labels).sum().item()

        epoch_loss = running_loss / len(train_loader.dataset)
        train_acc = correct_train / total_train
        train_losses.append(epoch_loss)

        # 验证阶段
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * inputs.size(0)

                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        val_loss = val_loss / len(val_loader.dataset)
        val_acc = correct / total

        val_losses.append(val_loss)
        val_accuracies.append(val_acc)

        # 学习率调度
        if scheduler and not (use_lr_warmup and epoch < warmup_epochs):
            if scheduler_type == 'cosine':
                scheduler.step()
            elif scheduler_type == 'cosine_warm_restarts':
                scheduler.step(epoch)
            elif scheduler_type == 'one_cycle':
                scheduler.step()
            elif scheduler_type == 'plateau':
                if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(val_loss)
            else:
                scheduler.step()

        # 记录到TensorBoard
        writer.add_scalar('Loss/train', epoch_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('Accuracy/train', train_acc, epoch)
        writer.add_scalar('Accuracy/val', val_acc, epoch)
        writer.add_scalar('LearningRate', optimizer.param_groups[0]['lr'], epoch)

        # 记录到Weights & Biases
        if use_wandb:
            wandb.log({
                'epoch': epoch,
                'train_loss': epoch_loss,
                'train_acc': train_acc,
                'val_loss': val_loss,
                'val_acc': val_acc,
                'learning_rate': optimizer.param_groups[0]['lr']
            })

        logging.info(f'Epoch {epoch + 1}/{num_epochs}')
        logging.info(f'Train Loss: {epoch_loss:.4f}, Train Acc: {train_acc:.4f}')
        logging.info(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')
        logging.info(f'Learning Rate: {optimizer.param_groups[0]["lr"]:.6f}')

        # 早停机制
        if val_loss < best_loss:
            best_loss = val_loss
            patience_counter = 0
            # 保存最佳模型
            if val_acc > best_acc:
                best_acc = val_acc
            # 生成基于模型类型的文件名
            model_filename = build_model_filename(config, classification_mode, label_space)
            best_model_path = os.path.join(model_save_dir, f"best_{model_filename}")

            # 如果使用EMA，保存EMA模型权重
            if ema is not None:
                ema.apply_shadow()
                torch.save(model.state_dict(), best_model_path)
                ema.restore()
            else:
                torch.save(model.state_dict(), best_model_path)
            if target_names is not None:
                save_classes_sidecar(best_model_path, classification_mode, target_names, label_space=label_space)
        else:
            patience_counter += 1
            if patience_counter >= early_stopping_patience:
                logging.info(f'Early stopping at epoch {epoch + 1}')
                break

        # 保存最新模型
        model_filename = build_model_filename(config, classification_mode, label_space)
        latest_model_path = os.path.join(model_save_dir, f"latest_{model_filename}")

        # 如果使用EMA，保存EMA模型权重
        if ema is not None:
            ema.apply_shadow()
            torch.save(model.state_dict(), latest_model_path)
            ema.restore()
        else:
            torch.save(model.state_dict(), latest_model_path)
        if target_names is not None:
            save_classes_sidecar(latest_model_path, classification_mode, target_names, label_space=label_space)

    writer.close()
    return model, train_losses, val_losses, val_accuracies


def setup_logging(log_file):
    """设置日志记录"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )


def main():
    parser = argparse.ArgumentParser(description='Train cervical cancer classification model')
    parser.add_argument('--epochs', type=int, help='Number of training epochs')
    parser.add_argument('--config', type=str, default='configs/default.yaml', help='Path to config file')
    parser.add_argument('--resume', type=str, help='Path to checkpoint to resume from')
    parser.add_argument('--use_wandb', action='store_true', help='Enable Weights & Biases tracking')
    parser.add_argument('--model_type', type=str, default=None,
                        choices=['resnet50', 'resnet101', 'efficientnet_b0', 'vit'],
                        help='Model type to use (overrides config file)')
    parser.add_argument('--classification_mode', type=str, default='binary',
                        choices=['binary', 'multiclass'],
                        help='Classification mode: binary (normal/abnormal) or multiclass')
    parser.add_argument('--batch_size', type=int, help='Batch size for training')
    parser.add_argument('--loss_type', type=str, default='cross_entropy',
                        choices=['cross_entropy', 'focal', 'weighted'],
                        help='Loss function type')
    parser.add_argument('--use_sampler', action='store_true',
                        help='Use balanced sampler for handling class imbalance')
    parser.add_argument('--use_lr_warmup', action='store_true',
                        help='Use learning rate warmup')
    parser.add_argument('--gamma', type=float, default=2.0,
                        help='Gamma value for focal loss')
    parser.add_argument('--attention_type', type=str, default='se',
                        choices=['se', 'cbam', 'none'],
                        help='Attention mechanism type')
    parser.add_argument('--scheduler_type', type=str, default='plateau',
                        choices=['plateau', 'cosine', 'cosine_warm_restarts', 'one_cycle'],
                        help='Learning rate scheduler type')
    parser.add_argument('--experiment_name', type=str, default='experiment',
                        help='Experiment name for logging')
    parser.add_argument('--use_mixup', action='store_true',
                        help='Use MixUp data augmentation')
    parser.add_argument('--mixup_alpha', type=float, default=0.2,
                        help='Alpha parameter for MixUp')
    parser.add_argument('--use_cutmix', action='store_true',
                        help='Use CutMix data augmentation')
    parser.add_argument('--cutmix_alpha', type=float, default=1.0,
                        help='Alpha parameter for CutMix')
    parser.add_argument('--use_ema', action='store_true',
                        help='Use Exponential Moving Average for model weights')
    parser.add_argument('--ema_decay', type=float, default=0.999,
                        help='Decay rate for EMA')
    parser.add_argument('--label_space', type=str, default=None,
                        choices=['sipakmed', 'herlev'],help='Multiclass label space (sipakmed=5类 | herlev=7类); '
                        'None 时读 config["model"]["label_space"]')

    args = parser.parse_args()

    # 设置日志
    os.makedirs('logs', exist_ok=True)
    setup_logging(f'logs/{args.experiment_name}.log')

    # 加载配置
    config = load_config(args.config)

    # 如果命令行指定了模型类型，则覆盖配置文件中的设置
    if args.model_type:
        config['model']['model_type'] = args.model_type
        config['model']['name'] = args.model_type

    if args.epochs is not None:
        config['training']['epochs'] = args.epochs

    if args.batch_size is not None:
        config['data']['batch_size'] = args.batch_size

    # 初始化Weights & Biases
    if args.use_wandb:
        wandb.init(
            project="cervical-cancer-detection",
            config=config,
            name=args.experiment_name
        )

    # 创建输出目录
    os.makedirs(config['paths']['model_save'], exist_ok=True)
    os.makedirs(config['paths']['log_dir'], exist_ok=True)
    os.makedirs(config['paths']['result_dir'], exist_ok=True)

    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device: {device}')

    # 创建数据加载器
    train_loader, val_loader, test_loader = create_data_loaders(config, args.classification_mode, args.label_space)

    # 根据分类模式设置类别数量和名称
    if args.classification_mode == 'binary':
        num_classes = len(BINARY_CLASSES)
        target_names = list(BINARY_CLASSES)
        label_space = None
    else:
        label_space = args.label_space or (config.get('model') or {}).get('label_space') \
            or DEFAULT_LABEL_SPACE
        target_names = get_multiclass_classes(label_space)
        num_classes = len(target_names)

    # 初始化模型
    use_attention = args.attention_type != 'none'
    model = CervicalCancerModel(
        num_classes=num_classes,
        pretrained=config['model']['pretrained'],
        dropout_rate=config['model']['dropout_rate'],
        model_type=config['model'].get('model_type', 'resnet50'),
        use_attention=use_attention,
        attention_type=args.attention_type if use_attention else None
    ).to(device)

    # 加载检查点（如果提供）
    if args.resume:
        model.load_state_dict(torch.load(args.resume, map_location=device))

    # 定义损失函数
    if args.loss_type == 'focal':
        criterion = FocalLoss(gamma=args.gamma)
        logging.info(f"Using Focal Loss with gamma={args.gamma}")
    elif args.loss_type == 'weighted':
        class_weights = compute_class_weights(train_loader)
        class_weights = class_weights.to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        logging.info(f"Using Weighted Cross Entropy Loss with weights: {class_weights}")
    else:
        criterion = nn.CrossEntropyLoss()
        logging.info("Using standard Cross Entropy Loss")

    # 定义优化器
    optimizer = optim.Adam(
        model.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay']
    )

    # 学习率调度器
    if args.scheduler_type == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config['training']['epochs'],
            eta_min=1e-6
        )
    elif args.scheduler_type == 'cosine_warm_restarts':
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=config['training']['epochs'] // 5,
            T_mult=1,
            eta_min=1e-6
        )
    elif args.scheduler_type == 'one_cycle':
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=config['training']['learning_rate'] * 10,
            epochs=config['training']['epochs'],
            steps_per_epoch=len(train_loader),
            pct_start=0.3
        )
    else:  # 默认使用plateau
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            patience=config['training']['patience'] // 2,
            factor=0.5,
            min_lr=1e-6
        )

    # 如果使用平衡采样器，重新创建训练加载器
    if args.use_sampler:
        sampler = get_balanced_sampler(train_loader)
        from torch.utils.data import DataLoader
        train_loader = DataLoader(
            train_loader.dataset,
            batch_size=config['data']['batch_size'],
            sampler=sampler,
            num_workers=train_loader.num_workers,
            pin_memory=train_loader.pin_memory
        )
        logging.info("Using balanced sampler for training")

    # 初始化EMA
    ema = None
    if args.use_ema:
        ema = EMA(model, decay=args.ema_decay)
        logging.info(f"Using Exponential Moving Average with decay={args.ema_decay}")

    # 训练模型
    model, train_losses, val_losses, val_accuracies = train_model(
        model, train_loader, val_loader, criterion, optimizer, scheduler,
        config['training']['epochs'], device,
        config['paths']['log_dir'], config['paths']['model_save'],
        config['training']['early_stopping_patience'],
        use_wandb=args.use_wandb,
        config=config,
        use_lr_warmup=args.use_lr_warmup,
        scheduler_type=args.scheduler_type,
        classification_mode=args.classification_mode,
        use_mixup=args.use_mixup,
        mixup_alpha=args.mixup_alpha,
        use_cutmix=args.use_cutmix,
        cutmix_alpha=args.cutmix_alpha,
        ema=ema,
        target_names=target_names,
        label_space=label_space,
    )

    # 绘制训练曲线
    from utils.visualization import plot_training_curves
    plot_training_curves(
        train_losses, val_losses, val_accuracies,
        os.path.join(config['paths']['result_dir'],
                     f'training_curves_{args.classification_mode}_{args.experiment_name}.png')
    )

    # 结束Weights & Biases会话
    if args.use_wandb:
        wandb.finish()


if __name__ == '__main__':
    main()