import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import argparse
from tqdm import tqdm
from sklearn.model_selection import KFold
import numpy as np

# 项目已通过 pip install -e . 安装，项目根由 .pth 提供，无需 sys.path 自举
from utils import load_config
from models import CervicalCancerModel
from utils.preprocess import CombinedCervicalCancerDataset, get_transforms


def create_kfold_data_loaders(dataset, k_folds=5, batch_size=32, fold_idx=0):
    """创建K折交叉验证的数据加载器"""
    kfold = KFold(n_splits=k_folds, shuffle=True, random_state=42)

    # 获取训练和验证索引
    splits = list(kfold.split(dataset))
    train_indices, val_indices = splits[fold_idx]

    # 创建数据子集
    train_subset = Subset(dataset, train_indices)
    val_subset = Subset(dataset, val_indices)

    # 创建数据加载器
    train_loader = DataLoader(
        train_subset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True
    )
    val_loader = DataLoader(
        val_subset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True
    )

    return train_loader, val_loader


def train_fold(model, train_loader, val_loader, criterion, optimizer, scheduler,
               num_epochs, device):
    """训练单个fold"""
    best_acc = 0.0
    best_loss = float('inf')

    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        for inputs, labels in tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs}', leave=False):
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)

            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()

        train_acc = correct_train / total_train

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

        # 学习率调度
        if scheduler:
            if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_loss)
            else:
                scheduler.step()

        # 保存最佳模型
        if val_loss < best_loss:
            best_loss = val_loss
            if val_acc > best_acc:
                best_acc = val_acc

        print(f'Fold - Epoch {epoch + 1}/{num_epochs}, Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}')

    return best_acc


def cross_validation(config, k_folds=5):
    """执行K折交叉验证"""
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # 获取数据转换
    train_transform = get_transforms(config, 'train')
    # TODO: 验证集当前共用 train_transform（含随机增强），折内指标含噪声、不可复现。
    #       修复需连同 create_kfold_data_loaders 一起改（需按 split 传入不同 transform），
    #       属独立课题，改动会影响历史指标可比性，故暂只标注。
    test_transform = get_transforms(config, 'test')  # noqa: F841

    # 创建数据集
    dataset_config = config['data']['datasets']
    dataset = CombinedCervicalCancerDataset(dataset_config, train_transform, config)

    # 存储每折的结果
    fold_results = []

    # 执行K折交叉验证
    for fold in range(k_folds):
        print(f"\n=== Fold {fold + 1}/{k_folds} ===")

        # 创建数据加载器
        train_loader, val_loader = create_kfold_data_loaders(
            dataset, k_folds, config['data']['batch_size'], fold
        )

        # 初始化模型
        model = CervicalCancerModel(
            num_classes=config['model']['num_classes'],
            pretrained=config['model']['pretrained'],
            dropout_rate=config['model']['dropout_rate']
        ).to(device)

        # 定义损失函数和优化器
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(
            model.parameters(),
            lr=config['training']['learning_rate'],
            weight_decay=config['training']['weight_decay']
        )

        # 学习率调度器
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            patience=config['training']['patience'] // 2,
            factor=0.5,
            min_lr=1e-6
        )

        # 训练fold
        best_acc = train_fold(
            model, train_loader, val_loader, criterion, optimizer, scheduler,
            config['training']['epochs'], device
        )

        fold_results.append(best_acc)
        print(f"Fold {fold + 1} Best Accuracy: {best_acc:.4f}")

    # 输出结果统计
    mean_acc = np.mean(fold_results)
    std_acc = np.std(fold_results)

    print("\n=== Cross Validation Results ===")
    print(f"Fold Accuracies: {[f'{acc:.4f}' for acc in fold_results]}")
    print(f"Mean Accuracy: {mean_acc:.4f} ± {std_acc:.4f}")

    return fold_results, mean_acc, std_acc


def main():
    parser = argparse.ArgumentParser(description='Perform K-Fold Cross Validation')
    parser.add_argument('--config', type=str, default='configs/default.yaml', help='Path to config file')
    parser.add_argument('--k_folds', type=int, default=5, help='Number of folds for cross validation')

    args = parser.parse_args()

    # 加载配置
    config = load_config(args.config)

    # 执行交叉验证
    fold_results, mean_acc, std_acc = cross_validation(config, args.k_folds)

    # 保存结果
    os.makedirs(config['paths']['result_dir'], exist_ok=True)
    result_file = os.path.join(config['paths']['result_dir'], 'cross_validation_results.txt')

    with open(result_file, 'w') as f:
        f.write("Cross Validation Results\n")
        f.write("========================\n")
        f.write(f"Fold Accuracies: {[f'{acc:.4f}' for acc in fold_results]}\n")
        f.write(f"Mean Accuracy: {mean_acc:.4f} ± {std_acc:.4f}\n")

    print(f"Results saved to {result_file}")


if __name__ == '__main__':
    main()
