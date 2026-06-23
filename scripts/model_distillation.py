import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import argparse
import yaml
from tqdm import tqdm

# 添加项目根目录到Python路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.insert(0, project_root)

# 导入项目模块
from utils import create_data_loaders, load_config
from models import CervicalCancerModel


class DistillationLoss(nn.Module):
    """知识蒸馏损失函数"""

    def __init__(self, temperature=3.0, alpha=0.7):
        super(DistillationLoss, self).__init__()
        self.temperature = temperature
        self.alpha = alpha
        self.ce_loss = nn.CrossEntropyLoss()
        self.kl_loss = nn.KLDivLoss(reduction='batchmean')

    def forward(self, student_logits, teacher_logits, labels):
        # 软标签损失（KL散度）
        soft_loss = self.kl_loss(
            torch.log_softmax(student_logits / self.temperature, dim=1),
            torch.softmax(teacher_logits / self.temperature, dim=1)
        ) * (self.temperature ** 2)

        # 硬标签损失（交叉熵）
        hard_loss = self.ce_loss(student_logits, labels)

        # 组合损失
        loss = self.alpha * soft_loss + (1 - self.alpha) * hard_loss
        return loss


def train_student_model(teacher_model, student_model, train_loader, val_loader,
                        criterion, optimizer, scheduler, num_epochs, device,
                        temperature=3.0, alpha=0.7):
    """训练学生模型"""
    distill_criterion = DistillationLoss(temperature, alpha)
    best_acc = 0.0

    for epoch in range(num_epochs):
        # 训练阶段
        student_model.train()
        teacher_model.eval()
        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, labels in tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs}'):
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            with torch.no_grad():
                teacher_logits = teacher_model(inputs)

            student_logits = student_model(inputs)

            loss = distill_criterion(student_logits, teacher_logits, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)

            _, predicted = torch.max(student_logits.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        epoch_loss = running_loss / len(train_loader.dataset)
        train_acc = correct / total

        # 验证阶段
        student_model.eval()
        val_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)

                student_logits = student_model(inputs)
                teacher_logits = teacher_model(inputs)
                loss = distill_criterion(student_logits, teacher_logits, labels)

                val_loss += loss.item() * inputs.size(0)

                _, predicted = torch.max(student_logits.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        val_loss = val_loss / len(val_loader.dataset)
        val_acc = correct / total

        # 学习率调度
        if scheduler:
            scheduler.step(val_loss)

        print(f'Epoch {epoch + 1}/{num_epochs}')
        print(f'Train Loss: {epoch_loss:.4f}, Train Acc: {train_acc:.4f}')
        print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')

        # 保存最佳模型
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(student_model.state_dict(), 'outputs/models/best_distilled_model.pth')

        # 保存最新模型
        torch.save(student_model.state_dict(), 'outputs/models/latest_distilled_model.pth')

    return student_model


def main():
    parser = argparse.ArgumentParser(description='Model Distillation for Cervical Cancer Detection')
    parser.add_argument('--config', type=str, default='configs/default.yaml', help='Path to config file')
    parser.add_argument('--teacher_model_path', type=str, required=True, help='Path to teacher model')
    parser.add_argument('--temperature', type=float, default=3.0, help='Distillation temperature')
    parser.add_argument('--alpha', type=float, default=0.7, help='Distillation loss weight')

    args = parser.parse_args()

    # 加载配置
    config = load_config(args.config)

    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # 创建数据加载器
    train_loader, val_loader, test_loader = create_data_loaders(config)

    # 加载教师模型（大型模型）
    teacher_model = CervicalCancerModel(
        num_classes=config['model']['num_classes'],
        pretrained=False,
        dropout_rate=config['model']['dropout_rate'],
        model_type='resnet101'  # 使用更大的模型作为教师
    )
    teacher_model.load_state_dict(torch.load(args.teacher_model_path, map_location=device))
    teacher_model.to(device)
    teacher_model.eval()

    # 初始化学生模型（小型模型）
    student_model = CervicalCancerModel(
        num_classes=config['model']['num_classes'],
        pretrained=False,
        dropout_rate=config['model']['dropout_rate'],
        model_type='resnet50'  # 使用较小的模型作为学生
    ).to(device)

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        student_model.parameters(),
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

    # 训练学生模型
    student_model = train_student_model(
        teacher_model, student_model, train_loader, val_loader,
        criterion, optimizer, scheduler,
        config['training']['epochs'], device,
        args.temperature, args.alpha
    )

    print("Distillation training completed!")


if __name__ == '__main__':
    main()