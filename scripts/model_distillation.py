import argparse
import json
import os

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

# 项目已通过 pip install -e . 安装，项目根由 .pth 提供，无需 sys.path 自举
from models import CervicalCancerModel
from utils import create_data_loaders, load_config
from utils.label_mapping import get_multiclass_classes


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
    os.makedirs('outputs/models', exist_ok=True)
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

    best_path = 'outputs/models/best_distilled_model.pth'
    if os.path.isfile(best_path):
        student_model.load_state_dict(torch.load(best_path, map_location=device))
        print(f'已加载最佳权重(val_acc={best_acc:.4f})')

    return student_model

def evaluate_and_report(teacher_model, student_model, test_loader, device, num_classes):
    """
    评估模型并生成报告
    """
    from scripts.evaluate import (
        calculate_auc_scores,
        calculate_f1_scores,
        calculate_precision_recall,
        evaluate_model,
    )

    print('\n=== 测试集评估 ===')
    results = {}
    for name, model in (('teacher', teacher_model), ('student', student_model)):
        model.eval()
        preds, labels, probs = evaluate_model(model, test_loader, device, num_classes)
        acc = float((preds == labels).mean())
        auc_macro, auc_micro = calculate_auc_scores(labels, probs, num_classes)
        f1_per, f1_macro, f1_micro = calculate_f1_scores(labels, preds, num_classes)
        precision, recall = calculate_precision_recall(labels, preds, num_classes)

        results[name] = {
            'accuracy': acc,
            'f1_macro': float(f1_macro),
            'f1_micro': float(f1_micro),
            'precision_macro': float(precision),
            'recall_macro': float(recall),
            'f1_per_class': f1_per.tolist() if hasattr(f1_per, 'tolist') else list(f1_per),
            'auc_macro': auc_macro,
            'auc_micro': auc_micro,
        }
        auc_txt = f' AUC={auc_macro:.4f}' if auc_macro is not None else ''
        print(f'[{name}] acc={acc:.4f} macro-F1={f1_macro:.4f} '
              f'precision={precision:.4f} recall={recall:.4f}{auc_txt}')

    t, s = results['teacher']['accuracy'], results['student']['accuracy']
    keep = s / t if t > 0 else 0.0
    results['retention'] = {
        'accuracy_ratio': keep,
        'accuracy_gap': t - s,
        'f1_macro_gap': results['teacher']['f1_macro'] - results['student']['f1_macro'],
    }
    print(f'\n性能保留率: {keep:.2%}(acc gap {t - s:+.4f})')

    out_dir = 'outputs/results'
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'distillation_eval.json')
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f'评估结果已保存: {out_path}')
    return results


def main():
    parser = argparse.ArgumentParser(description='Model Distillation for Cervical Cancer Detection')
    parser.add_argument('--config', type=str, default='configs/default.yaml', help='Path to config file')
    parser.add_argument('--teacher_model_path', type=str, required=True, help='Path to teacher model')
    parser.add_argument('--temperature', type=float, default=3.0, help='Distillation temperature')
    parser.add_argument('--alpha', type=float, default=0.7, help='Distillation loss weight')
    parser.add_argument('--classification_mode', type=str, default='multiclass',
                        choices=['binary', 'multiclass'],
                        help='分类模式；必须与 config 的 num_classes 一致')
    parser.add_argument('--label_space', type=str, default=None,
                        choices=['sipakmed', 'herlev'],
                        help='多分类标签空间(multiclass 时必填)')
    parser.add_argument('--allow_same_arch', action='store_true',
                        help='允许教师与学生同为 resnet50(自蒸馏，保留率指标无意义)')

    args = parser.parse_args()

    # 加载配置
    config = load_config(args.config)

    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # 创建数据加载器
    mode = args.classification_mode
    if mode == 'binary' and config['model']['num_classes'] != 2:
        raise SystemExit(
            f"配置矛盾：classification_mode=binary 但 num_classes="
            f"{config['model']['num_classes']}。改 configs 或加 --classification_mode multiclass"
        )
    if mode == 'multiclass' and not args.label_space:
        raise SystemExit("multiclass 模式必须指定 --label_space")

    if mode == 'multiclass':
        expected = len(get_multiclass_classes(args.label_space))
        if expected != config['model']['num_classes']:
            raise SystemExit(
                f"配置矛盾: label_space={args.label_space} 有 {expected} 类，"
                f"但 config.num_classes={config['model']['num_classes']}"
            )
    
    # 加载教师模型（大型模型）
    from scripts.evaluate import detect_model_type_from_state_dict

    if not os.path.isfile(args.teacher_model_path):
        raise SystemExit(f"未找到教师模型: {args.teacher_model_path}")


    teacher_state = torch.load(args.teacher_model_path, map_location=device)
    teacher_type = detect_model_type_from_state_dict(teacher_state)

    attention_keys = [k for k in teacher_state if 'attention' in k]
    if attention_keys:
        use_attention = True
        # 先判 CBAM（更具体：ca. / sa.），否则 SE
        attention_type = 'cbam' if any(
            'ca.' in k or 'sa.' in k or 'cbam' in k.lower() for k in attention_keys
        ) else 'se'
        print(f'教师模型: {teacher_type} + 注意力 {attention_type}')
    else:
        use_attention, attention_type = False, None
        print(f'教师模型类型（从权重探测）: {teacher_type}')

    teacher_model = CervicalCancerModel(
        num_classes=config['model']['num_classes'],
        pretrained=False,
        dropout_rate=config['model']['dropout_rate'],
        model_type=teacher_type,
        use_attention=use_attention,
        attention_type=attention_type
    )
    try:
        teacher_model.load_state_dict(teacher_state, strict=True)
    except RuntimeError as e:
        print(f'教师权重严格加载失败，改用 strict=False(部分权重随机初始化！): {e}')
        teacher_model.load_state_dict(teacher_state, strict=False)
    del teacher_state
    teacher_model.to(device)
    teacher_model.eval()

    if teacher_type == 'resnet50' and not args.allow_same_arch:
        raise SystemExit(
            "教师与学生同为 resnet50, 蒸馏退化为同架构自蒸馏，保留率指标无意义。\n"
            "  请改用 resnet101 / efficientnet_b2 或集成模型权重作为教师。\n"
            "  若确实要做自蒸馏（正则效果），加 --allow_same_arch 显式放行。"
        )

    train_loader, val_loader, test_loader = create_data_loaders(config, classification_mode=mode, label_space=args.label_space)


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

    # 评估模型
    evaluate_and_report(
        teacher_model, student_model, test_loader, device, config['model']['num_classes'],
    )

    print("Distillation training completed!")


if __name__ == '__main__':
    main()
