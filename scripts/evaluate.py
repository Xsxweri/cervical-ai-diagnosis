import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, roc_curve
import argparse
import json
import matplotlib.pyplot as plt
import logging

# 添加项目根目录到Python路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.insert(0, project_root)

# 现在可以正常导入项目模块
from utils import create_data_loaders, load_config
from models.cer_cancer_model import CervicalCancerModel


class EMA:
    """
    Exponential Moving Average implementation for model parameters
    与训练脚本中的EMA类保持一致，用于评估EMA模型
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


def mc_dropout_predict(model, inputs, n_samples=10):
    """蒙特卡洛Dropout预测"""
    model.train()  # 保持Dropout开启
    predictions = []

    with torch.no_grad():
        for _ in range(n_samples):
            output = model(inputs)
            prob = F.softmax(output, dim=1)
            predictions.append(prob.cpu().numpy())

    predictions = np.array(predictions)
    mean_pred = np.mean(predictions, axis=0)
    std_pred = np.std(predictions, axis=0)
    return mean_pred, std_pred


def evaluate_model_with_uncertainty(model, test_loader, device, num_classes, num_mc_samples=10):
    """使用蒙特卡洛Dropout进行模型评估以估计不确定性"""
    all_preds = []
    all_labels = []
    all_probs = []
    all_std_probs = []  # 存储不确定性

    for inputs, labels in test_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        # 使用蒙特卡洛Dropout预测
        mean_probs, std_probs = mc_dropout_predict(model, inputs, n_samples=num_mc_samples)
        preds = np.argmax(mean_probs, axis=1)

        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())
        all_probs.extend(mean_probs)
        all_std_probs.extend(std_probs)

    return np.array(all_preds), np.array(all_labels), np.array(all_probs), np.array(all_std_probs)


def evaluate_model(model, test_loader, device, num_classes):
    """评估模型性能"""
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    return np.array(all_preds), np.array(all_labels), np.array(all_probs)


def calculate_auc_scores(labels, probs, num_classes):
    """计算AUC分数"""
    try:
        if num_classes == 2:
            # 二分类AUC
            if probs.shape[1] >= 2:
                auc_score = roc_auc_score(labels, probs[:, 1])
                return auc_score
            else:
                logging.warning("Not enough columns in probs for binary AUC calculation")
                return None
        else:
            # 多分类AUC (macro和micro平均)
            if probs.shape[1] >= num_classes:
                auc_macro = roc_auc_score(labels, probs, multi_class='ovr', average='macro')
                auc_micro = roc_auc_score(labels, probs, multi_class='ovr', average='micro')
                return auc_macro, auc_micro
            else:
                logging.warning(
                    f"Probs only has {probs.shape[1]} columns, but {num_classes} classes expected for AUC calculation")
                return None, None
    except ValueError as e:
        logging.warning(f"Could not calculate AUC score: {e}")
        return None if num_classes == 2 else (None, None)
    except Exception as e:
        logging.warning(f"Unexpected error in AUC calculation: {e}")
        return None if num_classes == 2 else (None, None)


def calculate_f1_scores(labels, preds, num_classes):
    """计算每个类别的F1分数和宏观/微观平均"""
    from sklearn.metrics import f1_score
    f1_per_class = f1_score(labels, preds, average=None, zero_division=0)
    f1_macro = f1_score(labels, preds, average='macro', zero_division=0)
    f1_micro = f1_score(labels, preds, average='micro', zero_division=0)
    return f1_per_class, f1_macro, f1_micro


def calculate_precision_recall(labels, preds, num_classes):
    """计算精确率和召回率"""
    from sklearn.metrics import precision_score, recall_score
    precision = precision_score(labels, preds, average='macro', zero_division=0)
    recall = recall_score(labels, preds, average='macro', zero_division=0)
    return precision, recall


def plot_roc_curves(labels, probs, num_classes, target_names, save_path):
    """绘制ROC曲线"""
    plt.figure(figsize=(10, 8))

    if num_classes == 2:
        # 二分类ROC曲线
        if probs.shape[1] >= 2:
            fpr, tpr, _ = roc_curve(labels, probs[:, 1])
            auc_score = roc_auc_score(labels, probs[:, 1])
            plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc_score:.4f})')
    else:
        # 多分类ROC曲线
        n_classes_in_probs = probs.shape[1]
        for i in range(min(num_classes, n_classes_in_probs)):
            class_labels = (labels == i).astype(int)
            if len(np.unique(class_labels)) > 1:  # 确保有足够的样本
                fpr, tpr, _ = roc_curve(class_labels, probs[:, i])
                auc_score = roc_auc_score(class_labels, probs[:, i])
                class_name = target_names[i] if i < len(target_names) else f"Class_{i}"
                plt.plot(fpr, tpr, label=f'{class_name} (AUC = {auc_score:.4f})')

    plt.plot([0, 1], [0, 1], 'k--', label='Random')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def get_target_names(num_classes, classification_mode):
    """获取类别名称"""
    if classification_mode == 'binary':
        return ['abnormal', 'normal']
    else:
        all_target_names = [
            'carcinoma_in_situ', 'light_dysplastic', 'moderate_dysplastic',
            'normal_columnar', 'normal_intermediate', 'normal_superficial', 'severe_dysplastic'
        ]
        if num_classes <= len(all_target_names):
            return all_target_names[:num_classes]
        else:
            return [f"Class_{i}" for i in range(num_classes)]


def detect_model_type_from_state_dict(state_dict_keys):
    """从state_dict的键名中检测模型类型"""
    keys = list(state_dict_keys)

    # ViT: 有 patch_embed, blocks, cls_token 等键
    if any('patch_embed' in k or 'blocks.' in k or 'cls_token' in k for k in keys):
        return 'vit'

    # EfficientNet: 有 features.0, _bn0 等键
    if any('features.0.' in k or '_bn0' in k for k in keys):
        return 'efficientnet_b0'

    # ResNet: 有 layer1, layer4 等键（ResNet都有 conv1，不能作为ViT判断依据！）
    if any('layer1.' in k for k in keys):
        # 判断是 ResNet50 还是 ResNet101
        # ResNet50: layer4 有 3 个 block (layer4.0, layer4.1, layer4.2)
        # ResNet101: layer4 有 22 个 block
        layer4_indices = set()
        for k in keys:
            if 'layer4.' in k:
                parts = k.split('.')
                try:
                    idx = parts[parts.index('layer4') + 1]
                    layer4_indices.add(int(idx))
                except (ValueError, IndexError):
                    continue

        max_idx = max(layer4_indices) if layer4_indices else -1
        if max_idx >= 22:
            return 'resnet101'
        else:
            return 'resnet50'

    # 默认 fallback
    return 'resnet50'


def create_model_for_evaluation(model_state_dict, num_classes, config, use_attention, attention_type,
                                force_model_type=None):
    """根据state_dict创建合适的模型"""
    # 从state_dict键名推断模型类型
    state_dict_keys = model_state_dict.keys()

    # 优先使用强制指定的模型类型
    if force_model_type is not None:
        detected_model_type = force_model_type
        print(f"Using specified model type: {detected_model_type}")
    else:
        detected_model_type = detect_model_type_from_state_dict(state_dict_keys)
        print(f"Auto-detected model type: {detected_model_type}")

    print(f"Creating model: type={detected_model_type}, attention={use_attention}, attn_type={attention_type}")

    # 创建对应的模型
    model = CervicalCancerModel(
        num_classes=num_classes,
        pretrained=False,
        dropout_rate=config['model']['dropout_rate'],
        model_type=detected_model_type,
        use_attention=use_attention,
        attention_type=attention_type
    )

    return model


def main():
    parser = argparse.ArgumentParser(description='Evaluate cervical cancer classification model')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the trained model')
    parser.add_argument('--config', type=str, default='configs/default.yaml', help='Path to config file')
    parser.add_argument('--plot_roc', action='store_true', help='Plot ROC curves')
    parser.add_argument('--use_ema', action='store_true', help='Use Exponential Moving Average weights for evaluation')
    parser.add_argument('--classification_mode', type=str, default=None,
                        choices=['binary', 'multiclass'],
                        help='Classification mode: binary (normal/abnormal) or multiclass')
    parser.add_argument('--use_mc_dropout', action='store_true',
                        help='Use Monte Carlo Dropout for uncertainty estimation')
    parser.add_argument('--mc_samples', type=int, default=10,
                        help='Number of Monte Carlo samples for uncertainty estimation')
    parser.add_argument('--model_type', type=str, default=None,choices=['resnet50', 'resnet101', 'efficientnet_b0', 'vit'],
                        help='Manually specify model type (auto-detect if not set)')
    parser.add_argument('--attention_type', type=str, default=None,choices=['se', 'cbam', 'none'],
                        help='Manually specify attention type (auto-detect if not set)')

    args = parser.parse_args()

    # 加载配置
    config = load_config(args.config)

    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 创建数据加载器
    _, _, test_loader = create_data_loaders(config, args.classification_mode)

    # 根据分类模式设置类别数
    if args.classification_mode == 'binary' or (
            args.classification_mode is None and config['classification']['mode'] == 'binary'):
        num_classes = 2
        classification_mode = 'binary'
    else:
        num_classes = config['model']['num_classes']
        classification_mode = 'multiclass'

    # 加载模型时检查是否需要启用注意力机制
    model_state_dict = torch.load(args.model_path, map_location=device)

    # 检查state_dict中是否包含注意力机制相关的键
    if args.attention_type is not None:
        use_attention = (args.attention_type != 'none')
        attention_type = args.attention_type if use_attention else None
        print(f"Using specified attention: {use_attention}, type: {attention_type}")
    else:
        # 自动检测注意力机制
        attention_keys = [key for key in model_state_dict.keys() if 'attention' in key]
        use_attention = len(attention_keys) > 0
        attention_type = None

        if use_attention:
            # 先判断CBAM（更具体），再判断SE
            if any('ca.' in key or 'sa.' in key or 'cbam' in key.lower() for key in attention_keys):
                attention_type = 'cbam'
                print(f"Auto-detected attention: CBAM")
            elif any('se' in key.lower() for key in attention_keys):
                attention_type = 'se'
                print(f"Auto-detected attention: SE")
            else:
                attention_type = 'se'
                print(f"Auto-detected attention: SE (default)")
        else:
            print("No attention mechanism detected")

    # 根据模型权重自动检测模型类型并创建模型
    model = create_model_for_evaluation(
        model_state_dict, num_classes, config,
        use_attention, attention_type,
        force_model_type=args.model_type  # 传入命令行指定的模型类型
    )

    # 尝试加载模型权重
    try:
        model.load_state_dict(model_state_dict, strict=True)
        print("Model loaded successfully with strict=True")
    except RuntimeError as e:
        print(f"Error loading state_dict: {e}")
        print("Trying to load with strict=False... (some weights may be random!)")
        model.load_state_dict(model_state_dict, strict=False)

    model.to(device)

    # 如果使用EMA，应用EMA权重
    if args.use_ema:
        ema = EMA(model)
        ema.apply_shadow()
        print("Using EMA weights for evaluation")

    # 评估模型
    if args.use_mc_dropout:
        print(f"Using Monte Carlo Dropout with {args.mc_samples} samples for uncertainty estimation")
        preds, labels, probs, std_probs = evaluate_model_with_uncertainty(
            model, test_loader, device, num_classes, args.mc_samples)
    else:
        preds, labels, probs = evaluate_model(model, test_loader, device, num_classes)

    # 打印预测和标签的唯一值，了解实际类别数
    unique_preds = np.unique(preds)
    unique_labels = np.unique(labels)
    print(f"Unique predictions: {unique_preds}")
    print(f"Unique labels: {unique_labels}")
    print(f"Number of unique predictions: {len(unique_preds)}")
    print(f"Number of unique labels: {len(unique_labels)}")
    print(f"Max prediction value: {np.max(preds)}")
    print(f"Max label value: {np.max(labels)}")

    # 获取类别名称
    target_names = get_target_names(num_classes, classification_mode)
    print(f"Using target names: {target_names}")

    # 确保标签值在有效范围内
    if len(unique_labels) > num_classes or np.max(unique_labels) >= num_classes:
        print(f"Warning: Labels contain values outside expected range [0, {num_classes - 1}]")
        # 尝试修复标签
        labels = np.clip(labels, 0, num_classes - 1)

    # 计算评估指标
    labels_param = list(range(num_classes))
    cm = confusion_matrix(labels, preds, labels=labels_param)

    # 生成分类报告
    try:
        report = classification_report(
            labels, preds,
            labels=labels_param,
            target_names=target_names,
            output_dict=True,
            zero_division=0
        )
    except ValueError as e:
        print(f"Error generating classification report: {e}")
        # 如果仍有问题，使用默认类别名称
        report = classification_report(
            labels, preds,
            labels=labels_param,
            output_dict=True,
            zero_division=0
        )

    # 计算额外的评估指标
    f1_per_class, f1_macro, f1_micro = calculate_f1_scores(labels, preds, num_classes)
    precision, recall = calculate_precision_recall(labels, preds, num_classes)

    # 将额外指标添加到报告中
    report['f1_per_class'] = f1_per_class.tolist() if hasattr(f1_per_class, 'tolist') else f1_per_class
    report['f1_macro'] = f1_macro
    report['f1_micro'] = f1_micro
    report['precision_macro'] = precision
    report['recall_macro'] = recall

    # 计算AUC分数
    if num_classes == 2:
        auc_score = calculate_auc_scores(labels, probs, num_classes)
        if auc_score is not None:
            report['auc'] = auc_score
    else:
        auc_macro, auc_micro = calculate_auc_scores(labels, probs, num_classes)
        if auc_macro is not None and auc_micro is not None:
            report['auc_macro'] = auc_macro
            report['auc_micro'] = auc_micro

    # 打印结果
    print("Confusion Matrix:")
    print(cm)
    print("\nClassification Report:")
    try:
        print(classification_report(
            labels, preds,
            labels=labels_param,
            target_names=target_names,
            zero_division=0
        ))
    except:
        print(classification_report(labels, preds, labels=labels_param, zero_division=0))

    # 打印额外指标
    print(f"\nAdditional Metrics:")
    print(f"F1 Macro: {f1_macro:.4f}")
    print(f"F1 Micro: {f1_micro:.4f}")
    print(f"Precision (Macro): {precision:.4f}")
    print(f"Recall (Macro): {recall:.4f}")

    if num_classes == 2 and 'auc' in report:
        if report['auc'] is not None:
            print(f"AUC Score: {report['auc']:.4f}")
        else:
            print("AUC Score: N/A")
    elif num_classes > 2 and 'auc_macro' in report:
        if report['auc_macro'] is not None and report['auc_micro'] is not None:
            print(f"AUC Score (Macro): {report['auc_macro']:.4f}")
            print(f"AUC Score (Micro): {report['auc_micro']:.4f}")
        else:
            print("AUC Score (Macro): N/A")
            print("AUC Score (Micro): N/A")

    # 保存结果
    os.makedirs(config['paths']['result_dir'], exist_ok=True)

    # 保存混淆矩阵
    np.save(os.path.join(config['paths']['result_dir'], 'confusion_matrix.npy'), cm)

    # 保存分类报告
    with open(os.path.join(config['paths']['result_dir'], 'classification_report.json'), 'w') as f:
        json.dump(report, f, indent=4)

    # 如果使用蒙特卡洛dropout，保存不确定性信息
    if args.use_mc_dropout:
        uncertainty_data = {
            'mean_predictions': probs.tolist(),
            'std_predictions': std_probs.tolist(),
            'uncertainty_per_sample': np.mean(std_probs, axis=1).tolist()
        }
        with open(os.path.join(config['paths']['result_dir'], 'uncertainty_data.json'), 'w') as f:
            json.dump(uncertainty_data, f, indent=4)
        print(f"Uncertainty data saved to: {os.path.join(config['paths']['result_dir'], 'uncertainty_data.json')}")

    # 可视化混淆矩阵
    try:
        from utils.visualization import plot_confusion_matrix
        plot_confusion_matrix(
            cm,
            target_names,
            os.path.join(config['paths']['result_dir'], 'confusion_matrix.png')
        )
    except ImportError:
        print("Warning: Could not import visualization module. Skipping confusion matrix plot.")

    # 绘制ROC曲线
    if args.plot_roc:
        plot_roc_curves(
            labels, probs, num_classes, target_names,
            os.path.join(config['paths']['result_dir'], 'roc_curves.png')
        )

    print(f"Results saved to: {config['paths']['result_dir']}")


if __name__ == '__main__':
    main()