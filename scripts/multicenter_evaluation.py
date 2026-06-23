import sys
import os
import torch
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import argparse
import json
from collections import defaultdict

# 添加项目根目录到Python路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.insert(0, project_root)

from models import CervicalCancerModel
from utils import load_config


class MultiCenterEvaluator:
    """
    多中心模型评估器
    """

    def __init__(self, model_path, config):
        """
        初始化评估器

        Args:
            model_path: 模型路径
            config: 配置字典
        """
        self.model_path = model_path
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # 加载模型
        self.model = self._load_model()

        # 类别名称
        self.class_names = [
            'carcinoma_in_situ', 'light_dysplastic', 'moderate_dysplastic',
            'normal_columnar', 'normal_intermediate', 'normal_superficial', 'severe_dysplastic'
        ]

    def _load_model(self):
        """
        加载模型
        """
        model = CervicalCancerModel(
            num_classes=self.config['model']['num_classes'],
            pretrained=False,
            dropout_rate=self.config['model']['dropout_rate'],
            model_type=self.config['model'].get('model_type', 'resnet50')
        )

        state_dict = torch.load(self.model_path, map_location=self.device)
        model.load_state_dict(state_dict)
        model.to(self.device)
        model.eval()

        return model

    def evaluate_center(self, data_loader, center_name):
        """
        在特定中心的数据上评估模型

        Args:
            data_loader: 数据加载器
            center_name: 中心名称

        Returns:
            评估结果字典
        """
        print(f"正在评估中心: {center_name}")

        all_preds = []
        all_labels = []

        with torch.no_grad():
            for inputs, labels in data_loader:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(inputs)
                _, preds = torch.max(outputs, 1)

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        # 计算评估指标
        accuracy = accuracy_score(all_labels, all_preds)
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_preds, average='weighted')

        # 计算每个类别的指标
        class_precision, class_recall, class_f1, _ = precision_recall_fscore_support(
            all_labels, all_preds, average=None)

        # 混淆矩阵
        cm = confusion_matrix(all_labels, all_preds)

        results = {
            'center_name': center_name,
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1),
            'class_metrics': {
                self.class_names[i]: {
                    'precision': float(class_precision[i]),
                    'recall': float(class_recall[i]),
                    'f1_score': float(class_f1[i])
                } for i in range(len(self.class_names))
            },
            'confusion_matrix': cm.tolist()
        }

        print(f"中心 {center_name} 评估完成:")
        print(f"  准确率: {accuracy:.4f}")
        print(f"  精确率: {precision:.4f}")
        print(f"  召回率: {recall:.4f}")
        print(f"  F1分数: {f1:.4f}")

        return results

    def evaluate_all_centers(self, center_data_loaders):
        """
        在所有中心上评估模型

        Args:
            center_data_loaders: 各中心的数据加载器字典

        Returns:
            所有中心的评估结果
        """
        all_results = {}
        overall_metrics = defaultdict(list)

        for center_name, data_loader in center_data_loaders.items():
            result = self.evaluate_center(data_loader, center_name)
            all_results[center_name] = result

            # 收集总体指标用于计算统计信息
            overall_metrics['accuracy'].append(result['accuracy'])
            overall_metrics['precision'].append(result['precision'])
            overall_metrics['recall'].append(result['recall'])
            overall_metrics['f1_score'].append(result['f1_score'])

        # 计算总体统计信息
        summary = {
            'individual_centers': all_results,
            'overall_statistics': {
                'mean': {
                    'accuracy': float(np.mean(overall_metrics['accuracy'])),
                    'precision': float(np.mean(overall_metrics['precision'])),
                    'recall': float(np.mean(overall_metrics['recall'])),
                    'f1_score': float(np.mean(overall_metrics['f1_score']))
                },
                'std': {
                    'accuracy': float(np.std(overall_metrics['accuracy'])),
                    'precision': float(np.std(overall_metrics['precision'])),
                    'recall': float(np.std(overall_metrics['recall'])),
                    'f1_score': float(np.std(overall_metrics['f1_score']))
                },
                'min': {
                    'accuracy': float(np.min(overall_metrics['accuracy'])),
                    'precision': float(np.min(overall_metrics['precision'])),
                    'recall': float(np.min(overall_metrics['recall'])),
                    'f1_score': float(np.min(overall_metrics['f1_score']))
                },
                'max': {
                    'accuracy': float(np.max(overall_metrics['accuracy'])),
                    'precision': float(np.max(overall_metrics['precision'])),
                    'recall': float(np.max(overall_metrics['recall'])),
                    'f1_score': float(np.max(overall_metrics['f1_score']))
                }
            }
        }

        return summary


def create_center_data_loaders(center_paths, config):
    """
    为各个中心创建数据加载器

    Args:
        center_paths: 各中心数据路径字典
        config: 配置字典

    Returns:
        各中心的数据加载器字典
    """
    # 这是一个示例实现，您需要根据实际数据结构进行调整
    center_loaders = {}

    # 注意：这需要根据您的实际数据加载逻辑进行实现
    # 以下是一个示例框架
    """
    from utils.preprocess import CombinedCervicalCancerDataset, get_transforms
    from torch.utils.data import DataLoader

    test_transform = get_transforms(config, 'test')

    for center_name, data_path in center_paths.items():
        # 根据您的数据集格式加载数据
        dataset = CombinedCervicalCancerDataset(
            [{'type': 'custom', 'path': data_path}], 
            transform=test_transform, 
            config=config
        )

        loader = DataLoader(
            dataset, 
            batch_size=config['data']['batch_size'], 
            shuffle=False, 
            num_workers=4
        )

        center_loaders[center_name] = loader
    """

    # 由于需要具体的数据集实现，这里返回空字典作为示例
    return center_loaders


def main():
    parser = argparse.ArgumentParser(description='多中心模型评估')
    parser.add_argument('--model_path', type=str, required=True, help='模型路径')
    parser.add_argument('--config', type=str, default='configs/default.yaml', help='配置文件路径')
    parser.add_argument('--center_paths', type=str, nargs='+', required=True,
                        help='中心名称和路径对，格式: center1:/path/to/data1 center2:/path/to/data2')
    parser.add_argument('--output_path', type=str, default='outputs/multicenter_evaluation.json',
                        help='评估结果保存路径')

    args = parser.parse_args()

    # 解析中心路径
    center_paths = {}
    for item in args.center_paths:
        if ':' in item:
            center_name, path = item.split(':', 1)
            center_paths[center_name] = path
        else:
            print(f"警告: 无效的中心路径格式: {item}")

    if not center_paths:
        print("错误: 至少需要指定一个中心路径")
        return 1

    # 加载配置
    config = load_config(args.config)

    # 创建评估器
    evaluator = MultiCenterEvaluator(args.model_path, config)

    # 创建各中心的数据加载器
    print("正在创建各中心数据加载器...")
    center_data_loaders = create_center_data_loaders(center_paths, config)

    if not center_data_loaders:
        print("警告: 未能创建任何数据加载器，请检查数据路径和格式")
        print("您需要根据实际数据结构修改 create_center_data_loaders 函数")
        return 1

    # 执行评估
    print("开始多中心评估...")
    results = evaluator.evaluate_all_centers(center_data_loaders)

    # 保存结果
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    with open(args.output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"评估完成，结果已保存到: {args.output_path}")

    # 打印摘要
    print("\n=== 多中心评估摘要 ===")
    print(f"评估中心数: {len(results['individual_centers'])}")
    print(f"平均准确率: {results['overall_statistics']['mean']['accuracy']:.4f} "
          f"(±{results['overall_statistics']['std']['accuracy']:.4f})")
    print(f"平均F1分数: {results['overall_statistics']['mean']['f1_score']:.4f} "
          f"(±{results['overall_statistics']['std']['f1_score']:.4f})")

    print("\n各中心详细结果:")
    for center_name, center_result in results['individual_centers'].items():
        print(f"  {center_name}: 准确率={center_result['accuracy']:.4f}, "
              f"F1分数={center_result['f1_score']:.4f}")

    return 0


if __name__ == '__main__':
    sys.exit(main())