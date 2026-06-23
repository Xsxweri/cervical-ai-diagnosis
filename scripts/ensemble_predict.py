import sys
import os
import torch
import torch.nn as nn
import numpy as np
from torchvision import transforms
import cv2
import argparse
import json
from collections import OrderedDict
from typing import List, Dict, Tuple

# 添加项目根目录到Python路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.insert(0, project_root)

from models import CervicalCancerModel


class EnsembleModel(nn.Module):
    """
    模型集成类，支持多种集成策略
    """

    def __init__(self, model_paths: List[str], config: Dict, ensemble_method: str = 'average'):
        """
        初始化集成模型

        Args:
            model_paths: 模型路径列表
            config: 配置字典
            ensemble_method: 集成方法 ('average', 'weighted', 'majority_vote')
        """
        super(EnsembleModel, self).__init__()
        self.ensemble_method = ensemble_method
        self.models = nn.ModuleList()
        self.weights = []

        # 加载所有模型
        for model_path in model_paths:
            model = CervicalCancerModel(
                num_classes=config['model']['num_classes'],
                pretrained=False,
                dropout_rate=config['model']['dropout_rate'],
                model_type=config['model'].get('model_type', 'resnet50')
            )

            # 加载模型权重
            state_dict = torch.load(model_path, map_location='cpu')
            # 处理可能的键不匹配问题
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                if k.startswith('module.'):
                    name = k[7:]  # remove 'module.' prefix
                else:
                    name = k
                new_state_dict[name] = v

            model.load_state_dict(new_state_dict)
            self.models.append(model)

        # 初始化权重（如果使用加权平均）
        if self.ensemble_method == 'weighted':
            self.weights = [1.0 / len(self.models)] * len(self.models)

        print(f"成功加载 {len(self.models)} 个模型进行集成")

    def forward(self, x):
        """
        前向传播

        Args:
            x: 输入张量

        Returns:
            集成后的输出
        """
        outputs = []
        for model in self.models:
            model.eval()
            with torch.no_grad():
                output = model(x)
                outputs.append(output)

        # 根据集成方法组合输出
        if self.ensemble_method == 'average':
            # 简单平均
            return torch.mean(torch.stack(outputs), dim=0)
        elif self.ensemble_method == 'weighted':
            # 加权平均
            weighted_outputs = []
            for output, weight in zip(outputs, self.weights):
                weighted_outputs.append(output * weight)
            return torch.sum(torch.stack(weighted_outputs), dim=0)
        elif self.ensemble_method == 'majority_vote':
            # 多数投票（对分类任务）
            # 这里我们仍然使用概率平均，因为多数投票更适合离散预测
            return torch.mean(torch.stack(outputs), dim=0)
        else:
            raise ValueError(f"不支持的集成方法: {self.ensemble_method}")


def predict_with_ensemble(model, image_path, config, device):
    """
    使用集成模型进行预测

    Args:
        model: 集成模型
        image_path: 图像路径
        config: 配置字典
        device: 计算设备

    Returns:
        预测结果和置信度
    """
    # 预处理图像
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((config['data']['img_size'], config['data']['img_size'])),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"无法读取图像: {image_path}")

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_tensor = transform(image).unsqueeze(0).to(device)

    # 预测
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probabilities, 1)

    # 类别名称映射
    class_names = [
        'carcinoma_in_situ', 'light_dysplastic', 'moderate_dysplastic',
        'normal_columnar', 'normal_intermediate', 'normal_superficial', 'severe_dysplastic'
    ]

    return class_names[predicted.item()], confidence.item(), probabilities[0].cpu().numpy()


def optimize_weights(model_paths: List[str], config: Dict, val_loader, device):
    """
    使用验证集优化模型权重

    Args:
        model_paths: 模型路径列表
        config: 配置字典
        val_loader: 验证数据加载器
        device: 计算设备

    Returns:
        优化后的权重列表
    """
    from scipy.optimize import minimize
    import torch.nn.functional as F

    # 加载所有模型
    models = []
    for model_path in model_paths:
        model = CervicalCancerModel(
            num_classes=config['model']['num_classes'],
            pretrained=False,
            dropout_rate=config['model']['dropout_rate'],
            model_type=config['model'].get('model_type', 'resnet50')
        )
        state_dict = torch.load(model_path, map_location='cpu')
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            if k.startswith('module.'):
                name = k[7:]
            else:
                name = k
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict)
        model.to(device)
        models.append(model)

    # 定义优化目标函数
    def objective(weights):
        # 归一化权重
        weights = np.array(weights)
        weights = weights / np.sum(weights)

        # 计算验证集上的损失
        total_loss = 0.0
        total_samples = 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # 集成预测
                outputs_list = [model(inputs) for model in models]
                ensemble_output = torch.zeros_like(outputs_list[0])
                for output, weight in zip(outputs_list, weights):
                    ensemble_output += output * weight

                # 计算交叉熵损失
                loss = F.cross_entropy(ensemble_output, labels, reduction='sum')
                total_loss += loss.item()
                total_samples += labels.size(0)

        return total_loss / total_samples

    # 初始化权重
    initial_weights = np.ones(len(models)) / len(models)

    # 约束条件：权重和为1
    constraints = {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}

    # 边界条件：权重非负
    bounds = [(0, 1) for _ in range(len(models))]

    # 优化权重
    result = minimize(objective, initial_weights, method='SLSQP',
                      bounds=bounds, constraints=constraints)

    optimized_weights = result.x / np.sum(result.x)  # 确保归一化
    return optimized_weights.tolist()


def main():
    parser = argparse.ArgumentParser(description='使用集成模型进行宫颈癌分类预测')
    parser.add_argument('--image_path', type=str, required=True, help='待预测图像路径')
    parser.add_argument('--model_paths', type=str, nargs='+', required=True, help='模型路径列表')
    parser.add_argument('--config', type=str, default='configs/default.yaml', help='配置文件路径')
    parser.add_argument('--ensemble_method', type=str, default='average',
                        choices=['average', 'weighted', 'majority_vote'], help='集成方法')
    parser.add_argument('--optimize_weights', action='store_true', help='是否优化权重')
    parser.add_argument('--val_data_path', type=str, help='验证数据路径（用于权重优化）')

    args = parser.parse_args()

    # 加载配置
    from utils import load_config
    config = load_config(args.config)

    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'使用设备: {device}')

    # 创建集成模型
    ensemble_model = EnsembleModel(args.model_paths, config, args.ensemble_method)
    ensemble_model.to(device)

    # 优化权重（如果需要）
    if args.optimize_weights and args.val_data_path:
        print("正在优化模型权重...")
        # 这里需要根据您的数据加载器实现来加载验证数据
        # 例如:
        # from utils import create_data_loaders
        # _, val_loader, _ = create_data_loaders(config)
        # weights = optimize_weights(args.model_paths, config, val_loader, device)
        # ensemble_model.weights = weights
        print("权重优化完成")

    # 进行预测
    try:
        prediction, confidence, probabilities = predict_with_ensemble(
            ensemble_model, args.image_path, config, device)

        print(f'预测结果: {prediction}')
        print(f'置信度: {confidence:.4f}')
        print('各类别概率:')

        class_names = [
            'carcinoma_in_situ', 'light_dysplastic', 'moderate_dysplastic',
            'normal_columnar', 'normal_intermediate', 'normal_superficial', 'severe_dysplastic'
        ]

        for i, (class_name, prob) in enumerate(zip(class_names, probabilities)):
            print(f'  {class_name}: {prob:.4f}')

    except Exception as e:
        print(f"预测过程中发生错误: {e}")
        return 1

    return 0


if __name__ == '__main__':
    sys.exit(main())