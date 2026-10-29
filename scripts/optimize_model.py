import sys
import os
import torch
from torch.quantization import quantize_dynamic
import argparse
from collections import OrderedDict

# 项目已通过 pip install -e . 安装，项目根由 .pth 提供，无需 sys.path 自举
from models import CervicalCancerModel


def quantize_model(model_path, output_path, config):
    """
    对模型进行量化以减小模型大小和提高推理速度

    Args:
        model_path: 原始模型路径
        output_path: 量化后模型保存路径
        config: 配置字典
    """
    print("正在加载模型...")

    # 加载模型权重
    state_dict = torch.load(model_path, map_location='cpu')
    state_dict = OrderedDict(
        (k[7:] if k.startswith('module.') else k, v) for k, v in state_dict.items()
    )

    use_attention = any(k.startswith('attention.') or '.attention.' in k for k in state_dict)
    attention_type = None
    if use_attention:
        if any('attention.ca.' in k or 'attention.sa.' in k for k in state_dict):
            attention_type = 'cbam'
        elif any('attention.fc.' in k for k in state_dict):
            attention_type = 'se'

    fc_w = state_dict.get('fc.weight')
    if fc_w is None:
        raise KeyError("state_dict 缺少 fc.weight，无法确定类别数")
    num_classes = int(fc_w.shape[0])

    # 创建模型实例
    model = CervicalCancerModel(
        num_classes=num_classes,
        pretrained=False,
        dropout_rate=config['model']['dropout_rate'],
        model_type=CervicalCancerModel.detect_model_type(state_dict),
        use_attention=use_attention,
        attention_type=attention_type,
    )

    model.load_state_dict(state_dict, strict=True)
    model.eval()

    print(f"原始模型大小: {os.path.getsize(model_path) / (1024 * 1024):.2f} MB")

    # 对模型进行动态量化
    print("正在进行动态量化...")
    quantized_model = quantize_dynamic(
        model,
        {torch.nn.Linear, torch.nn.Conv2d},
        dtype=torch.qint8
    )

    # 保存量化后的模型
    torch.save(quantized_model.state_dict(), output_path)
    print(f"量化完成，模型已保存到: {output_path}")
    print(f"量化后模型大小: {os.path.getsize(output_path) / (1024 * 1024):.2f} MB")

    # 计算压缩率
    original_size = os.path.getsize(model_path)
    quantized_size = os.path.getsize(output_path)
    compression_ratio = (1 - quantized_size / original_size) * 100
    print(f"模型压缩率: {compression_ratio:.2f}%")


def convert_to_torchscript(model_path, output_path, config, example_input_shape=(1, 3, 224, 224)):
    """
    将模型转换为TorchScript格式以提高推理效率

    Args:
        model_path: 原始模型路径
        output_path: TorchScript模型保存路径
        config: 配置字典
        example_input_shape: 示例输入形状
    """
    print("正在加载模型...")

    # 创建模型实例
    model = CervicalCancerModel(
        num_classes=config['model']['num_classes'],
        pretrained=False,
        dropout_rate=config['model']['dropout_rate'],
        model_type=config['model'].get('model_type', 'resnet50')
    )

    # 加载模型权重
    state_dict = torch.load(model_path, map_location='cpu')
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if k.startswith('module.'):
            name = k[7:]  # remove 'module.' prefix
        else:
            name = k
        new_state_dict[name] = v

    model.load_state_dict(new_state_dict)
    model.eval()

    # 创建示例输入
    example_input = torch.rand(example_input_shape)

    # 转换为TorchScript
    print("正在转换为TorchScript...")
    traced_script_module = torch.jit.trace(model, example_input)

    # 保存TorchScript模型
    traced_script_module.save(output_path)
    print(f"TorchScript模型已保存到: {output_path}")
    print(f"TorchScript模型大小: {os.path.getsize(output_path) / (1024 * 1024):.2f} MB")


def main():
    parser = argparse.ArgumentParser(description='模型优化工具')
    parser.add_argument('--model_path', type=str, required=True, help='原始模型路径')
    parser.add_argument('--output_path', type=str, required=True, help='优化后模型保存路径')
    parser.add_argument('--config', type=str, default='configs/default.yaml', help='配置文件路径')
    parser.add_argument('--method', type=str, default='quantize',
                        choices=['quantize', 'torchscript'], help='优化方法')
    parser.add_argument('--input_shape', type=int, nargs=4, default=[1, 3, 224, 224],
                        help='示例输入形状 [batch, channels, height, width]')

    args = parser.parse_args()

    # 加载配置
    from utils import load_config
    config = load_config(args.config)

    if args.method == 'quantize':
        quantize_model(args.model_path, args.output_path, config)
    elif args.method == 'torchscript':
        convert_to_torchscript(args.model_path, args.output_path, config, tuple(args.input_shape))
    else:
        print(f"不支持的优化方法: {args.method}")
        return 1

    return 0


if __name__ == '__main__':
    sys.exit(main())
