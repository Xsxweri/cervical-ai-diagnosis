import torch
import torch.nn as nn
from torchvision import transforms
import cv2
import numpy as np
import argparse
import os
from PIL import Image

from models import CervicalCancerModel
from utils import load_config, visualize_prediction


def predict_image(model, image_path, device, config):
    """预测单个图像"""
    # 预处理图像
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((config['data']['img_size'], config['data']['img_size'])),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_tensor = transform(image).unsqueeze(0).to(device)

    # 预测
    model.eval()
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probabilities, 1)

    # 类别名称映射
    class_names = [
        'carcinoma_in_situ', 'light_dysplastic', 'moderate_dysplastic',
        'normal_columnar', 'normal_intermediate', 'normal_superficial', 'severe_dysplastic'
    ]

    return class_names[predicted.item()], confidence.item()


def main():
    parser = argparse.ArgumentParser(description='Predict cervical cancer from an image')
    parser.add_argument('--image_path', type=str, required=True, help='Path to the image to predict')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the trained model')
    parser.add_argument('--config', type=str, default='configs/default.yaml', help='Path to config file')

    args = parser.parse_args()

    # 加载配置
    config = load_config(args.config)

    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 加载模型
    model = CervicalCancerModel(
        num_classes=config['model']['num_classes'],
        pretrained=False,
        dropout_rate=config['model']['dropout_rate']
    )
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.to(device)

    # 进行预测
    prediction, confidence = predict_image(model, args.image_path, device, config)

    print(f'Prediction: {prediction}')
    print(f'Confidence: {confidence:.4f}')

    # 可视化结果
    visualize_prediction(args.image_path, prediction, confidence,
                         os.path.join(config['paths']['result_dir'], 'prediction_result.png'))


if __name__ == '__main__':
    main()