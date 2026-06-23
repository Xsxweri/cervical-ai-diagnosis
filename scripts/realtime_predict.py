import sys
import os
import torch
import time
import cv2
import numpy as np
from torchvision import transforms
import argparse
from collections import OrderedDict

# 添加项目根目录到Python路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.insert(0, project_root)

from models import CervicalCancerModel


class RealTimePredictor:
    """
    实时预测器，优化推理速度
    """

    def __init__(self, model_path, config, use_torchscript=False):
        """
        初始化实时预测器

        Args:
            model_path: 模型路径
            config: 配置字典
            use_torchscript: 是否使用TorchScript模型
        """
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.use_torchscript = use_torchscript

        # 加载模型
        self.model = self._load_model(model_path)
        self.model.to(self.device)
        self.model.eval()

        # 初始化预处理变换
        self.transform = self._create_transform()

        # 类别名称
        self.class_names = [
            'carcinoma_in_situ', 'light_dysplastic', 'moderate_dysplastic',
            'normal_columnar', 'normal_intermediate', 'normal_superficial', 'severe_dysplastic'
        ]

        # 预热模型
        self._warmup_model()

    def _load_model(self, model_path):
        """
        加载模型

        Args:
            model_path: 模型路径

        Returns:
            加载的模型
        """
        if self.use_torchscript:
            # 加载TorchScript模型
            model = torch.jit.load(model_path, map_location=self.device)
        else:
            # 加载标准PyTorch模型
            model = CervicalCancerModel(
                num_classes=self.config['model']['num_classes'],
                pretrained=False,
                dropout_rate=self.config['model']['dropout_rate'],
                model_type=self.config['model'].get('model_type', 'resnet50')
            )

            state_dict = torch.load(model_path, map_location=self.device)
            # 处理可能的键不匹配问题
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                if k.startswith('module.'):
                    name = k[7:]  # remove 'module.' prefix
                else:
                    name = k
                new_state_dict[name] = v

            model.load_state_dict(new_state_dict)

        return model

    def _create_transform(self):
        """
        创建图像预处理变换

        Returns:
            预处理变换
        """
        return transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((self.config['data']['img_size'], self.config['data']['img_size'])),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def _warmup_model(self):
        """
        预热模型以获得更准确的推理时间测量
        """
        print("正在预热模型...")
        dummy_input = torch.randn(1, 3, self.config['data']['img_size'],
                                  self.config['data']['img_size']).to(self.device)

        # 运行几次前向传播
        with torch.no_grad():
            for _ in range(10):
                _ = self.model(dummy_input)

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        print("模型预热完成")

    def preprocess_image(self, image):
        """
        预处理图像

        Args:
            image: 输入图像 (BGR格式)

        Returns:
            预处理后的张量
        """
        # 转换颜色空间
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # 应用变换
        image_tensor = self.transform(image_rgb).unsqueeze(0)

        return image_tensor

    def predict(self, image_tensor):
        """
        对单个图像进行预测

        Args:
            image_tensor: 预处理后的图像张量

        Returns:
            预测结果 (类别, 置信度, 所有概率)
        """
        image_tensor = image_tensor.to(self.device)

        with torch.no_grad():
            start_time = time.time()
            outputs = self.model(image_tensor)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            inference_time = time.time() - start_time

            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)

        return (self.class_names[predicted.item()],
                confidence.item(),
                probabilities[0].cpu().numpy(),
                inference_time)

    def batch_predict(self, image_tensors, batch_size=4):
        """
        批量预测以提高效率

        Args:
            image_tensors: 图像张量列表
            batch_size: 批次大小

        Returns:
            预测结果列表
        """
        results = []

        # 将列表转换为批次张量
        batch_tensor = torch.cat(image_tensors, dim=0).to(self.device)

        with torch.no_grad():
            start_time = time.time()
            outputs = self.model(batch_tensor)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            inference_time = time.time() - start_time

            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            confidences, predictions = torch.max(probabilities, 1)

            # 处理每个预测结果
            for i in range(len(predictions)):
                results.append((
                    self.class_names[predictions[i].item()],
                    confidences[i].item(),
                    probabilities[i].cpu().numpy()
                ))

        return results, inference_time / len(image_tensors)


def main():
    parser = argparse.ArgumentParser(description='实时宫颈癌分类预测')
    parser.add_argument('--model_path', type=str, required=True, help='模型路径')
    parser.add_argument('--config', type=str, default='configs/default.yaml', help='配置文件路径')
    parser.add_argument('--input_source', type=str, default='0',
                        help='输入源: 摄像头索引(0,1,...) 或 视频文件路径')
    parser.add_argument('--use_torchscript', action='store_true', help='使用TorchScript模型')
    parser.add_argument('--batch_size', type=int, default=1, help='批次大小')
    parser.add_argument('--display', action='store_true', help='显示实时结果')

    args = parser.parse_args()

    # 加载配置
    from utils import load_config
    config = load_config(args.config)

    # 创建预测器
    predictor = RealTimePredictor(args.model_path, config, args.use_torchscript)

    # 确定输入源
    try:
        input_source = int(args.input_source)
    except ValueError:
        input_source = args.input_source  # 视频文件路径

    # 打开视频流
    cap = cv2.VideoCapture(input_source)

    if not cap.isOpened():
        print(f"错误: 无法打开输入源 {input_source}")
        return 1

    print(f"开始实时预测，输入源: {input_source}")
    print("按 'q' 或 'ESC' 退出")

    frame_count = 0
    total_inference_time = 0
    fps_calculation_interval = 30  # 每30帧计算一次FPS

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1

            # 预处理图像
            image_tensor = predictor.preprocess_image(frame)

            # 进行预测
            prediction, confidence, probabilities, inference_time = predictor.predict(image_tensor)
            total_inference_time += inference_time

            # 计算FPS
            if frame_count % fps_calculation_interval == 0:
                avg_inference_time = total_inference_time / fps_calculation_interval
                fps = 1.0 / avg_inference_time if avg_inference_time > 0 else 0
                total_inference_time = 0

            # 显示结果
            if args.display:
                # 在图像上绘制结果
                cv2.putText(frame, f'Prediction: {prediction}',
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame, f'Confidence: {confidence:.4f}',
                            (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                if 'fps' in locals():
                    cv2.putText(frame, f'FPS: {fps:.2f}',
                                (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                cv2.imshow('Cervical Cancer Detection', frame)

                # 检查退出键
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == 27:  # 'q' 或 ESC
                    break

            # 打印结果到控制台
            print(f"Frame {frame_count}: {prediction} (confidence: {confidence:.4f})")

    except KeyboardInterrupt:
        print("\n用户中断程序")
    except Exception as e:
        print(f"发生错误: {e}")
    finally:
        # 清理资源
        cap.release()
        if args.display:
            cv2.destroyAllWindows()

    print(f"处理完成，共处理 {frame_count} 帧")

    return 0


if __name__ == '__main__':
    sys.exit(main())