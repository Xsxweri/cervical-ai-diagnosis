import torch
import torch.nn.functional as F
import cv2
import numpy as np
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from torchvision.transforms import functional as TF
import matplotlib.pyplot as plt


class GradCAMVisualizer:
    def __init__(self, model, target_layer=None):
        if target_layer is None:
            # 自动检测最后一个卷积层
            if hasattr(model, 'base_model'):
                # 根据模型类型自动选择目标层
                if isinstance(model.base_model, models.ResNet):
                    target_layer = model.base_model.layer4[-1].conv3
                elif 'efficientnet' in str(type(model.base_model)).lower():
                    target_layer = model.base_model.features[-1][0].block[2][0]

    def visualize(self, image_tensor, class_idx=None, image_path=None):
        """
        可视化图像的Grad-CAM

        Args:
            image_tensor: 输入图像张量
            class_idx: 目标类别索引，如果为None则使用模型预测的最高概率类别
            image_path: 原始图像路径，用于叠加热力图

        Returns:
            可视化图像
        """
        # 确保模型在评估模式
        self.model.eval()

        # 添加批次维度
        if image_tensor.dim() == 3:
            input_tensor = image_tensor.unsqueeze(0)
        else:
            input_tensor = image_tensor

        # 如果指定了类别，则使用该类别，否则使用模型预测的类别
        if class_idx is not None:
            targets = [lambda x: F.one_hot(torch.tensor(class_idx), num_classes=x.size(1))]
        else:
            targets = None

        # 生成热力图
        grayscale_cam = self.cam(input_tensor=input_tensor, targets=targets)
        grayscale_cam = grayscale_cam[0, :]

        # 如果提供了原始图像路径，则叠加热力图
        if image_path:
            # 读取原始图像
            rgb_img = cv2.imread(image_path)
            rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)
            rgb_img = cv2.resize(rgb_img, (224, 224))  # 调整到模型输入尺寸
            rgb_img = np.float32(rgb_img) / 255

            # 叠加热力图
            visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
            return visualization, grayscale_cam

        return grayscale_cam

    def save_visualization(self, image_tensor, output_path, class_idx=None, image_path=None):
        """
        保存Grad-CAM可视化结果

        Args:
            image_tensor: 输入图像张量
            output_path: 输出图像路径
            class_idx: 目标类别索引
            image_path: 原始图像路径
        """
        visualization, _ = self.visualize(image_tensor, class_idx, image_path)

        # 保存图像
        plt.figure(figsize=(8, 8))
        plt.imshow(visualization)
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()