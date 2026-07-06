"""
Grad-CAM 可解释性 Tool - 生成热力图并解读
"""

import torch
import torch.nn.functional as F
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from typing import Dict, Any, Optional
import logging
import os

logger = logging.getLogger(__name__)


class GradCAMTool:
    """
    LangGraph Tool: Grad-CAM 可解释性分析
    
    生成热力图显示模型"关注"的区域，帮助医生理解AI判断依据
    """
    name: str = "gradcam_explainer"
    description: str = """
    生成Grad-CAM热力图，可视化模型决策依据：
    - 高亮显示模型关注的细胞区域
    - 分析注意力分布是否合理（核区 vs 胞浆）
    - 检测模型是否关注无关区域（背景、杂质）
    
    输入: 原始图像 + 模型 + 目标类别
    输出: 热力图路径 + 可解释性分析
    """
    
    def __init__(self, model, target_layer: str = "layer4"):
        """
        Args:
            model: PyTorch 模型
            target_layer: 目标层名称（ResNet 默认 layer4）
        """
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # 注册钩子
        self._register_hooks()
    
    def _register_hooks(self):
        """注册前向/反向传播钩子"""
        def forward_hook(module, input, output):
            self.activations = output.detach()
        
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()
        
        # 获取目标层
        target = dict(self.model.named_modules()).get(self.target_layer)
        if target is None:
            # 自动查找最后一个卷积层
            target = self._find_last_conv_layer()
        
        target.register_forward_hook(forward_hook)
        target.register_full_backward_hook(backward_hook)
    
    def _find_last_conv_layer(self):
        """自动查找最后一个卷积层"""
        last_conv = None
        for name, module in self.model.named_modules():
            if isinstance(module, torch.nn.Conv2d):
                last_conv = module
        return last_conv
    
    def run(self, image: Image.Image, target_class: Optional[int] = None) -> Dict[str, Any]:
        """
        生成 Grad-CAM 热力图
        
        Args:
            image: 输入图像 (PIL RGB)
            target_class: 目标类别索引，None则使用预测类别
            
        Returns:
            {
                "heatmap_path": str,      # 热力图保存路径
                "overlay_path": str,      # 叠加图保存路径
                "attention_analysis": str, # 注意力分布分析
                "is_valid_attention": bool, # 注意力是否合理
                "confidence": float       # 对该分析的信心
            }
        """
        import torchvision.transforms as transforms
        
        # 预处理
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        input_tensor = transform(image).unsqueeze(0)
        
        # 前向传播
        self.model.eval()
        output = self.model(input_tensor)
        
        if target_class is None:
            target_class = output.argmax(dim=1).item()
        
        # 反向传播
        self.model.zero_grad()
        one_hot = torch.zeros_like(output)
        one_hot[0, target_class] = 1
        output.backward(gradient=one_hot, retain_graph=True)
        
        # 计算 Grad-CAM
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, squeeze=True)
        cam = F.relu(cam)
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        
        # 生成热力图
        cam_np = cam.cpu().numpy()
        heatmap = self._generate_heatmap(image, cam_np)
        overlay = self._generate_overlay(image, cam_np)
        
        # 分析注意力分布
        analysis, is_valid = self._analyze_attention(cam_np, image.size)
        
        # 保存
        os.makedirs("outputs/gradcam", exist_ok=True)
        heatmap_path = f"outputs/gradcam/heatmap_{target_class}.png"
        overlay_path = f"outputs/gradcam/overlay_{target_class}.png"
        heatmap.save(heatmap_path)
        overlay.save(overlay_path)
        
        return {
            "heatmap_path": heatmap_path,
            "overlay_path": overlay_path,
            "attention_analysis": analysis,
            "is_valid_attention": is_valid,
            "target_class": target_class,
            "confidence": 0.85 if is_valid else 0.6
        }
    
    def _generate_heatmap(self, original_image: Image.Image, cam: np.ndarray) -> Image.Image:
        """生成热力图"""
        # 上采样到原图尺寸
        cam_resized = cv2.resize(cam, original_image.size)
        
        # 应用颜色映射
        heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        
        return Image.fromarray(heatmap)
    
    def _generate_overlay(self, original_image: Image.Image, cam: np.ndarray, alpha: float = 0.5) -> Image.Image:
        """生成叠加图"""
        cam_resized = cv2.resize(cam, original_image.size)
        heatmap = self._generate_heatmap(original_image, cam_resized)
        
        # 叠加
        original_array = np.array(original_image)
        overlay = alpha * heatmap + (1 - alpha) * original_array
        overlay = np.clip(overlay, 0, 255).astype(np.uint8)
        
        return Image.fromarray(overlay)
    
    def _analyze_attention(self, cam: np.ndarray, image_size: tuple) -> tuple:
        """
        分析注意力分布是否合理
        
        Returns:
            (分析文本, 是否合理)
        """
        h, w = cam.shape
        
        # 将CAM分成中心区和边缘区
        center_mask = np.zeros_like(cam)
        margin = min(h, w) // 4
        center_mask[margin:-margin, margin:-margin] = 1
        
        center_attention = (cam * center_mask).sum() / cam.sum() if cam.sum() > 0 else 0
        
        # 分析
        if center_attention > 0.7:
            analysis = "模型注意力集中在图像中心区域，符合细胞核通常位于细胞中央的形态学特征，注意力分布合理"
            is_valid = True
        elif center_attention > 0.4:
            analysis = "模型注意力部分分散，部分关注细胞边缘或背景，建议结合形态学Tool复核"
            is_valid = True
        else:
            analysis = "警告：模型注意力严重偏离细胞主体，可能关注背景、杂质或伪影，该预测可信度低，必须人工复核"
            is_valid = False
        
        # 检查是否有多峰（注意力分散）
        from scipy.ndimage import label
        labeled, num_features = label(cam > 0.5 * cam.max())
        if num_features > 3:
            analysis += f"；检测到注意力分散为{num_features}个区域，提示模型可能捕捉到多个细胞或噪声"
        
        return analysis, is_valid
    
    async def arun(self, image: Image.Image, target_class: Optional[int] = None) -> Dict[str, Any]:
        """异步版本"""
        return self.run(image, target_class)


# 便捷函数：快速创建Tool实例
def create_gradcam_tool(model, target_layer: str = "layer4") -> GradCAMTool:
    """从已有模型创建 GradCAM Tool"""
    return GradCAMTool(model, target_layer)