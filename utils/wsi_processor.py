"""
全片影像（WSI）处理器，用于大型宫颈细胞影像
支持基于补丁的千兆像素图像推断
"""
import numpy as np
import cv2
from typing import List, Tuple, Dict
import torch
from PIL import Image


class WSIProcessor:
    """全切片图像处理器，支持超大图像分块推理"""

    def __init__(self, patch_size: int = 1024, overlap: int = 128):
        """
        Args:
            patch_size: 分块大小（像素）
            overlap: 重叠区域大小（避免边界效应）
        """
        self.patch_size = patch_size
        self.overlap = overlap

    def split_image(self, image: np.ndarray) -> List[Tuple[int, int, np.ndarray]]:
        """
        将大图像分割为重叠的patches

        Returns:
            List of (x, y, patch) 元组
        """
        h, w = image.shape[:2]
        patches = []

        step = self.patch_size - self.overlap

        for y in range(0, h, step):
            for x in range(0, w, step):
                y_end = min(y + self.patch_size, h)
                x_end = min(x + self.patch_size, w)

                patch = image[y:y_end, x:x_end]

                if patch.shape[0] > 0 and patch.shape[1] > 0:
                    patches.append((x, y, patch))

        return patches

    def merge_predictions(
            self,
            image_shape: Tuple[int, int],
            predictions: List[Tuple[int, int, np.ndarray]]
    ) -> np.ndarray:
        """
        合并patches的预测结果为完整图像

        Args:
            image_shape: 原始图像尺寸 (H, W)
            predictions: List of (x, y, pred_map) 元组

        Returns:
            合并后的预测图
        """
        h, w = image_shape[:2]
        merged = np.zeros((h, w), dtype=np.float32)
        weight_map = np.zeros((h, w), dtype=np.float32)

        for x, y, pred in predictions:
            pred_h, pred_w = pred.shape[:2]

            x_end = min(x + pred_w, w)
            y_end = min(y + pred_h, h)

            actual_w = x_end - x
            actual_h = y_end - y

            pred_cropped = pred[:actual_h, :actual_w]

            merged[y:y_end, x:x_end] += pred_cropped
            weight_map[y:y_end, x:x_end] += 1

        merged /= np.maximum(weight_map, 1)

        return merged

    def process_wsi(
            self,
            image_path: str,
            model,
            device: torch.device,
            batch_size: int = 8
    ) -> Dict:
        """
        处理全切片图像

        Args:
            image_path: 图像路径
            model: 分类模型
            device: 计算设备
            batch_size: 批次大小

        Returns:
            包含预测结果的字典
        """
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"无法读取图像: {image_path}")

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        patches = self.split_image(image_rgb)

        results = []

        for i in range(0, len(patches), batch_size):
            batch_patches = patches[i:i + batch_size]
            batch_images = []

            for x, y, patch in batch_patches:
                resized = cv2.resize(patch, (224, 224))
                normalized = resized.astype(np.float32) / 255.0

                mean = [0.485, 0.456, 0.406]
                std = [0.229, 0.224, 0.225]
                for c in range(3):
                    normalized[:, :, c] = (normalized[:, :, c] - mean[c]) / std[c]

                batch_images.append(normalized)

            batch_tensor = torch.FloatTensor(batch_images).permute(0, 3, 1, 2).to(device)

            with torch.no_grad():
                outputs = model(batch_tensor)
                probs = torch.softmax(outputs, dim=1)
                preds = torch.argmax(probs, dim=1)

            for j, (x, y, patch) in enumerate(batch_patches):
                pred_map = np.full(
                    (patch.shape[0], patch.shape[1]),
                    preds[j].item(),
                    dtype=np.float32
                )
                results.append((x, y, pred_map))

        merged_pred = self.merge_predictions(image.shape[:2], results)

        return {
            'merged_prediction': merged_pred,
            'patches': results,
            'image_shape': image.shape
        }
