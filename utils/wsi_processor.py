"""
全片影像（WSI）处理器 —— **骨架，尚未接入任何调用链**。

后续扩展 WSI 时以此为起点。接线前必须补齐以下缺口：

  决策理由（不是工作量问题，是前提不成立）：
  1. 无宫颈细胞学 WSI 标注数据集 → 无法给出任何敏感性/特异性；
  2. 检测/分类模型训练于单细胞裁剪图（SIPaKMeD CROPPED / Herlev），
     与全片尺度存在域偏移；
  3. 第 3 条缺口的「组织占少数、用 Otsu 找组织」是**组织学**假设；
     细胞学涂片的细胞单层铺满整个圆盘，圆盘内按组织密度筛块会
     静默丢掉散在异常细胞 —— 这是医学上错误的，规格需重写；
  4. 单片吞吐：4.8e9 px / 512² ≈ 1.1 万块，GPU 40–90 分钟/片，
     与病理科日均数百片不匹配。

**重新启动的条件（全部满足才动）**：
  * 获得 ≥100 例带病变级标注的 LBC / Pap WSI；
  * 在 WSI 分块上重训检测器，而非复用 SIPaKMeD 的 YOLO；
  * 前置完成「大 FOV 分块推理」并用 RIVA 验证（见
    utils/tiling.py 的规划）——那是单细胞裁剪图到真实视野的
    必要跨越，也是 WSI 的低成本前置。
"""
import numpy as np
import cv2
from typing import List, Tuple, Dict
import torch


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
