"""
全片影像（WSI）处理器 —— **骨架，尚未接入任何调用链**。

后续扩展 WSI 时以此为起点。接线前必须补齐以下缺口：

  1. 读取：SVS 是金字塔 TIFF，`cv2.imread` 无法读取 → 需 openslide-python
     （Windows 上另需 openslide-bin 提供 DLL）；
  2. 内存：当前 `cv2.imread` 整片读入，千兆像素图会 OOM → 应按 level 分块 ReadRegion；
  3. 背景：全片大部分为空白，当前逐块推理浪费 90%+ 算力，且稀释"异常占比"
     → 需先做组织掩膜（如 Otsu + 面积阈值）；
  4. 边缘块：`split_image` 会把残片 resize 成 224×224，长宽比失真
     → 应 padding 或直接丢弃过小块；
  5. 聚合：`merge_predictions` 对**类别索引**取平均（如 (0+4)/2=2）无语义
     → 应改为概率投票 / 按 patch 计数；
  6. 级联：当前只调分类模型，未复用 `services.vision_service` 的二分类+多分类+形态学链路，
     因此产不出 cells / abnormal_cells，无法喂给 `DiagnosisState`；
  7. API：`.svs` 目前在 `agent_api.py` 的上传白名单之外（会被 400 拒绝）；
     接线时需同步放开，并改为异步任务（前端 axios 超时 120s，WSI 必然超时）。

⚠️ 以上完成前，**请勿**把 `.svs` 加进 `agent_api.py` 的 `allowed_exits`——
   否则用户会拿到难以理解的 500（内部错误编号），而不是明确的"暂不支持"。
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
