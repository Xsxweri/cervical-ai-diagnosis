"""
YOLOv8 Model for Cervical Cell Detection and Segmentation
"""

import torch
import numpy as np
from ultralytics import YOLO
import cv2
from typing import List, Tuple, Dict, Optional, Union
import os


class YOLOv8CellDetector:
    """
    YOLOv8模型用于宫颈细胞检测和分割
    """

    def __init__(self, model_path: Optional[str] = None, task: str = 'detect'):
        """
        初始化YOLOv8细胞检测器

        Args:
            model_path: 预训练模型路径或模型名称
            task: 任务类型 ('detect', 'segment', 'classify')
        """
        if model_path is None:
            # 使用默认的YOLOv8模型
            model_path = 'yolov8n.pt'  # 可以替换为 yolov8s.pt, yolov8m.pt 等

        self.model = YOLO(model_path, task=task)
        self.task = task

    def detect_cells(self, image: Union[str, np.ndarray],
                     conf_threshold: float = 0.25,
                     iou_threshold: float = 0.45) -> Dict:
        """
        检测图像中的细胞

        Args:
            image: 图像路径或numpy数组
            conf_threshold: 置信度阈值
            iou_threshold: IOU阈值

        Returns:
            包含检测结果的字典
        """
        results = self.model.predict(
            source=image,
            conf=conf_threshold,
            iou=iou_threshold,
            verbose=False
        )

        return self._process_results(results)

    def segment_cells(self, image: Union[str, np.ndarray],
                      conf_threshold: float = 0.25,
                      iou_threshold: float = 0.45) -> Dict:
        """
        分割图像中的细胞

        Args:
            image: 图像路径或numpy数组
            conf_threshold: 置信度阈值
            iou_threshold: IOU阈值

        Returns:
            包含分割结果的字典
        """
        results = self.model.predict(
            source=image,
            conf=conf_threshold,
            iou=iou_threshold,
            verbose=False
        )

        return self._process_results(results)

    def _process_results(self, results) -> Dict:
        """
        处理YOLOv8的结果
        Args:
            results: YOLOv8模型输出结果
        Returns:
            处理后的结果字典
        """
        processed_results = {
            'boxes': [],
            'masks': [],
            'keypoints': [],
            'class_ids': [],
            'confidences': [],
            'class_names': []
        }

        for result in results:
            # 处理边界框
            if hasattr(result, 'boxes') and result.boxes is not None:
                boxes = result.boxes
                processed_results['boxes'] = boxes.xyxy.cpu().numpy().tolist() if boxes.xyxy is not None else []
                processed_results['class_ids'] = boxes.cls.cpu().numpy().tolist() if boxes.cls is not None else []
                processed_results['confidences'] = boxes.conf.cpu().numpy().tolist() if boxes.conf is not None else []

                # 获取类别名称
                if hasattr(result.names, '__getitem__'):
                    processed_results['class_names'] = [result.names[int(cls)] for cls in
                                                        processed_results['class_ids']]

            # 处理分割掩码
            if hasattr(result, 'masks') and result.masks is not None:
                masks = result.masks
                processed_results['masks'] = masks.data.cpu().numpy().tolist() if masks.data is not None else []

            # 处理关键点
            if hasattr(result, 'keypoints') and result.keypoints is not None:
                keypoints = result.keypoints
                processed_results[
                    'keypoints'] = keypoints.data.cpu().numpy().tolist() if keypoints.data is not None else []

        return processed_results

    def train_model(self, data_yaml_path: str, epochs: int = 100,
                    imgsz: int = 640, batch_size: int = 16,
                    device: Union[str, int] = 0) -> None:
        """
        训练YOLOv8模型

        Args:
            data_yaml_path: 数据集配置文件路径
            epochs: 训练轮数
            imgsz: 图像尺寸
            batch_size: 批次大小
            device: 训练设备 (0 for GPU, 'cpu' for CPU)
        """
        self.model.train(
            data=data_yaml_path,
            epochs=epochs,
            imgsz=imgsz,
            batch=batch_size,
            device=device
        )

    def save_model(self, save_path: str) -> None:
        """
        保存模型

        Args:
            save_path: 保存路径
        """
        self.model.save(save_path)

class CervicalCellDatasetProcessor:
    """
    宫颈细胞数据集处理器，用于准备YOLO格式的数据集
    """

    def __init__(self, dataset_root: str):
        """
        初始化数据集处理器

        Args:
            dataset_root: 数据集根目录
        """
        self.dataset_root = dataset_root
        self.annotations = []

    def create_yolo_dataset_structure(self) -> None:
        """
        创建YOLO格式的数据集目录结构
        """
        dirs = [
            os.path.join(self.dataset_root, 'images', 'train'),
            os.path.join(self.dataset_root, 'images', 'val'),
            os.path.join(self.dataset_root, 'images', 'test'),
            os.path.join(self.dataset_root, 'labels', 'train'),
            os.path.join(self.dataset_root, 'labels', 'val'),
            os.path.join(self.dataset_root, 'labels', 'test')
        ]

        for dir_path in dirs:
            os.makedirs(dir_path, exist_ok=True)

    def convert_annotations_to_yolo_format(self,
                                           annotations: List[Dict],
                                           output_dir: str,
                                           image_size: Tuple[int, int]) -> None:
        """
        将标注转换为YOLO格式

        Args:
            annotations: 标注列表
            output_dir: 输出目录
            image_size: 图像尺寸 (width, height)
        """
        for annotation in annotations:
            image_id = annotation['image_id']
            boxes = annotation['boxes']
            class_ids = annotation['class_ids']

            # 创建YOLO格式的标注文件
            label_file = os.path.join(output_dir, f"{image_id}.txt")

            with open(label_file, 'w') as f:
                for box, class_id in zip(boxes, class_ids):
                    # 转换为YOLO格式 (center_x, center_y, width, height) 归一化到[0,1]
                    x1, y1, x2, y2 = box
                    center_x = ((x1 + x2) / 2) / image_size[0]
                    center_y = ((y1 + y2) / 2) / image_size[1]
                    width = (x2 - x1) / image_size[0]
                    height = (y2 - y1) / image_size[1]

                    f.write(f"{class_id} {center_x:.6f} {center_y:.6f} {width:.6f} {height:.6f}\n")

    def visualize_detections(image: np.ndarray,
                             results: Dict,
                             class_colors: Optional[Dict] = None) -> np.ndarray:
        """
        可视化检测结果

        Args:
            image: 输入图像
            results: 检测结果
            class_colors: 类别颜色映射

        Returns:
            带有标注的图像
        """
        if class_colors is None:
            class_colors = {
                0: (255, 0, 0),  # 蓝色
                1: (0, 255, 0),  # 绿色
                2: (0, 0, 255),  # 红色
            }

        vis_image = image.copy()

        # 绘制边界框
        if 'boxes' in results and len(results['boxes']) > 0:
            boxes = results['boxes']
            class_ids = results['class_ids']
            confidences = results['confidences']

            for i, (box, class_id, conf) in enumerate(zip(boxes, class_ids, confidences)):
                x1, y1, x2, y2 = map(int, box)
                color = class_colors.get(int(class_id), (255, 255, 0))  # 默认黄色
                # 绘制边界框
                cv2.rectangle(vis_image, (x1, y1), (x2, y2), color, 2)

                # 添加标签
                label = f"Class {int(class_id)}: {conf:.2f}"
                cv2.putText(vis_image, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        # 绘制分割掩码
        if 'masks' in results and len(results['masks']) > 0:
            masks = results['masks']
            class_ids = results['class_ids']

            for i, (mask, class_id) in enumerate(zip(masks, class_ids)):
                color = class_colors.get(int(class_id), (255, 255, 0))

                # 将掩码应用到图像上
                colored_mask = np.zeros_like(vis_image)
                colored_mask[mask > 0.5] = color

                # 混合原图和掩码
                vis_image = cv2.addWeighted(vis_image, 0.7, colored_mask, 0.3, 0)

        return vis_image