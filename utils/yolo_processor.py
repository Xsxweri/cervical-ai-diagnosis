"""
YOLO宫颈细胞检测格式数据处理器
处理注释解析、转换和验证
"""

import os
from typing import List, Dict, Tuple
import numpy as np
import cv2
from utils.label_mapping import SIPAKMED_CLASSES

class YOLOProcessor:
    """YOLO格式数据处理工具"""

    def __init__(self, class_names: List[str] = None):
        """
        Args:
            class_names: 类别名称列表
        """
        self.class_names = class_names or list(SIPAKMED_CLASSES)
        self.class_to_idx = {name: i for i, name in enumerate(self.class_names)}
        self.idx_to_class = {i: name for name, i in self.class_to_idx.items()}

    def parse_yolo_annotation(
            self,
            label_path: str,
            image_width: int,
            image_height: int
    ) -> List[Dict]:
        """
        解析YOLO格式标注文件

        YOLO格式: class_id center_x center_y width height (归一化到[0,1])

        Returns:
            标注列表，每个元素包含 class_id, bbox (x1,y1,x2,y2)
        """
        annotations = []

        if not os.path.exists(label_path):
            return annotations

        with open(label_path, 'r') as f:
            lines = f.readlines()

        for line in lines:
            line = line.strip()
            if not line:
                continue

            parts = line.split()
            if len(parts) != 5:
                continue

            class_id = int(parts[0])
            center_x = float(parts[1])
            center_y = float(parts[2])
            width = float(parts[3])
            height = float(parts[4])

            x1 = int((center_x - width / 2) * image_width)
            y1 = int((center_y - height / 2) * image_height)
            x2 = int((center_x + width / 2) * image_width)
            y2 = int((center_y + height / 2) * image_height)

            annotations.append({
                'class_id': class_id,
                'class_name': self.idx_to_class.get(class_id, 'unknown'),
                'bbox': [x1, y1, x2, y2]
            })

        return annotations

    def convert_to_yolo_format(
            self,
            bbox: List[int],
            image_width: int,
            image_height: int
    ) -> Tuple[float, float, float, float]:
        """
        将边界框转换为YOLO格式

        Args:
            bbox: [x1, y1, x2, y2] 像素坐标
            image_width: 图像宽度
            image_height: 图像高度

        Returns:
            (center_x, center_y, width, height) 归一化坐标
        """
        x1, y1, x2, y2 = bbox

        center_x = ((x1 + x2) / 2) / image_width
        center_y = ((y1 + y2) / 2) / image_height
        width = (x2 - x1) / image_width
        height = (y2 - y1) / image_height

        return center_x, center_y, width, height

    def create_annotation_file(
            self,
            output_path: str,
            annotations: List[Dict],
            image_width: int,
            image_height: int
    ):
        """
        创建YOLO格式标注文件

        Args:
            output_path: 输出文件路径 (.txt)
            annotations: 标注列表
            image_width: 图像宽度
            image_height: 图像高度
        """
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        with open(output_path, 'w') as f:
            for ann in annotations:
                class_id = ann['class_id']
                bbox = ann['bbox']

                center_x, center_y, width, height = self.convert_to_yolo_format(
                    bbox, image_width, image_height
                )

                f.write(f"{class_id} {center_x:.6f} {center_y:.6f} {width:.6f} {height:.6f}\n")

    def validate_dataset(self, dataset_root: str) -> Dict:
        """
        验证YOLO数据集完整性

        Args:
            dataset_root: 数据集根目录
                期望结构:
                dataset_root/
                ├── images/train/, val/, test/
                └── labels/train/, val/, test/

        Returns:
            验证报告字典
        """
        report = {
            'total_images': 0,
            'total_annotations': 0,
            'missing_annotations': [],
            'empty_annotations': [],
            'class_distribution': {}
        }

        for split in ['train', 'val', 'test']:
            images_dir = os.path.join(dataset_root, 'images', split)
            labels_dir = os.path.join(dataset_root, 'labels', split)

            if not os.path.exists(images_dir):
                continue

            image_files = [f for f in os.listdir(images_dir)
                           if f.lower().endswith(('.jpg', '.png', '.bmp'))]

            for img_file in image_files:
                report['total_images'] += 1

                base_name = os.path.splitext(img_file)[0]
                label_file = f"{base_name}.txt"
                label_path = os.path.join(labels_dir, label_file)

                if not os.path.exists(label_path):
                    report['missing_annotations'].append(img_file)
                    continue

                annotations = self.parse_yolo_annotation(
                    label_path,
                    image_width=1,
                    image_height=1
                )

                if len(annotations) == 0:
                    report['empty_annotations'].append(img_file)

                report['total_annotations'] += len(annotations)

                for ann in annotations:
                    class_name = ann['class_name']
                    report['class_distribution'][class_name] = \
                        report['class_distribution'].get(class_name, 0) + 1

        return report

    def visualize_annotations(
            self,
            image_path: str,
            label_path: str,
            output_path: str = None
    ) -> np.ndarray:
        """
        可视化YOLO标注

        Args:
            image_path: 图像路径
            label_path: 标注文件路径
            output_path: 保存路径（可选）

        Returns:
            带标注的图像
        """
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"无法读取图像: {image_path}")

        h, w = image.shape[:2]
        annotations = self.parse_yolo_annotation(label_path, w, h)

        colors = [
            (0, 255, 0),  # 绿色 - normal
            (0, 0, 255),  # 红色 - abnormal
            (255, 255, 0),  # 黄色 - unknown
        ]

        for ann in annotations:
            x1, y1, x2, y2 = ann['bbox']
            class_id = ann['class_id']
            color = colors[class_id] if class_id < len(colors) else (255, 0, 255)

            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

            label = f"{ann['class_name']} ({class_id})"
            cv2.putText(
                image, label, (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2
            )

        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            cv2.imwrite(output_path, image)

        return image