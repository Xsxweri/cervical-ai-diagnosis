"""
将SiPaKMeD数据集转换为YOLO格式的工具
"""

import os
import cv2
import numpy as np
from typing import List, Tuple
import random
import shutil
from pathlib import Path


class SiPaKMeDConverter:
    """
    SiPaKMeD数据集转换器，将数据集转换为YOLO格式
    """

    def __init__(self, dataset_root: str, output_root: str):
        """
        初始化转换器

        Args:
            dataset_root: SiPaKMeD数据集根目录
            output_root: 输出YOLO格式数据集的根目录
        """
        self.dataset_root = dataset_root
        self.output_root = output_root

        # SiPaKMeD数据集类别
        self.classes = {
            'im_Dyskeratotic': 0,
            'im_Koilocytotic': 1,
            'im_Metaplastic': 2,
            'im_Parabasal': 3,
            'im_Superficial-Intermediate': 4
        }

        # 创建输出目录结构
        self._create_output_structure()

    def _create_output_structure(self):
        """
        创建输出目录结构
        """
        dirs = [
            os.path.join(self.output_root, 'images', 'train'),
            os.path.join(self.output_root, 'images', 'val'),
            os.path.join(self.output_root, 'images', 'test'),
            os.path.join(self.output_root, 'labels', 'train'),
            os.path.join(self.output_root, 'labels', 'val'),
            os.path.join(self.output_root, 'labels', 'test')
        ]

        for dir_path in dirs:
            os.makedirs(dir_path, exist_ok=True)

    def _parse_dat_file(self, dat_path: str) -> List[List[float]]:
        """
        解析.dat文件，获取轮廓点坐标

        Args:
            dat_path: .dat文件路径

        Returns:
            轮廓点坐标列表
        """
        points = []
        try:
            with open(dat_path, 'r') as f:
                for line in f:
                    if ',' in line:
                        x, y = map(float, line.strip().split(','))
                        points.append([x, y])
        except Exception as e:
            print(f"Error parsing {dat_path}: {e}")

        return points

    def _convert_to_yolo_bbox(self, points: List[List[float]],
                              img_width: int, img_height: int) -> Tuple[float, float, float, float]:
        """
        将轮廓点转换为YOLO格式的边界框

        Args:
            points: 轮廓点坐标列表
            img_width: 图像宽度
            img_height: 图像高度

        Returns:
            YOLO格式的边界框 (center_x, center_y, width, height)
        """
        if not points:
            return 0.0, 0.0, 0.0, 0.0

        points = np.array(points)
        x_min, y_min = np.min(points, axis=0)
        x_max, y_max = np.max(points, axis=0)

        # 转换为YOLO格式（归一化到0-1）
        center_x = ((x_min + x_max) / 2) / img_width
        center_y = ((y_min + y_max) / 2) / img_height
        width = (x_max - x_min) / img_width
        height = (y_max - y_min) / img_height

        return center_x, center_y, width, height

    def convert_dataset(self, train_ratio: float = 0.7, val_ratio: float = 0.15):
        """
        转换整个数据集

        Args:
            train_ratio: 训练集比例
            val_ratio: 验证集比例
        """
        print("开始转换SiPaKMeD数据集为YOLO格式...")

        # 遍历每个类别目录
        for class_dir, class_id in self.classes.items():
            class_path = os.path.join(self.dataset_root, class_dir)
            if not os.path.exists(class_path):
                print(f"警告: 类别目录不存在 {class_path}")
                continue

            # 检查是否存在双层嵌套结构
            nested_path = os.path.join(class_path, class_dir)
            if os.path.exists(nested_path) and os.path.isdir(nested_path):
                actual_class_path = nested_path
                print(f"检测到双层嵌套结构，使用: {actual_class_path}")
            else:
                actual_class_path = class_path

            # 优先检查CROPPED子目录（实际数据和.dat文件所在位置）
            cropped_path = os.path.join(actual_class_path, 'CROPPED')
            if os.path.exists(cropped_path) and os.path.isdir(cropped_path):
                data_path = cropped_path
                print(f"  检测到CROPPED子目录，使用: {data_path}")
            else:
                data_path = actual_class_path

            print(f"处理类别: {class_dir} (ID: {class_id})")

            # 获取所有.bmp图像文件
            bmp_files = [f for f in os.listdir(data_path) if f.endswith('.bmp')]

            # 随机打乱文件列表
            random.shuffle(bmp_files)

            # 计算分割点
            train_end = int(len(bmp_files) * train_ratio)
            val_end = int(len(bmp_files) * (train_ratio + val_ratio))

            # 分配到不同集合
            train_files = bmp_files[:train_end]
            val_files = bmp_files[train_end:val_end]
            test_files = bmp_files[val_end:]

            print(f"  训练集: {len(train_files)} 张图像")
            print(f"  验证集: {len(val_files)} 张图像")
            print(f"  测试集: {len(test_files)} 张图像")

            # 处理每个集合
            self._process_files(data_path, train_files, 'train', class_id)
            self._process_files(data_path, val_files, 'val', class_id)
            self._process_files(data_path, test_files, 'test', class_id)

        print("数据集转换完成!")

    def _process_files(self, class_path: str, files: List[str],
                       split: str, class_id: int):
        """
        处理指定集合的文件

        Args:
            class_path: 类别目录路径
            files: 文件列表
            split: 集合名称 (train/val/test)
            class_id: 类别ID
        """
        for filename in files:
            # 构建图像路径
            img_path = os.path.join(class_path, filename)

            # 检查图像文件是否存在
            if not os.path.exists(img_path):
                continue

            # 读取图像获取尺寸
            img = cv2.imread(img_path)
            if img is None:
                continue

            img_height, img_width = img.shape[:2]

            # 复制图像到输出目录
            output_img_path = os.path.join(self.output_root, 'images', split, filename)
            shutil.copy2(img_path, output_img_path)

            # 处理对应的.dat文件
            base_name = os.path.splitext(filename)[0]
            dat_files = [f for f in os.listdir(class_path)
                         if f.startswith(base_name + '_') and f.endswith('.dat')]

            # 创建YOLO标签文件
            label_filename = os.path.splitext(filename)[0] + '.txt'
            label_path = os.path.join(self.output_root, 'labels', split, label_filename)

            # 写入标签文件
            with open(label_path, 'w') as f:
                for dat_file in dat_files:
                    dat_path = os.path.join(class_path, dat_file)
                    points = self._parse_dat_file(dat_path)

                    if points:
                        # 转换为YOLO边界框格式
                        center_x, center_y, width, height = self._convert_to_yolo_bbox(
                            points, img_width, img_height)

                        # 写入YOLO格式标签: class_id center_x center_y width height
                        f.write(f"{class_id} {center_x:.6f} {center_y:.6f} {width:.6f} {height:.6f}\n")

def create_yolo_config(dataset_root: str, output_path: str):
    """
    创建YOLO配置文件

    Args:
        dataset_root: 数据集根目录
        output_path: 输出配置文件路径
    """
    config_content = f"""# SiPaKMeD宫颈细胞数据集配置文件（用于YOLOv8训练）

# 数据集根目录路径
path: {dataset_root}

# 训练、验证、测试集路径
train: images/train
val: images/val
test: images/test

# 类别数量
nc: 5

# 类别名称（对应SiPaKMeD数据集的5个类别）
names: 
  0: Dyskeratotic
  1: Koilocytotic
  2: Metaplastic
  3: Parabasal
  4: SuperficialIntermediate
"""

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(config_content)

    print(f"YOLO配置文件已创建: {output_path}")


# 使用示例
if __name__ == "__main__":
    # 设置路径
    dataset_root = "data/sipakmed"
    output_root = "data/sipakmed_yolo"
    config_path = "configs/sipakmed_yolo.yaml"

    # 创建转换器并转换数据集
    converter = SiPaKMeDConverter(dataset_root, output_root)
    converter.convert_dataset(train_ratio=0.7, val_ratio=0.15)

    # 创建YOLO配置文件
    create_yolo_config(output_root, config_path)