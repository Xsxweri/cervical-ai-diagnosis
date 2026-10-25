import os
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import yaml
import random
import logging
from utils.label_mapping import (
    BINARY_CLASSES,
    BINARY_CLASS_MAPPING,
    SIPAKMED_DIR_MAPPING,
    DEFAULT_LABEL_SPACE,
    get_multiclass_classes,
)

def load_config(config_path="configs/default.yaml"):
    """加载配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config

class CombinedCervicalCancerDataset(Dataset):
    def __init__(self, datasets_config, transform=None, config=None, classification_mode='binary', label_space=None):
        """
        初始化数据集

        Args:
            datasets_config: 数据集配置
            transform: 数据变换
            config: 配置文件
            classification_mode: 分类模式 ('binary' 或 'multiclass')
            label_space: 多分类标签空间名（'sipakmed' | 'herlev'）；None 时用默认。
        """
        self.transform = transform
        self.config = config or load_config()
        self.classification_mode = classification_mode

        # 定义原始类别名称
        self.original_class_names = [
            'carcinoma_in_situ', 'light_dysplastic', 'moderate_dysplastic',
            'normal_columnar', 'normal_intermediate', 'normal_superficial', 'severe_dysplastic'
        ]

        if self.classification_mode == 'binary':
            # 定义统一的类别映射（二分类）
            self.label_space = None
            self.class_mapping = dict(BINARY_CLASS_MAPPING)

            #索引 0 = abnormal, 1 = normal（权重文件语义）
            self.unified_classes = list(BINARY_CLASSES)
            assert set(self.class_mapping.values()) == set(self.unified_classes), (
                "二分类映射与 canonical 定义不一致："
                f"{sorted(set(self.class_mapping.values()))} vs {self.unified_classes}"
            )
            self.class_to_idx = {cls: i for i, cls in enumerate(self.unified_classes)}
            self.target_names = self.unified_classes

        else:
            # 多分类模式：标签空间由 label_space 显式指定（默认 sipakmed 5 类）。
            # 禁止再用 sorted()/拼接，避免标签空间随数据集组合漂移。
            self.label_space = label_space or DEFAULT_LABEL_SPACE
            classes = get_multiclass_classes(self.label_space)
            self.class_to_idx = {cls: i for i, cls in enumerate(classes)}
            self.target_names = classes

        self.images = []
        self.skipped_images = {}
        self._load_images(datasets_config)
    def _record_skipped(self, dataset_type: str, class_name: str, count: int):
        """记录因不在当前标签空间而被跳过的图像数，用于显式告警/报错"""
        if count <= 0:
            return
        key = f"{dataset_type}/{class_name}"
        self.skipped_images[key] = self.skipped_images.get(key, 0) + count

    def _load_images(self, datasets_config):
        for dataset_info in datasets_config:
            dataset_type = dataset_info['type']
            root_dir = dataset_info['path']

            print(f"Processing {dataset_type} dataset at {root_dir}")

            if not os.path.exists(root_dir):
                print(f"  ERROR: Directory does not exist: {root_dir}")
                continue

            if not os.path.isdir(root_dir):
                print(f"  ERROR: {root_dir} is not a directory")
                continue

            print(f"  Directory exists: {os.path.exists(root_dir)}")
            print(f"  Is directory: {os.path.isdir(root_dir)}")

            if dataset_type == 'sipakmed':
                self._load_sipakmed_images(root_dir)
            elif dataset_type == 'herlev':
                self._load_herlev_images(root_dir)

        if self.skipped_images:
            total_skipped = sum(self.skipped_images.values())
            detail = ", ".join(f"{k}({v})" for k, v in self.skipped_images.items())
            msg = (f"标签空间 '{self.label_space}' 不包含以下类别的图像，共跳过 "
                   f"{total_skipped} 张: {detail}")
            if not self.images:
                raise ValueError(msg + "；没有任何图像可训练，请检查 label_space 与数据集配置。")
            logging.warning(msg)

    def _load_sipakmed_images(self, root_dir):
        # 加载SIPaKMeD数据集图像
        print(f"Loading SIPaKMeD images from: {root_dir}")

        # SIPaKMeD数据集的类别映射
        sipakmed_class_mapping = SIPAKMED_DIR_MAPPING

        # 遍历所有可能的类别文件夹
        for folder_name in os.listdir(root_dir):
            folder_path = os.path.join(root_dir, folder_name)
            if os.path.isdir(folder_path) and folder_name in sipakmed_class_mapping:
                original_class_name = sipakmed_class_mapping[folder_name]
                image_count = 0

                # 检查是否存在双层嵌套结构
                nested_path = os.path.join(folder_path, folder_name)
                if os.path.exists(nested_path) and os.path.isdir(nested_path):
                    target_path = nested_path
                else:
                    target_path = folder_path

                # 优先检查CROPPED子目录（实际数据所在位置）
                cropped_path = os.path.join(target_path, 'CROPPED')
                if os.path.exists(cropped_path) and os.path.isdir(cropped_path):
                    actual_path = cropped_path
                    print(f"  检测到CROPPED子目录，使用: {actual_path}")
                    print(f"    CROPPED目录文件总数: {len(os.listdir(actual_path))}")
                    bmp_count = len(
                        [f for f in os.listdir(actual_path) if f.lower().endswith(('.bmp', '.jpg', '.png', '.jpeg'))])
                    print(f"    其中图像文件数: {bmp_count}")
                else:
                    actual_path = target_path
                    print(f"  未找到CROPPED子目录，使用: {actual_path}")

                # 遍历文件夹中的所有文件
                file_count = 0
                skipped_count = 0
                for filename in os.listdir(actual_path):
                    file_count += 1
                    if filename.lower().endswith(('.bmp', '.jpg', '.png', '.jpeg')):
                        image_path = os.path.join(actual_path, filename)
                        if self.classification_mode == 'binary':
                            if original_class_name in self.class_mapping:
                                mapped_class = self.class_mapping[original_class_name]
                                self.images.append((image_path, self.class_to_idx[mapped_class]))
                                image_count += 1
                                if image_count <= 5:
                                    print(f"    Loaded: {image_path}")
                        else:
                            if original_class_name in self.class_to_idx:
                                self.images.append((image_path, self.class_to_idx[original_class_name]))
                                image_count += 1
                                if image_count <= 5:
                                    print(f"    Loaded: {image_path}")
                            else:
                                skipped_count += 1

                self._record_skipped('sipakmed', folder_name, skipped_count)
                print(f"  扫描文件数: {file_count}, 找到图像数: {image_count}, 跳过: {skipped_count}")
                print(f"  Found {image_count} images in class: {folder_name} (mapped to {original_class_name})")

                # 如果没有找到图像，打印调试信息
                if image_count == 0:
                    print(f"    DEBUG actual_path: {actual_path}")
                    print(f"    DEBUG actual_path exists: {os.path.exists(actual_path)}")
                    print(f"    DEBUG contents sample: {os.listdir(actual_path)[:10]}")

    def _load_herlev_images(self, root_dir):
        # 加载Herlev数据集图像
        print(f"Loading Herlev images from: {root_dir}")
        skipped_count = 0
        for class_name in os.listdir(root_dir):
            class_dir = os.path.join(root_dir, class_name)
            if os.path.isdir(class_dir):
                image_count = 0
                for filename in os.listdir(class_dir):
                    if filename.lower().endswith(('.bmp', '.jpg', '.png', '.jpeg')):
                        image_path = os.path.join(class_dir, filename)
                        if self.classification_mode == 'binary':
                            # 使用映射后的类别
                            if class_name in self.class_mapping:
                                mapped_class = self.class_mapping[class_name]
                                self.images.append((image_path, self.class_to_idx[mapped_class]))
                                image_count += 1
                                if image_count <= 5:  # 只打印前5个文件路径用于调试
                                    print(f"    Loaded: {image_path}")
                        else:
                            # 多分类：仅当类名在当前标签空间内才加载
                            if class_name in self.class_to_idx:
                                self.images.append((image_path, self.class_to_idx[class_name]))
                                image_count += 1
                                if image_count <= 5:
                                    print(f"    Loaded: {image_path}")
                            else:
                                skipped_count += 1
                print(f"  Found {image_count} images in class: {class_name}")
                self._record_skipped('herlev', class_name, skipped_count)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_path, label = self.images[idx]

        # 加载图像
        try:
            image = Image.open(image_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            # 返回一个默认的空白图像
            image = Image.new('RGB', (224, 224), color='black')

        # 应用变换
        if self.transform:
            image = self.transform(image)

        return image, label


# 随机擦除变换
class RandomErasing:
    """Randomly selects a rectangle region in an image and erases its pixels."""

    def __init__(self, probability=0.5, sl=0.02, sh=0.4, r1=0.3, mean=(0.4914, 0.4822, 0.4465)):
        self.probability = probability
        self.mean = mean
        self.sl = sl
        self.sh = sh
        self.r1 = r1

    def __call__(self, img):
        if random.uniform(0, 1) >= self.probability:
            return img

        for _ in range(100):
            area = img.size()[1] * img.size()[2]

            target_area = random.uniform(self.sl, self.sh) * area
            aspect_ratio = random.uniform(self.r1, 1 / self.r1)

            h = int(round(torch.sqrt(target_area * aspect_ratio)))
            w = int(round(torch.sqrt(target_area / aspect_ratio)))

            if w < img.size()[2] and h < img.size()[1]:
                x1 = random.randint(0, img.size()[1] - h)
                y1 = random.randint(0, img.size()[2] - w)
                if img.size()[0] == 3:
                    img[0, x1:x1 + h, y1:y1 + w] = self.mean[0]
                    img[1, x1:x1 + h, y1:y1 + w] = self.mean[1]
                    img[2, x1:x1 + h, y1:y1 + w] = self.mean[2]
                else:
                    img[0, x1:x1 + h, y1:y1 + w] = self.mean[0]
                return img

        return img


def get_transforms(config, mode='train'):
    """获取数据增强和预处理变换"""
    img_size = config['data']['img_size']

    if mode == 'train':
        # 增强的数据增强策略
        transform_list = [
            transforms.Resize((int(img_size * 1.2), int(img_size * 1.2))),  # 先放大一点
            transforms.RandomCrop((img_size, img_size)),  # 随机裁剪
            transforms.RandomHorizontalFlip(p=0.5 if config['augmentation'].get('horizontal_flip', True) else 0),
            transforms.RandomVerticalFlip(p=0.5 if config['augmentation'].get('vertical_flip', True) else 0),
            transforms.RandomRotation(config['augmentation'].get('rotation_range', 20)),
            transforms.ColorJitter(
                brightness=config['augmentation'].get('brightness_range', [0.8, 1.2]),
                contrast=config['augmentation'].get('contrast_range', [0.8, 1.2]),
                saturation=config['augmentation'].get('saturation_range', [0.8, 1.2]),
                hue=config['augmentation'].get('hue_range', [-0.1, 0.1])
            ),
            transforms.RandomAffine(
                degrees=0,  # 不再使用旋转，因为已经有RandomRotation
                translate=(config['augmentation'].get('translate_range', 0.1),
                           config['augmentation'].get('translate_range', 0.1)),
                scale=config['augmentation'].get('scale_range', [0.9, 1.1]),
                shear=config['augmentation'].get('shear_range', 10)
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]

        # 添加随机擦除
        if config['augmentation'].get('random_erasing', False):
            transform_list.append(
                RandomErasing(
                    probability=config['augmentation'].get('random_erasing_prob', 0.5),
                    mean=[0.485, 0.456, 0.406]
                )
            )
    else:
        transform_list = [
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]

    return transforms.Compose(transform_list)