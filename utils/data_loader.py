import os
import random
from collections import defaultdict

import torch
from torch.utils.data import DataLoader, ConcatDataset, Subset

def case_key(image_path: str) -> str:
    """提取"病例/视野"标识，用于分组划分。

    同一病例会产出多张细胞裁剪图（如 001_01 / 001_02），必须落在同一 split，
    否则测试集里的病例模型全都见过 → 指标虚高。

    SIPaKMeD: im_Koilocytotic/im_Koilocytotic/CROPPED/001_01.bmp -> CROPPED/001
    Herlev:   light_dysplastic/00001.bmp                         -> light_dysplastic/00001
    """
    parts = os.path.normpath(image_path).split(os.sep)
    stem = os.path.splitext(parts[-1])[0]
    dirs = parts[:-1]

    # 跳过 CROPPED 这一层，取真正的类别目录
    if dirs and dirs[-1] == "CROPPED":
        cls_dir = dirs[-2] if len(dirs) >= 2 else ""
    else:
        cls_dir = dirs[-1] if dirs else ""

    if "_" in stem:
        case = stem.split("_")[0]                  # 001_01 -> 001
    else:
        seg = stem.split("-")
        case = "-".join(seg[:2]) if len(seg) >= 3 else stem

    return f"{cls_dir}/{case}"

def group_split_indices(dataset, train_ratio: float, val_ratio: float,
                        seed: int = 42):
    """按病例分组划分数据集，返回 (train_idx, val_idx, test_idx)。

    Args:
        dataset: CombinedCervicalCancerDataset（需要 .images = [(path, label)]）
    """
    groups = defaultdict(list)
    for idx, (path, _label) in enumerate(dataset.images):
        groups[case_key(path)].append(idx)

    keys = sorted(groups)
    random.Random(seed).shuffle(keys)

    n = len(keys)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    train_keys = keys[:n_train]
    val_keys = keys[n_train:n_train + n_val]
    test_keys = keys[n_train + n_val:]

    def _flat(sub):
        return [i for k in sub for i in groups[k]]

    train_idx, val_idx, test_idx = _flat(train_keys), _flat(val_keys), _flat(test_keys)

    # 硬校验：任何病例都不得跨 split
    overlap = (set(train_keys) & set(test_keys)) | (set(train_keys) & set(val_keys)) \
              | (set(val_keys) & set(test_keys))
    if overlap:
        raise RuntimeError(f"病例泄漏：{len(overlap)} 个病例跨 split")

    if not train_idx or not val_idx or not test_idx:
        raise ValueError(
            f"分组划分后出现空集：train={len(train_idx)} val={len(val_idx)} test={len(test_idx)}；"
            f"病例总数 {n}，请检查 train/val 比例或数据量")

    print(f"[split] 病例数 train={len(train_keys)} val={len(val_keys)} test={len(test_keys)} "
          f"| 图像数 train={len(train_idx)} val={len(val_idx)} test={len(test_idx)}")
    return train_idx, val_idx, test_idx

def create_data_loaders(config=None, classification_mode='binary', label_space=None):
    """创建训练、验证和测试数据加载器

    Args:
        config: 配置文件
        classification_mode: 分类模式 ('binary' 或 'multiclass')
        label_space: 多分类标签空间名；None 时从 config['model']['label_space'] 读取
    """
    # 将导入语句移到函数内部，避免循环导入
    from .preprocess import CombinedCervicalCancerDataset, get_transforms, load_config

    config = config or load_config()

    # 未显式传入时回落到配置文件
    if label_space is None:
        label_space = (config.get('model') or {}).get('label_space')

    # 获取数据转换
    train_transform = get_transforms(config, 'train')
    test_transform = get_transforms(config, 'test')

    # 加载数据集配置
    dataset_config = config['data']['datasets']

    # 添加调试信息：打印数据集配置
    print("=== Dataset Configuration ===")
    for ds in dataset_config:
        print(f"Dataset type: {ds['type']}, Path: {ds['path']}")

    # 创建组合数据集
    combined_dataset = CombinedCervicalCancerDataset(
        dataset_config, train_transform, config, classification_mode,
        label_space=label_space,
    )

    # 添加调试信息：检查数据集大小
    print(f"\n=== Dataset Sizes ===")
    print(f"Combined dataset size: {len(combined_dataset)}")

    if len(combined_dataset) == 0:
        print("\n=== Error: No images found in the dataset ===")
        # 打印每个数据集的详细信息
        for dataset_info in dataset_config:
            dataset_type = dataset_info['type']
            root_dir = dataset_info['path']
            print(f"\nChecking {dataset_type} dataset at {root_dir}:")

            if os.path.exists(root_dir):
                print(f"  Directory exists: {os.path.exists(root_dir)}")
                print(f"  Is directory: {os.path.isdir(root_dir)}")

                if os.path.isdir(root_dir):
                    classes = os.listdir(root_dir)
                    print(f"  Classes found: {classes}")

                    for class_name in classes:
                        class_path = os.path.join(root_dir, class_name)
                        if os.path.isdir(class_path):
                            image_files = [f for f in os.listdir(class_path) if
                                           f.lower().endswith(('.bmp', '.jpg', '.png', '.jpeg'))]
                            print(f"    Class '{class_name}': {len(image_files)} images")

                            # 打印前5个图像文件名
                            if image_files:
                                print(f"    First 5 files: {image_files[:5]}")
                        else:
                            print(f"    '{class_name}' is not a directory")
                else:
                    print(f"  {root_dir} is not a directory")
            else:
                print(f"  Directory does not exist: {root_dir}")

        raise ValueError("No images found in the dataset. Please check your data paths and file structure.")

    # 计算划分大小
    train_ratio = config['data']['train_ratio']
    val_ratio = config['data']['val_ratio']
    test_ratio = config['data']['test_ratio']

    total_size = len(combined_dataset)
    print(f"Total dataset size: {total_size}")
    print(f"Train ratio: {train_ratio}, Val ratio: {val_ratio}, Test ratio: {test_ratio}")

    # 确保划分比例之和不超过1
    if train_ratio + val_ratio + test_ratio > 1.0:
        raise ValueError("Sum of train_ratio, val_ratio, and test_ratio should not exceed 1.0")


    # 划分数据集
    train_idx, val_idx, test_idx = group_split_indices(
        combined_dataset, train_ratio, val_ratio, seed=42)
    train_dataset = Subset(combined_dataset, train_idx)
    val_dataset = Subset(combined_dataset, val_idx)
    test_dataset = Subset(combined_dataset, test_idx)

    # 创建数据加载器
    batch_size = config['data']['batch_size']
    print(f"Using batch size: {batch_size}")

    # 根据批量大小调整num_workers
    num_workers = min(4, batch_size // 4)  # 根据批量大小调整工作线程数
    num_workers = max(1, num_workers)  # 至少保持1个工作线程

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True
    )

    return train_loader, val_loader, test_loader