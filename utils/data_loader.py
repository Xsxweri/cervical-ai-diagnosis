import torch
from torch.utils.data import DataLoader, ConcatDataset, random_split
import os


def create_data_loaders(config=None, classification_mode='binary'):
    """创建训练、验证和测试数据加载器

    Args:
        config: 配置文件
        classification_mode: 分类模式 ('binary' 或 'multiclass')
    """
    # 将导入语句移到函数内部，避免循环导入
    from .preprocess import CombinedCervicalCancerDataset, get_transforms, load_config

    config = config or load_config()

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
        dataset_config, train_transform, config, classification_mode
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

    train_size = int(train_ratio * total_size)
    val_size = int(val_ratio * total_size)
    test_size = total_size - train_size - val_size

    # 添加调试信息：检查划分大小
    print(f"Train size: {train_size}, Val size: {val_size}, Test size: {test_size}")

    if train_size <= 0:
        raise ValueError(f"Train size is {train_size}, which is not positive. Check your train_ratio setting.")
    if val_size <= 0:
        raise ValueError(f"Val size is {val_size}, which is not positive. Check your val_ratio setting.")
    if test_size <= 0:
        raise ValueError(f"Test size is {test_size}, which is not positive. Check your test_ratio setting.")

    # 划分数据集
    train_dataset, val_dataset, test_dataset = random_split(
        combined_dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)  # 确保可重复性
    )

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