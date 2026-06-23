"""
宫颈细胞检测YOLOv8训练脚本
"""

import argparse
import os
from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser(description='Train YOLOv8 for cervical cell detection')
    parser.add_argument('--data', type=str, required=True,
                        help='Path to dataset YAML configuration file')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--batch', type=int, default=16,
                        help='Batch size')
    parser.add_argument('--imgsz', type=int, default=640,
                        help='Image size')
    parser.add_argument('--device', type=int, default=0,
                        help='GPU device ID (0 for first GPU)')
    parser.add_argument('--model', type=str, default='yolov8n.pt',
                        help='Pre-trained model path (yolov8n/s/m/l/x.pt)')
    parser.add_argument('--name', type=str, default='cervical_cell_exp',
                        help='Experiment name')
    parser.add_argument('--patience', type=int, default=10,
                        help='Early stopping patience')
    parser.add_argument('--lr0', type=float, default=0.01,
                        help='Initial learning rate')

    args = parser.parse_args()

    # 创建输出目录
    os.makedirs('runs/detect', exist_ok=True)

    # 加载模型
    print(f"Loading model: {args.model}")
    model = YOLO(args.model)

    # 训练配置
    train_config = {
        'data': args.data,
        'epochs': args.epochs,
        'batch': args.batch,
        'imgsz': args.imgsz,
        'device': args.device,
        'name': args.name,
        'patience': args.patience,
        'lr0': args.lr0,
        'optimizer': 'SGD',
        'momentum': 0.937,
        'weight_decay': 0.0005,
        'warmup_epochs': 3,
        'warmup_momentum': 0.8,
        'warmup_bias_lr': 0.1,
        'box': 7.5,  # box loss gain
        'cls': 0.5,  # cls loss gain
        'dfl': 1.5,  # dfl loss gain
        'hsv_h': 0.015,  # HSV-Hue augmentation
        'hsv_s': 0.7,  # HSV-Saturation augmentation
        'hsv_v': 0.4,  # HSV-Value augmentation
        'degrees': 0.0,  # rotation augmentation
        'translate': 0.1,  # translation augmentation
        'scale': 0.5,  # scale augmentation
        'shear': 0.0,  # shear augmentation
        'perspective': 0.0,  # perspective augmentation
        'flipud': 0.0,  # vertical flip augmentation
        'fliplr': 0.5,  # horizontal flip augmentation
        'mosaic': 1.0,  # mosaic augmentation
        'mixup': 0.0,  # mixup augmentation
        'copy_paste': 0.0,  # copy-paste augmentation
        'verbose': True
    }

    print(f"Starting training...")
    print(f"Dataset: {args.data}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch}")
    print(f"Image size: {args.imgsz}")
    print(f"Device: {args.device}")

    # 开始训练
    results = model.train(**train_config)

    print(f"\nTraining completed!")
    print(f"Best model saved to: runs/detect/{args.name}/weights/best.pt")
    print(f"Last model saved to: runs/detect/{args.name}/weights/last.pt")

    return results


if __name__ == '__main__':
    main()
