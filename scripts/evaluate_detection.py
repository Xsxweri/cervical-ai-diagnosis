"""
YOLOv8 宫颈细胞检测模型评估脚本
支持: 标准COCO指标(mAP)、异常细胞检测分析、置信度阈值调参、可视化对比
"""

import sys
import os
import argparse
import json
from pathlib import Path
from typing import List, Dict, Tuple
from collections import defaultdict
import numpy as np

# 添加项目根目录到Python路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.insert(0, project_root)

from ultralytics import YOLO
import cv2
from utils.label_mapping import (
    SIPAKMED_CLASSES,
    SIPAKMED_CLASS_TO_IDX,
    SIPAKMED_ABNORMAL_CLASSES,
    SIPAKMED_NORMAL_CLASSES,
)

class DetectionEvaluator:
    """
    YOLOv8检测模型评估器
    """

    # SiPaKMeD 5类细胞定义
    CLASS_NAMES = {i: name for i, name in enumerate(SIPAKMED_CLASSES)}

    # 医学分类: 异常 vs 正常
    ABNORMAL_CLASSES = {SIPAKMED_CLASS_TO_IDX[n] for n in SIPAKMED_ABNORMAL_CLASSES}
    NORMAL_CLASSES = {SIPAKMED_CLASS_TO_IDX[n] for n in SIPAKMED_NORMAL_CLASSES}

    def __init__(self, model_path: str, conf: float = 0.001, iou: float = 0.7,deploy_conf: float = 0.25, deploy_iou: float = 0.45):
        """
        初始化评估器

        Args:
            model_path: YOLO模型路径
            conf: 置信度阈值
            iou: NMS IoU阈值
        """
        self.model = YOLO(model_path)
        self.conf, self.iou = conf, iou  # 评估用
        self.deploy_conf, self.deploy_iou = deploy_conf, deploy_iou # 统计用
        print(f" 加载模型: {model_path}")
        print(f"   置信度阈值: {conf}, IoU阈值: {iou}")

    def evaluate_coco_metrics(self, data_yaml: str) -> Dict:
        """
        标准COCO指标评估 (mAP@0.5, mAP@0.5:0.95等)

        Args:
            data_yaml: 数据集YAML配置文件路径

        Returns:
            指标字典
        """
        print("\n" + "=" * 50)
        print("[1/3] 标准COCO指标评估")
        print("=" * 50)

        # 运行验证
        metrics = self.model.val(
            data=data_yaml,
            conf=self.conf,
            iou=self.iou,
            verbose=False
        )

        results = {
            'eval_params': {'conf': self.conf, 'iou': self.iou}, 
            'mAP50': float(metrics.box.map50),
            'mAP75': float(metrics.box.map75),
            'mAP50_95': float(metrics.box.map),
            'precision': float(metrics.box.mp),
            'recall': float(metrics.box.mr),
            'per_class': {}
        }

        # 每类AP
        for i, name in self.CLASS_NAMES.items():
            cls_result = {}
            # 使用 class_result 方法获取每类指标
            try:
                p, r, ap50, ap = metrics.box.class_result(i)
                cls_result = {
                    'precision': float(p),
                    'recall': float(r),
                    'AP50': float(ap50),
                    'AP50_95': float(ap),
                }
            except (IndexError, AttributeError):
                # 如果该类没有数据
                cls_result = {
                    'precision': 0.0,
                    'recall': 0.0,
                    'AP50': 0.0,
                    'AP50_95': 0.0,
                }

            results['per_class'][name] = cls_result

        # 打印结果
        print(f"\n整体指标:")
        print(f"     mAP@0.5:    {results['mAP50']:.4f}")
        print(f"     mAP@0.75:   {results['mAP75']:.4f}")
        print(f"     mAP@0.5:0.95: {results['mAP50_95']:.4f}")
        print(f"     Precision:  {results['precision']:.4f}")
        print(f"     Recall:     {results['recall']:.4f}")

        print(f"\n  📋 每类指标:")
        for name, cls_metrics in results['per_class'].items():
            print(f"{name:20s}: P={cls_metrics['precision']:.4f}, R={cls_metrics['recall']:.4f}, AP50={cls_metrics['AP50']:.4f}")

        results['metric_caveat'] = (
            "SIPaKMeD 单细胞裁剪图,每图1个目标框(≈整图)且无背景干扰；"
            "mAP50 ≈ mAP75属预期现象,不反映真实多细胞视野的检测能力。"
        )

        return results

    def evaluate_abnormal_detection(self, image_dir: str, gt_dir: str = None) -> Dict:
        """
        异常细胞检测评估（医学场景专用）

        Args:
            image_dir: 测试图像目录
            gt_dir: 真实标注目录(可选，用于计算准确率)

        Returns:
            异常检测统计结果
        """
        print("\n" + "=" * 50)
        print("[2/3] 异常细胞检测分析")
        print("=" * 50)

        image_paths = list(Path(image_dir).glob('*'))
        image_paths = [p for p in image_paths if p.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff']]

        if not image_paths:
            print(f"未找到图像: {image_dir}")
            return {}

        total_stats = {
            'eval_params': {'deploy_conf': self.deploy_conf, 'deploy_iou': self.deploy_iou},
            'total_images': len(image_paths),
            'total_cells': 0,
            'abnormal_cells': 0,
            'normal_cells': 0,
            'per_image': [],
            'per_class_counts': defaultdict(int)
        }

        for img_path in image_paths:
            # 检测
            results = self.model(str(img_path), conf=self.deploy_conf,iou=self.deploy_iou, verbose=False)[0]

            img_stats = {
                'image': str(img_path.name),
                'total_cells': 0,
                'abnormal_cells': 0,
                'normal_cells': 0,
                'class_distribution': defaultdict(int)
            }

            for box in results.boxes:
                cls_id = int(box.cls)
                conf = float(box.conf)
                img_stats['total_cells'] += 1
                img_stats['class_distribution'][cls_id] += 1
                total_stats['per_class_counts'][self.CLASS_NAMES.get(cls_id, f'Class_{cls_id}')] += 1

                if cls_id in self.ABNORMAL_CLASSES:
                    img_stats['abnormal_cells'] += 1
                    total_stats['abnormal_cells'] += 1
                else:
                    img_stats['normal_cells'] += 1
                    total_stats['normal_cells'] += 1

            total_stats['total_cells'] += img_stats['total_cells']
            total_stats['per_image'].append(img_stats)

        # 计算比例
        if total_stats['total_cells'] > 0:
            total_stats['abnormal_ratio'] = total_stats['abnormal_cells'] / total_stats['total_cells']
            total_stats['normal_ratio'] = total_stats['normal_cells'] / total_stats['total_cells']
        else:
            total_stats['abnormal_ratio'] = 0
            total_stats['normal_ratio'] = 0

        total_stats['avg_cells_per_image'] = total_stats['total_cells'] / len(image_paths) if image_paths else 0

        # 打印结果
        print(f"\n处理图像: {total_stats['total_images']} 张")
        print(f"总细胞数: {total_stats['total_cells']}")
        print(f"异常细胞: {total_stats['abnormal_cells']} ({total_stats['abnormal_ratio']:.2%})")
        print(f"正常细胞: {total_stats['normal_cells']} ({total_stats['normal_ratio']:.2%})")
        print(f"平均每图: {total_stats['avg_cells_per_image']:.1f} 个细胞")

        print(f"\n各类别分布:")
        for cls_name, count in sorted(total_stats['per_class_counts'].items()):
            ratio = count / total_stats['total_cells'] if total_stats['total_cells'] > 0 else 0
            print(f"     {cls_name:20s}: {count:4d} ({ratio:.2%})")

        return dict(total_stats)

    def evaluate_conf_thresholds(self, data_yaml: str, thresholds: List[float] = None) -> Dict:
        """
        置信度阈值调参评估
        """
        if thresholds is None:
            thresholds = [0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7]

        print("\n" + "=" * 50)
        print("[3/3] 置信度阈值调参")
        print("=" * 50)
        print(f"\n  测试阈值: {thresholds}")

        threshold_results = []

        for thresh in thresholds:
            print(f"\n  测试阈值 {thresh}...")
            metrics = self.model.val(
                data=data_yaml,
                conf=thresh,
                iou=self.iou,
                verbose=False
            )

            result = {
                'threshold': thresh,
                'mAP50': float(metrics.box.map50),
                'mAP50_95': float(metrics.box.map),
                'precision': float(metrics.box.mp),
                'recall': float(metrics.box.mr),
            }
            threshold_results.append(result)

        # 打印对比表
        print(f"\n  {'Threshold':>10s} | {'mAP@0.5':>8s} | {'mAP@0.5:0.95':>12s} | {'Precision':>9s} | {'Recall':>7s}")
        print("  " + "-" * 60)
        for r in threshold_results:
            print(f"  {r['threshold']:>10.2f} | {r['mAP50']:>8.4f} | {r['mAP50_95']:>12.4f} | {r['precision']:>9.4f} | {r['recall']:>7.4f}")

        # 找到最优阈值
        best_map50 = max(threshold_results, key=lambda x: x['mAP50'])
        best_recall = max(threshold_results, key=lambda x: x['recall'])
        best_precision = max(threshold_results, key=lambda x: x['precision'])

        print(f"\n  最优mAP@0.5:   threshold={best_map50['threshold']}, mAP50={best_map50['mAP50']:.4f}")
        print(f"  最优Recall:    threshold={best_recall['threshold']}, recall={best_recall['recall']:.4f}")
        print(f"  最优Precision: threshold={best_precision['threshold']}, precision={best_precision['precision']:.4f}")

        return {
            'thresholds_tested': thresholds,
            'results': threshold_results,
            'best': {
                'mAP50': best_map50,
                'recall': best_recall,
                'precision': best_precision
            }
        }

    def visualize_samples(self, image_dir: str, output_dir: str, num_samples: int = 5):
        """
        可视化样本检测结果

        Args:
            image_dir: 图像目录
            output_dir: 输出目录
            num_samples: 可视化样本数
        """
        os.makedirs(output_dir, exist_ok=True)

        image_paths = list(Path(image_dir).glob('*'))
        image_paths = [p for p in image_paths if p.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']]

        if not image_paths:
            return

        # 随机选择样本
        np.random.shuffle(image_paths)
        selected = image_paths[:num_samples]

        for img_path in selected:
            results = self.model(str(img_path), conf=self.conf, verbose=False)[0]

            # 绘制结果
            annotated = results.plot(line_width=2, font_size=0.6)

            # 保存
            out_path = os.path.join(output_dir, f"vis_{img_path.name}")
            cv2.imwrite(out_path, annotated)

        print(f"\n 可视化结果已保存: {output_dir} ({len(selected)}张)")


def main():
    parser = argparse.ArgumentParser(description='Evaluate YOLOv8 cervical cell detection model')
    parser.add_argument('--model', type=str, required=True,
                        help='Path to trained YOLO model (best.pt)')
    parser.add_argument('--data', type=str, default='configs/sipakmed_yolo.yaml',
                        help='Dataset YAML config for COCO metrics')
    parser.add_argument('--image_dir', type=str,
                        help='Directory of test images for abnormal cell analysis')
    parser.add_argument('--conf', type=float, default=0.001,
                        help='Confidence threshold (COCO 评估用 0.001;部署推理用 0.25)')
    parser.add_argument('--iou', type=float, default=0.7,
                        help='NMS IoU threshold (COCO 评估用 0.7;部署可用 0.45)')
    parser.add_argument('--output', type=str, default='outputs/detection_evaluation.json',
                        help='Output path for evaluation results')
    parser.add_argument('--visualize', action='store_true',
                        help='Generate visualization samples')
    parser.add_argument('--vis_dir', type=str, default='outputs/detection_vis',
                        help='Visualization output directory')
    parser.add_argument('--vis_num', type=int, default=5,
                        help='Number of visualization samples')
    parser.add_argument('--tune_conf', action='store_true',
                        help='Run confidence threshold tuning')
    parser.add_argument('--conf_range', type=float, nargs='+',
                        default=[0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5],
                        help='Confidence thresholds to test')

    args = parser.parse_args()

    # 初始化评估器
    evaluator = DetectionEvaluator(args.model, args.conf, args.iou)

    # 收集所有结果
    all_results = {
        'model_path': args.model,
        'eval_params': {
            # COCO mAP 评估用低阈值，让 PR 曲线覆盖完整召回区间
            'coco_conf': args.conf,
            'coco_iou': args.iou,
            # 异常细胞统计用部署阈值（模拟真实推理）
            'deploy_conf': evaluator.deploy_conf,
            'deploy_iou': evaluator.deploy_iou,
            # 环境版本：ultralytics 升级会改变默认代次与评估实现
            'ultralytics_version': __import__('ultralytics').__version__,
        },
        'dataset': {
            'yaml': args.data,
            'split': 'val',
        },
    }

    # 1. 标准COCO指标
    if args.data and os.path.exists(args.data):
        coco_results = evaluator.evaluate_coco_metrics(args.data)
        all_results['coco_metrics'] = coco_results
    else:
        print(f"\n 数据集配置未找到: {args.data}，跳过COCO指标评估")

    # 2. 异常细胞检测分析
    if args.image_dir and os.path.exists(args.image_dir):
        abnormal_results = evaluator.evaluate_abnormal_detection(args.image_dir)
        all_results['abnormal_detection'] = abnormal_results
    else:
        print(f"\n 测试图像目录未指定或不存在，跳过异常细胞分析")

    # 3. 置信度阈值调参
    if args.tune_conf and args.data and os.path.exists(args.data):
        tune_results = evaluator.evaluate_conf_thresholds(args.data, args.conf_range)
        all_results['threshold_tuning'] = tune_results

    # 4. 可视化
    if args.visualize and args.image_dir and os.path.exists(args.image_dir):
        evaluator.visualize_samples(args.image_dir, args.vis_dir, args.vis_num)
        all_results['visualization'] = {
            'output_dir': args.vis_dir,
            'num_samples': args.vis_num
        }

    # 保存结果
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    print(f"\n{'=' * 50}")
    print(f" 评估完成！结果保存: {args.output}")
    print(f"{'=' * 50}")

    # 打印最终建议
    print("\n 使用建议:")
    if 'threshold_tuning' in all_results:
        best = all_results['threshold_tuning']['best']
        print(f"   • 追求最高mAP:   conf={best['mAP50']['threshold']}")
        print(f"   • 追求最高召回:  conf={best['recall']['threshold']} (减少漏诊)")
        print(f"   • 追求最高精确:  conf={best['precision']['threshold']} (减少误诊)")


if __name__ == '__main__':
    main()