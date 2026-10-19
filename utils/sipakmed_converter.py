"""
将SiPaKMeD数据集转换为YOLO格式的工具。
------------------------------------------------------------------
1. 文件名冲突导致近半数据被静默覆盖
2. 切分不可复现
3. 标签语义错误
4. 缺少完整性校验

注意：SiPaKMeD 是"单细胞裁剪图"分类数据集，用它训练出的检测器无法泛化到真实多细胞视野（mAP会虚高）。
要获得可信的细胞检测能力，必须补充真实视野（多细胞、重叠、炎症/杂质背景）的实例标注数据。
"""

import os
import sys
import cv2
import numpy as np
from typing import Dict, List, Optional, Tuple
import random
import shutil
from pathlib import Path
from collections import defaultdict

try:
    from utils.label_mapping import (
        SIPAKMED_CLASSES,
        SIPAKMED_DIR_MAPPING,
        NUCLEUS_CLASS_NAME,
    )
except ImportError:  # 支持直接以脚本方式运行: python utils/sipakmed_converter.py
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from utils.label_mapping import (
        SIPAKMED_CLASSES,
        SIPAKMED_DIR_MAPPING,
        NUCLEUS_CLASS_NAME,
    )



class SiPaKMeDConverter:
    """SiPaKMeD数据集转换器，将数据集转换为YOLO格式"""

    def __init__(self, dataset_root: str, output_root: str, seed: int = 42,
                 include_nucleus: bool = False, clean: bool = True):
        """
        Args:
            dataset_root: SiPaKMeD数据集根目录
            output_root: 输出YOLO格式数据集的根目录
            seed: 划分随机种子（固定以保证可复现）
            include_nucleus: 是否额外输出胞核框（独立 nucleus 类别）
            clean: 转换前是否清空输出目录中的旧文件（避免残留孤儿文件）
        """
        self.dataset_root = dataset_root
        self.output_root = output_root
        self.seed = seed
        self.include_nucleus = include_nucleus
        self.clean = clean

        # 类别 id 与顺序统一来自 canonical 定义（禁止就地硬编码）
        self.classes = {
            dir_name: SIPAKMED_CLASSES.index(class_name)
            for dir_name, class_name in SIPAKMED_DIR_MAPPING.items()
        }
        # 胞核类别 id 追加在细胞类别之后
        self.nucleus_class_id: Optional[int] = (
            len(SIPAKMED_CLASSES) if include_nucleus else None
        )

        # 创建输出目录结构
        self._create_output_structure()

    def _create_output_structure(self):
        """创建输出目录结构；clean=True 时清空旧的 images/labels 内容"""
        for split in ('train', 'val', 'test'):
            for kind in ('images', 'labels'):
                os.makedirs(os.path.join(self.output_root, kind, split), exist_ok=True)
        if self.clean:
            self._clean_output()

    def _clean_output(self):
        """清空输出目录中的图片/标签文件，避免重跑时残留孤儿文件"""
        removed = 0
        for split in ('train', 'val', 'test'):
            for kind in ('images', 'labels'):
                d = os.path.join(self.output_root, kind, split)
                for name in os.listdir(d):
                    p = os.path.join(d, name)
                    if os.path.isfile(p):
                        os.remove(p)
                        removed += 1
        if removed:
            print(f"[clean] 已清理输出目录中的 {removed} 个旧文件")


    def _dat_search_dirs(self, class_path: str) -> List[str]:
        """
        .dat 文件可能与被选中的图像目录同级或位于其父目录，
        依次返回存在的候选目录。
        """
        dirs = [class_path]
        parent = os.path.dirname(class_path.rstrip("/\\"))
        if parent and parent != class_path:
            dirs.append(parent)
        return [d for d in dirs if os.path.isdir(d)]

    def _find_dat_files(self, class_path: str, base_name: str, kind: str) -> List[str]:
        """
        查找某个基名对应的轮廓文件。

        Args:
            base_name: 图像基名，如 ``001_01`` 或 ``001``
            kind: ``'cyt'``(胞浆) 或 ``'nuc'``(胞核)
        Returns:
            排序后的 .dat 文件绝对路径列表
        """
        pattern = f"{base_name}_{kind}"
        for d in self._dat_search_dirs(class_path):
            found = sorted(
                f for f in os.listdir(d)
                if f.startswith(pattern) and f.endswith('.dat')
            )
            if found:
                return [os.path.join(d, f) for f in found]
        return []

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

    def convert_dataset(self, train_ratio: float = 0.7, val_ratio: float = 0.15,
                        seed: Optional[int] = None) -> Dict:
        """
        转换整个数据集。

        Args:
            train_ratio: 训练集比例
            val_ratio: 验证集比例
            seed: 随机种子，缺省使用构造时的 self.seed

        Returns:
            转换报告 dict（各类别/各 split 的图像数、总图像数）
        """
        seed = self.seed if seed is None else seed
        rng = random.Random(seed)  # 局部 RNG，避免污染全局 random 状态
        print(f"开始转换SiPaKMeD数据集为YOLO格式... (seed={seed}, "
              f"include_nucleus={self.include_nucleus})")

        report: Dict = {
            "seed": seed,
            "include_nucleus": self.include_nucleus,
            "splits": {s: {} for s in ('train', 'val', 'test')},
            "missing_classes": [],
            "total": {},
        }

        # 遍历每个类别目录
        for class_dir, class_id in self.classes.items():
            class_path = os.path.join(self.dataset_root, class_dir)
            if not os.path.exists(class_path):
                print(f"警告: 类别目录不存在 {class_path}")
                report["missing_classes"].append(class_dir)
                continue

            # 检查是否存在双层嵌套结构
            nested_path = os.path.join(class_path, class_dir)
            if os.path.isdir(nested_path):
                actual_class_path = nested_path
                print(f"检测到双层嵌套结构，使用: {actual_class_path}")
            else:
                actual_class_path = class_path

            # 优先检查 CROPPED 子目录（实际数据和 .dat 文件所在位置）
            cropped_path = os.path.join(actual_class_path, 'CROPPED')
            data_path = cropped_path if os.path.isdir(cropped_path) else actual_class_path
            print(f"处理类别: {class_dir} (ID: {class_id}) -> {data_path}")

            # 获取所有 .bmp 图像文件；先排序再洗牌，保证可复现
            bmp_files = sorted(
                f for f in os.listdir(data_path) if f.lower().endswith('.bmp')
            )
            # 按病例分组划分：同一 001_xx 的多张裁剪图必须落在同一 split
            by_case = defaultdict(list)
            for f in bmp_files:
                by_case[os.path.splitext(f)[0].split("_")[0]].append(f)

            case_ids = sorted(by_case)
            rng.shuffle(case_ids)

            n_cases = len(case_ids)
            tr_end = int(n_cases * train_ratio)
            va_end = int(n_cases * (train_ratio + val_ratio))

            def _files_of(ids):
                out = []
                for c in ids:
                    out.extend(by_case[c])
                return out

            split_files = {
                'train': _files_of(case_ids[:tr_end]),
                'val':   _files_of(case_ids[tr_end:va_end]),
                'test':  _files_of(case_ids[va_end:]),
            }

            for split, files in split_files.items():
                written = self._process_files(data_path, files, split, class_id, class_dir)
                report["splits"][split][class_dir] = written
                print(f"  {split}: {written}/{len(files)} 张图像")

        for split in ('train', 'val', 'test'):
            report["total"][split] = sum(report["splits"][split].values())

        print(f"数据集转换完成! 总计: {report['total']}")
        return report

    @staticmethod
    def _format_box(class_id: int, box: Tuple[float, float, float, float]) -> Optional[str]:
        """把 (cx,cy,w,h) 归一化框裁剪到 [0,1] 并格式化为 YOLO 一行；非法返回 None。"""
        cx, cy, w, h = box
        x1 = max(0.0, cx - w / 2.0)
        x2 = min(1.0, cx + w / 2.0)
        y1 = max(0.0, cy - h / 2.0)
        y2 = min(1.0, cy + h / 2.0)
        nw, nh = x2 - x1, y2 - y1
        if nw <= 1e-6 or nh <= 1e-6:
            return None
        return (f"{class_id} {(x1 + x2) / 2:.6f} {(y1 + y2) / 2:.6f} "
                f"{nw:.6f} {nh:.6f}")

    def _process_files(self, class_path: str, files: List[str], split: str,
                       class_id: int, class_dir: str) -> int:
        """
        处理指定集合的文件。

        修复点:
          * 输出文件名加类别前缀 -> 全局唯一，杜绝同名覆盖；
          * cyt 轮廓 -> 细胞实例框(class=细胞类型)；nuc 轮廓 -> 可选 nucleus 类别。
        """
        written = 0
        for filename in files:
            img_path = os.path.join(class_path, filename)
            if not os.path.exists(img_path):
                continue
            img = cv2.imread(img_path)
            if img is None:
                continue
            img_height, img_width = img.shape[:2]
            base_name = os.path.splitext(filename)[0]

            # === 关键修复：输出文件名加类别前缀，保证跨类别/跨 split 全局唯一 ===
            out_name = f"{class_dir}__{filename}"
            output_img_path = os.path.join(self.output_root, 'images', split, out_name)
            shutil.copy2(img_path, output_img_path)

            lines: List[str] = []

            # 1) 细胞实例框：每个胞浆轮廓一个实例，class = 细胞类型
            for dat_path in self._find_dat_files(class_path, base_name, 'cyt'):
                points = self._parse_dat_file(dat_path)
                if not points:
                    continue
                box = self._convert_to_yolo_bbox(points, img_width, img_height)
                line = self._format_box(class_id, box)
                if line:
                    lines.append(line)

            # 2) 胞核框（可选）：以独立 nucleus 类别输出，避免与"细胞"混淆
            if self.include_nucleus and self.nucleus_class_id is not None:
                for dat_path in self._find_dat_files(class_path, base_name, 'nuc'):
                    points = self._parse_dat_file(dat_path)
                    if not points:
                        continue
                    box = self._convert_to_yolo_bbox(points, img_width, img_height)
                    line = self._format_box(self.nucleus_class_id, box)
                    if line:
                        lines.append(line)

            label_path = os.path.join(
                self.output_root, 'labels', split,
                os.path.splitext(out_name)[0] + '.txt'
            )
            with open(label_path, 'w') as f:
                if lines:
                    f.write("\n".join(lines) + "\n")

            written += 1
        return written

def create_yolo_config(dataset_root: str, output_path: str,
                       include_nucleus: bool = False):
    """
    创建YOLO配置文件（类别名/数量统一来自 canonical 定义）。

    Args:
        dataset_root: 数据集根目录
        output_path: 输出配置文件路径
        include_nucleus: 是否包含 nucleus 类别（与转换时保持一致）
    """
    names = list(SIPAKMED_CLASSES) + ([NUCLEUS_CLASS_NAME] if include_nucleus else [])
    names_block = "\n".join(f"  {i}: {n}" for i, n in enumerate(names))

    config_content = f"""# SiPaKMeD宫颈细胞数据集配置文件（用于YOLOv8训练）

# 数据集根目录路径
path: {dataset_root}

# 训练、验证、测试集路径
train: images/train
val: images/val
test: images/test

# 类别数量
nc: {len(names)}

# 类别名称
names:
{names_block}
"""

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(config_content)

    print(f"YOLO配置文件已创建: {output_path} (nc={len(names)})")


def validate_yolo_dataset(output_root: str) -> Dict:
    """
    校验转换后的 YOLO 数据集完整性，返回问题列表与统计信息。

    检查项:
      1. 图像 basename 全局唯一（捕捉同名覆盖/跨 split 重复）
      2. 每个图像都有对应标签文件，反之亦然
      3. 标签行格式合法、坐标在 [0,1]、宽高 > 0
      4. 各 split / 各类别统计

    Returns:
        {"ok": bool, "issues": [...], "stats": {...}}
    """
    issues: List[str] = []
    splits = ('train', 'val', 'test')
    img_root = os.path.join(output_root, 'images')
    lbl_root = os.path.join(output_root, 'labels')

    seen_stems: Dict[str, str] = {}
    stats: Dict = {"per_split": {}, "per_class": {s: {} for s in splits}}
    total_boxes = 0

    for split in splits:
        img_dir = os.path.join(img_root, split)
        lbl_dir = os.path.join(lbl_root, split)
        imgs = sorted(f for f in os.listdir(img_dir)
                      if f.lower().endswith(('.bmp', '.jpg', '.jpeg', '.png')))
        lbls = sorted(f for f in os.listdir(lbl_dir) if f.endswith('.txt'))
        img_stems = {os.path.splitext(f)[0] for f in imgs}
        lbl_stems = {os.path.splitext(f)[0] for f in lbls}

        for stem in img_stems:
            if stem in seen_stems:
                issues.append(
                    f"duplicate stem '{stem}' in {seen_stems[stem]} and {split}"
                )
            else:
                seen_stems[stem] = split

        for stem in img_stems - lbl_stems:
            issues.append(f"[{split}] 图像缺少标签: {stem}")
        for stem in lbl_stems - img_stems:
            issues.append(f"[{split}] 标签缺少图像: {stem}")

        box_count = 0
        for lbl in lbls:
            with open(os.path.join(lbl_dir, lbl)) as f:
                for ln, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split()
                    if len(parts) != 5:
                        issues.append(f"[{split}/{lbl}:{ln}] 字段数!=5: {line}")
                        continue
                    try:
                        cid = int(parts[0])
                        cx, cy, w, h = map(float, parts[1:])
                    except ValueError:
                        issues.append(f"[{split}/{lbl}:{ln}] 非数值: {line}")
                        continue
                    if not (0.0 <= cx <= 1.0 and 0.0 <= cy <= 1.0
                            and 0.0 < w <= 1.0 and 0.0 < h <= 1.0):
                        issues.append(f"[{split}/{lbl}:{ln}] 非法坐标: {line}")
                    box_count += 1
                    stats["per_class"][split][cid] = stats["per_class"][split].get(cid, 0) + 1
        total_boxes += box_count
        stats["per_split"][split] = {"images": len(imgs), "labels": len(lbls), "boxes": box_count}

    stats["total_boxes"] = total_boxes
    stats["unique_stems"] = len(seen_stems)
    import collections
    case_splits: Dict[str, set] = collections.defaultdict(set)
    for split in splits:
        img_dir = os.path.join(img_root, split)
        for f in os.listdir(img_dir):
            stem = os.path.splitext(f)[0]
            cls_name, _, cell = stem.partition("__")
            case_id = cell.split("_")[0]
            case_splits[f"{cls_name}/{case_id}"].add(split)

    case_leaks = {k: sorted(v) for k, v in case_splits.items() if len(v) > 1}
    if case_leaks:
        sample = list(case_leaks.items())[:3]
        issues.append(
            f"病例级泄漏: {len(case_leaks)}/{len(case_splits)} 个病例跨 split，例如 {sample}"
        )
    stats["cases"] = len(case_splits)
    stats["case_leaks"] = len(case_leaks)
    stats["cells_per_case"] = round(
        sum(stats["per_split"][s]["images"] for s in splits) / max(len(case_splits), 1), 2
    )

    return {"ok": not issues, "issues": issues, "stats": stats}


# 使用示例
if __name__ == "__main__":
    dataset_root = "data/sipakmed"
    output_root = "data/sipakmed_yolo"
    config_path = "configs/sipakmed_yolo.yaml"

    # 转换（include_nucleus=True 时可额外导出胞核框）
    converter = SiPaKMeDConverter(dataset_root, output_root, seed=42)
    report = converter.convert_dataset(train_ratio=0.7, val_ratio=0.15)

    create_yolo_config(output_root, config_path,
                       include_nucleus=converter.include_nucleus)

    # 转换后立即自检
    result = validate_yolo_dataset(output_root)
    print("\n[validate] ok =", result["ok"])
    print("[validate] stats =", result["stats"])
    if result["issues"]:
        print("[validate] 共 %d 个问题:" % len(result["issues"]))
        for issue in result["issues"][:50]:
            print("  -", issue)
        raise SystemExit(1)