"""
SiPaKMeD -> YOLO 转换器回归测试。

覆盖两处历史致命问题：
  1. 同名文件跨类别覆盖导致数据丢失 -> 输出文件名必须带类别前缀且全局唯一；
  2. split 不可复现 -> 相同 seed 必须得到完全相同的划分。
"""

import os

import pytest
from PIL import Image

from utils.label_mapping import SIPAKMED_CLASSES
from utils.sipakmed_converter import (
    SiPaKMeDConverter,
    validate_yolo_dataset,
    create_yolo_config,
)

# 使用真正的 SiPaKMeD 目录名，以命中转换器的类别解析逻辑
CLASS_A = "im_Dyskeratotic"        # class id 0
CLASS_B = "im_Koilocytotic"        # class id 1
# 两个类别故意使用完全相同的文件名（旧实现会互相覆盖）
SAME_NAMES = ["001_01.bmp", "002_01.bmp", "003_01.bmp", "004_01.bmp"]

CYT = "8,8\n24,8\n24,24\n8,24\n"     # 胞浆轮廓 -> 细胞框
NUC = "12,12\n18,12\n18,18\n12,18\n"  # 胞核轮廓


def _make_source(root: str):
    """构造一个最小可用的 SiPaKMeD 源数据集。"""
    for class_dir in (CLASS_A, CLASS_B):
        cropped = os.path.join(root, class_dir, "CROPPED")
        os.makedirs(cropped, exist_ok=True)
        for name in SAME_NAMES:
            Image.new("RGB", (64, 64), "white").save(os.path.join(cropped, name))
            base = os.path.splitext(name)[0]
            with open(os.path.join(cropped, f"{base}_cyt.dat"), "w") as f:
                f.write(CYT)
            with open(os.path.join(cropped, f"{base}_nuc.dat"), "w") as f:
                f.write(NUC)


def _list_images(output_root):
    names = []
    for split in ("train", "val", "test"):
        d = os.path.join(output_root, "images", split)
        names.extend(sorted(os.listdir(d)))
    return names


def test_no_data_loss_and_unique_filenames(tmp_path):
    src = str(tmp_path / "sipakmed")
    out = str(tmp_path / "yolo")
    _make_source(src)

    report = SiPaKMeDConverter(src, out, seed=42).convert_dataset(0.5, 0.25)

    expected = len(SAME_NAMES) * 2
    assert sum(report["total"].values()) == expected, "存在图像被覆盖丢失"

    names = _list_images(out)
    assert len(names) == expected
    assert len(set(names)) == expected, "输出文件名必须全局唯一"

    # 同名源文件在两个类别下都必须存活（旧实现会被覆盖）
    assert f"{CLASS_A}__001_01.bmp" in names
    assert f"{CLASS_B}__001_01.bmp" in names


def test_validate_reports_clean_dataset(tmp_path):
    src = str(tmp_path / "sipakmed")
    out = str(tmp_path / "yolo")
    _make_source(src)

    SiPaKMeDConverter(src, out, seed=42).convert_dataset(0.5, 0.25)
    result = validate_yolo_dataset(out)

    assert result["ok"], result["issues"]
    assert result["stats"]["unique_stems"] == len(SAME_NAMES) * 2


def test_split_is_deterministic_for_same_seed(tmp_path):
    src = str(tmp_path / "sipakmed")
    out1 = str(tmp_path / "yolo1")
    out2 = str(tmp_path / "yolo2")
    _make_source(src)

    SiPaKMeDConverter(src, out1, seed=123).convert_dataset(0.5, 0.25)
    SiPaKMeDConverter(src, out2, seed=123).convert_dataset(0.5, 0.25)

    def snapshot(root):
        return {
            split: sorted(os.listdir(os.path.join(root, "images", split)))
            for split in ("train", "val", "test")
        }

    assert snapshot(out1) == snapshot(out2), "相同 seed 必须得到相同的划分"


def test_label_uses_cell_class_and_one_box_per_cell(tmp_path):
    src = str(tmp_path / "sipakmed")
    out = str(tmp_path / "yolo")
    _make_source(src)

    SiPaKMeDConverter(src, out, seed=42).convert_dataset(0.5, 0.25)

    # 找到 CLASS_B (Koilocytotic -> id 1) 的任意一个标签
    found = False
    for split in ("train", "val", "test"):
        lbl_dir = os.path.join(out, "labels", split)
        for fn in os.listdir(lbl_dir):
            if not fn.startswith(CLASS_B + "__"):
                continue
            with open(os.path.join(lbl_dir, fn)) as f:
                lines = [ln.split() for ln in f if ln.strip()]
            # 仅胞浆轮廓 -> 恰好 1 个细胞框，且类别正确
            assert len(lines) == 1
            assert int(lines[0][0]) == SIPAKMED_CLASSES.index("Koilocytotic")
            found = True
            break
        if found:
            break
    assert found


def test_include_nucleus_adds_separate_class(tmp_path):
    src = str(tmp_path / "sipakmed")
    out = str(tmp_path / "yolo")
    _make_source(src)

    SiPaKMeDConverter(src, out, seed=42, include_nucleus=True).convert_dataset(0.5, 0.25)

    lbl_dir = os.path.join(out, "labels", "train")
    fn = os.listdir(lbl_dir)[0]
    with open(os.path.join(lbl_dir, fn)) as f:
        class_ids = {int(ln.split()[0]) for ln in f if ln.strip()}

    # 细胞类别 + 独立 nucleus 类别（id == len(SIPAKMED_CLASSES)）
    assert len(SIPAKMED_CLASSES) in class_ids
    assert class_ids & set(range(len(SIPAKMED_CLASSES)))


def test_config_matches_class_definitions(tmp_path):
    out = str(tmp_path / "yolo")
    cfg = str(tmp_path / "data.yaml")
    create_yolo_config(out, cfg, include_nucleus=False)

    with open(cfg) as f:
        content = f.read()
    for name in SIPAKMED_CLASSES:
        assert name in content
    assert "nc: 5" in content
