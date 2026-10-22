import io
import os
import json
import glob
import asyncio
import zipfile
import logging
from typing import List, Dict, Any, Tuple, Optional
from enum import Enum

from PIL import Image
import numpy as np
import torch
import torchvision.transforms as transforms

# 第三方/项目模块
from ultralytics import YOLO
from models.cer_cancer_model import CervicalCancerModel
from tools.morphology_tool import MorphologyTool
from utils.label_mapping import (
    BINARY_CLASSES,
    BINARY_ABNORMAL_IDX,
    is_abnormal_binary,
    MULTICLASS_CLASSES,
    NUCLEUS_CLASS_NAME,
)
from utils.uncertainty import (
    UNCERTAINTY_HIGH,
    UNCERTAINTY_MEDIUM,
    UNCERTAINTY_PERCENTILE,
    METRIC_NAME,
    normalized_entropy,
    risk_level_from_ratio,
    risk_scope_for,
)

"""
封装说明:
- 提供同步函数 `analyze_image_from_path(file_path)` 供内部调用
- 提供 `analyze_pil_image(image)` 供 FastAPI 主服务 agent_api.py 调用
- 提供 MC Dropout 不确定性估计
"""

# Logger 配置
logger = logging.getLogger("vision_service")

#配置项(支持环境变量覆盖)
YOLO_WEIGHTS = os.getenv("YOLO_WEIGHTS", "runs/detect/cervical_cell_exp4/weights/best.pt")
BINARY_WEIGHTS = os.getenv("BINARY_WEIGHTS", "outputs/models/best_resnet50_binary_model.pth")
MULTI_WEIGHTS = os.getenv("MULTI_WEIGHTS", "outputs/models/best_resnet50_multiclass_sipakmed_model.pth")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MC_DROPOUT_SAMPLES = int(os.getenv("MC_DROPOUT_SAMPLES", "10"))
ABNORMAL_CONF_THRESHOLD = float(os.getenv("ABNORMAL_CONF_THRESHOLD", "0.5"))

# 类别定义
assert BINARY_CLASSES[BINARY_ABNORMAL_IDX] == "abnormal", "二分类索引顺序被破坏！"
assert MULTICLASS_CLASSES[0] == "Dyskeratotic", "多分类索引顺序被破坏！"

def _load_classes_sidecar(weights_path: str, fallback, expected_n: int):
    """优先读取与权重同名的 .classes.json；校验长度后返回，否则回退 canonical。"""
    sidecar = os.path.splitext(weights_path)[0] + ".classes.json"
    classes = None
    if os.path.isfile(sidecar):
        try:
            with open(sidecar, "r", encoding="utf-8") as f:
                data = json.load(f)
            classes = data.get("classes") if isinstance(data, dict) else data
        except Exception as e:
            logger.warning(f"读取类别表 sidecar 失败({sidecar}): {e}")
    if isinstance(classes, list) and classes:
        if len(classes) != expected_n:
            raise RuntimeError(
                f"{os.path.basename(sidecar)} 类别数({len(classes)}) 与权重输出维度"
                f"({expected_n}) 不一致，拒绝加载以免标签错配。"
            )
        if list(classes) != list(fallback):
            logger.warning(f"类别表与 canonical 不一致，以 sidecar 为准: {classes}")
        return [str(c) for c in classes]
    return list(fallback)

class ModelType(str, Enum):
    BINARY = "binary"
    MULTICLASS = "multiclass"

def _require_file(path: str, what: str):
    """权重缺失时给出可操作的错误，而不是裸 FileNotFoundError。"""
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"{what} 权重不存在: {path}\n"
            f"  请先训练该模型，或用环境变量覆盖路径"
            f"（YOLO_WEIGHTS / BINARY_WEIGHTS / MULTI_WEIGHTS）。"
        )

# 模型管理器
class DualModelManager:
    """统一管理模型(YOLO + 二分类 + 多分类)"""

    def __init__(self):
        self.detector: Optional[YOLO] = None
        self.binary_classifier: Optional[CervicalCancerModel] = None
        self.multi_classifier: Optional[CervicalCancerModel] = None
        self.binary_classes: List[str] = list(BINARY_CLASSES)
        self.multi_classes: List[str] = list(MULTICLASS_CLASSES)
        self._load_all()
    
    def _load_all(self):
        try:
            # 加载 YOLO 检测模型
            _require_file(YOLO_WEIGHTS, "YOLO 检测")
            logger.info(f"[DualModel] 从 {YOLO_WEIGHTS} 加载检测模型")
            self.detector = YOLO(YOLO_WEIGHTS)

            # 2. 二分类模型
            _require_file(BINARY_WEIGHTS, "二分类")
            logger.info(f"[DualModel] 从 {BINARY_WEIGHTS} 加载二进制分类器")
            binary_state = torch.load(BINARY_WEIGHTS, map_location=DEVICE, weights_only=True)
            n_bin = self._fc_out_dim(binary_state)
            self.binary_classes = _load_classes_sidecar(BINARY_WEIGHTS, BINARY_CLASSES, n_bin)
            self.binary_classifier = self._build_classifier(binary_state, num_classes=len(self.binary_classes))
            self.binary_classifier.load_state_dict(binary_state)
            self.binary_classifier.to(DEVICE).eval()

            # 3. 多分类模型
            _require_file(MULTI_WEIGHTS, "多分类")
            logger.info(f"[DualModel] 从 {MULTI_WEIGHTS} 加载多分类器")
            multi_state = torch.load(MULTI_WEIGHTS, map_location=DEVICE, weights_only=True)
            n_multi = self._fc_out_dim(multi_state)
            self.multi_classes = _load_classes_sidecar(MULTI_WEIGHTS, MULTICLASS_CLASSES, n_multi)
            self.multi_classifier = self._build_classifier(multi_state, num_classes=n_multi)
            self.multi_classifier.load_state_dict(multi_state)
            self.multi_classifier.to(DEVICE).eval()

            logger.info("三个模型全部顺利加载.")
        except Exception as e:
            logger.error(f"[Dualmodel] 模型加载失败: {e}", exc_info=True)
            raise RuntimeError(f"模型初始化失败: {e}")

    @staticmethod
    def _build_classifier(state_dict: dict, num_classes: int) -> CervicalCancerModel:
        """根据state_dict自动判断backbone与注意力机制,并构建模型"""
        keys = list(state_dict.keys())
        use_attention = any(k.startswith("attention.") or ".attention." in k for k in keys)
        attention_type = None
        if use_attention:
            if any("attention.ca." in k or "attention.sa." in k for k in keys):
                attention_type = "cbam"
            elif any("attention.fc." in k for k in keys):
                attention_type = "se"
        return CervicalCancerModel(
            num_classes=num_classes, pretrained=False,
            model_type=CervicalCancerModel.detect_model_type(state_dict),
            use_attention=use_attention, attention_type=attention_type,
        )

    @staticmethod
    def _fc_out_dim(state_dict: dict) -> int:
        """从 state_dict 的 fc.weight 推断类别数(权重真实输出维度)"""
        fc = next((v for k, v in state_dict.items() if k.endswith("fc.weight")), None)
        if fc is None:
            raise RuntimeError("state_dict 中找不到 fc.weight,无法确定类别数。")
        return int(fc.shape[0])

    # 二分类预测
    def predict_binary(self, tensor: torch.Tensor) -> Tuple[int, float, np.ndarray]:
        """返回 (pred_idx, confidence, probs_array)"""
        with torch.no_grad():
            logits = self.binary_classifier(tensor)
            probs = torch.softmax(logits, dim=1)
            pred_idx = int(torch.argmax(probs, dim=1).item())
            conf = float(probs[0][pred_idx].item())
        return pred_idx, conf, probs[0].cpu().numpy()

    #多分类预测
    def predict_multi(self, tensor: torch.Tensor) -> Tuple[int, float, np.ndarray]:
        """返回 (pred_idx, confidence, probs_array)"""
        with torch.no_grad():
            logits = self.multi_classifier(tensor)
            probs = torch.softmax(logits, dim=1)
            pred_idx = int(torch.argmax(probs, dim=1).item())
            conf = float(probs[0][pred_idx].item())
        return pred_idx, conf, probs[0].cpu().numpy()


#全局模型管理器实例
model_manager: Optional[DualModelManager] = None

def get_model_manager() -> DualModelManager:
    global model_manager
    if model_manager is None:
        model_manager = DualModelManager()
    return model_manager

# === 预处理 & MC Dropout ===
def preprocess_image_pil(image: Image.Image, mode: str = "eval") -> torch.Tensor:
    if mode == "train":
        transform = transforms.Compose([
            transforms.Resize((268, 268)), transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(p=0.5), transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(20),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize((224, 224)), transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    return transform(image)

def mc_dropout_predict(model, inputs: torch.Tensor, n_samples: int = 10):
    """MC Dropout 推理: 多次前向传播获取预测分布"""
    with torch.no_grad():
        features = model.forward_features(inputs)
        preds = torch.stack([
            torch.softmax(model.forward_head(features, dropout_active=True), dim=1)
            for _ in range(n_samples)
        ])  # (n_samples, B, K)
    return preds.mean(dim=0).cpu().numpy(), preds.std(dim=0).cpu().numpy()

NON_CELL_CLASS_NAMES = {NUCLEUS_CLASS_NAME.lower()}

def _is_cell_box(box, detector) -> bool:
    """判断检测框是否为细胞框；无法判定时保守返回 True。"""
    try:
        cls_id = int(box.cls.item()) if hasattr(box.cls, "item") else int(box.cls)
    except Exception:
        return True

    names = getattr(detector, "names", None)
    if isinstance(names, dict):
        cls_name = names.get(cls_id)
    elif isinstance(names, (list, tuple)) and 0 <= cls_id < len(names):
        cls_name = names[cls_id]
    else:
        cls_name = None

    if cls_name is not None and str(cls_name).lower() in NON_CELL_CLASS_NAMES:
        return False
    return True

# ---- 检测框去重参数（可用环境变量调整）----
DEDUP_IOA = float(os.getenv("DEDUP_IOA", "0.85"))   # 小框被大框包含的比例
DEDUP_IOU = float(os.getenv("DEDUP_IOU", "0.60"))   # 同位重复的重叠比例
MIN_BOX_SIDE = int(os.getenv("MIN_BOX_SIDE", "8"))  # 退化框最小边长


def _box_area(b):
    return max(0, b[2] - b[0]) * max(0, b[3] - b[1])


def _intersection(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    return max(0, x2 - x1) * max(0, y2 - y1)


def dedup_boxes(boxes: List[Tuple[List[int], float]],
                ioa_thresh: float = DEDUP_IOA,
                iou_thresh: float = DEDUP_IOU) -> List[Tuple[List[int], float]]:
    """去除"同一细胞"的重复检测框。

    SIPaKMeD 单细胞裁剪图经检测后实测 1.78 框/图(应 ≈1.0)，两类重复：
      1. 包含关系：胞核小框落在整细胞框内 → 保留较大的整细胞框
         （分类器训练用整细胞裁剪图，胞核小图属分布外输入）
      2. 同位重复：两个几乎相同的整图框 → 保留面积大者（先入）

    实现：按面积降序，逐个与已保留框比较；若被已保留框高度覆盖
    (IoA ≥ ioa_thresh)或高度重叠(IoU ≥ iou_thresh)则丢弃。

    Args:
        boxes: [(xyxy, conf), ...]
    Returns:
        去重后的 [(xyxy, conf), ...]（面积降序）
    """
    if len(boxes) <= 1:
        return list(boxes)

    order = sorted(boxes, key=lambda t: _box_area(t[0]), reverse=True)
    kept: List[Tuple[List[int], float]] = []

    for bbox, conf in order:
        area = _box_area(bbox)
        if area <= 0:
            continue
        dup = False
        for kbox, _ in kept:
            inter = _intersection(bbox, kbox)
            if inter <= 0:
                continue
            karea = _box_area(kbox)
            ioa = inter / area
            iou = inter / (area + karea - inter)
            area_ratio = area / karea if karea > 0 else 1.0
            if ioa >= ioa_thresh and area_ratio <= 0.5:
                dup = True
                break
            if iou >= iou_thresh and area_ratio >= 0.8:
                dup = True
                break
        if not dup:
            kept.append((bbox, conf))
    return kept

# === 核心双模型分析函数 ===
def _analyze_pil_image_sync(
        image: Image.Image,
        conf: float = 0.25,
        iou: float = 0.45,
        mc_samples: int = MC_DROPOUT_SAMPLES
) -> Dict[str, Any]:
    """
        双模型级联推理:
        1. YOLO 检测所有细胞
        2. 二分类模型筛查每个细胞
        3. 仅对 abnormal 细胞调用多分类模型进行细粒度分型
        4. MC Dropout 对最终使用的模型做不确定性估计
    """
    empty_result = {
        "total_cells": 0,
        "abnormal_cells": 0,
        "cells": [],
        "highest_risk": "none",
        "interpretable": False,
        "unreadable_reason": (
            "未检测到可判读的宫颈细胞，无法给出风险分级。"
            "请确认图像为细胞学视野、清晰度足够，且未为大面积空白或非细胞内容。"
        ),
        "risk_level": None,
        "risk_scope": None,
        "uncertainty_thresholds": {"high": UNCERTAINTY_HIGH, "medium": UNCERTAINTY_MEDIUM},
        "max_uncertainty": 0.0,
        "uncertainty_metric": METRIC_NAME,
        "uncertainty_aggregation": f"p{UNCERTAINTY_PERCENTILE}",
        "needs_clarification": False,
        "detected_boxes_raw": 0,
        "dedup_removed": 0,
        "error": None
    }
    mgr = get_model_manager()
    try:
        results = mgr.detector(image, conf=conf, iou=iou)
        res0 = results[0]
        boxes = getattr(res0, "boxes", [])

        # ---- Step 0: 收集 + 去重（避免同一细胞被重复计数）----
        raw: List[Tuple[List[int], float]] = []
        for box in boxes:
            if not _is_cell_box(box, mgr.detector):
                continue
            xyxy = box.xyxy[0].tolist() if hasattr(box, "xyxy") else box.tolist()
            x1, y1, x2, y2 = map(int, xyxy)
            if (x2 - x1) < MIN_BOX_SIDE or (y2 - y1) < MIN_BOX_SIDE:
                continue                                   # 过滤退化框
            conf_b = float(box.conf.item()) if hasattr(box, "conf") else 1.0
            raw.append(([x1, y1, x2, y2], conf_b))

        kept = dedup_boxes(raw)
        n_raw, n_kept = len(raw), len(kept)

        cells: List[Dict[str, Any]] = []
        if not kept:
            empty_result["detected_boxes_raw"] = n_raw
            empty_result["dedup_removed"] = n_raw - n_kept
            return empty_result

        # ---- Step 1: 批量预处理（一次 stack，后续复用）----
        crops = [image.crop(tuple(b)).convert("RGB") for b, _ in kept]
        batch = torch.stack([preprocess_image_pil(c, mode="eval") for c in crops]).to(DEVICE)

        # ---- Step 2: 二分类（batch 一次前向）----
        with torch.no_grad():
            bin_probs_all = torch.softmax(mgr.binary_classifier(batch), dim=1).cpu().numpy()
        bin_idx_all = bin_probs_all.argmax(axis=1)

        abnormal_flags = [
            is_abnormal_binary(int(i)) and float(p[i]) >= ABNORMAL_CONF_THRESHOLD
            for i, p in zip(bin_idx_all, bin_probs_all)
        ]

        # ---- Step 3: 多分类（仅异常子集，一次前向）----
        multi_probs_map: Dict[int, np.ndarray] = {}
        abnormal_idx = [i for i, f in enumerate(abnormal_flags) if f]
        if abnormal_idx:
            with torch.no_grad():
                out = torch.softmax(mgr.multi_classifier(batch[abnormal_idx]), dim=1).cpu().numpy()
            multi_probs_map = {i: out[pos] for pos, i in enumerate(abnormal_idx)}

        # ---- Step 4: MC Dropout（按活跃模型分组，各一次批量采样）----
        morph_tool = MorphologyTool()

        for model, mtype, want_abnormal in (
            (mgr.binary_classifier, ModelType.BINARY, False),
            (mgr.multi_classifier, ModelType.MULTICLASS, True),
        ):
            group = [i for i, f in enumerate(abnormal_flags) if f == want_abnormal]
            if not group:
                continue
            mean_probs, std_probs = mc_dropout_predict(
                model, batch[group], n_samples=mc_samples)     # 返回 (len(group), K)

            for pos, i in enumerate(group):
                bbox, _ = kept[i]
                bin_idx = int(bin_idx_all[i])
                bin_conf = float(bin_probs_all[i][bin_idx])

                if want_abnormal:
                    probs = multi_probs_map[i]
                    cls_idx = int(probs.argmax())
                    final_class = mgr.multi_classes[cls_idx]
                    final_conf = float(probs[cls_idx])
                    prob_dict = {n: round(float(p), 4)
                                 for n, p in zip(mgr.multi_classes, probs)}
                else:
                    final_class = mgr.binary_classes[bin_idx]
                    final_conf = bin_conf
                    prob_dict = {n: round(float(p), 4)
                                 for n, p in zip(mgr.binary_classes, bin_probs_all[i])}

                cells.append({
                    "bbox": bbox,
                    "binary_result": {"class": mgr.binary_classes[bin_idx],
                                      "confidence": round(bin_conf, 4)},
                    "final_class": final_class,
                    "final_confidence": round(final_conf, 4),
                    "model_used": mtype.value,
                    "uncertainty": round(normalized_entropy(mean_probs[pos]), 4),
                    "uncertainty_metric": METRIC_NAME,
                    "uncertainty_std": round(float(np.max(std_probs[pos])), 4),
                    "probabilities": prob_dict,
                    "morphology": morph_tool.run(crops[i]),   # CPU，无法批处理
                })

        cells.sort(key=lambda c: (c["bbox"][1], c["bbox"][0]))   # 输出顺序稳定

        def _cell_risk_key(c):
            """排序键：异常优先；同为异常时取置信度更高者。"""
            is_abn = c["binary_result"]["class"] == BINARY_CLASSES[BINARY_ABNORMAL_IDX]
            return (1 if is_abn else 0, float(c.get("final_confidence") or 0.0))
        
        total_cells = len(cells)
        abnormal_cells = sum(1 for c in cells if c["binary_result"]["class"] == BINARY_CLASSES[BINARY_ABNORMAL_IDX])
        highest_risk = max(cells, key=_cell_risk_key)["final_class"] if cells else "none"
        per_cell_uncertainty = [c["uncertainty"] for c in cells]
        if per_cell_uncertainty:
            image_uncertainty = float(np.percentile(per_cell_uncertainty, UNCERTAINTY_PERCENTILE))
        else:
            image_uncertainty = 0.0
        ratio = abnormal_cells / total_cells if total_cells else 0.0
        coarse_risk = risk_level_from_ratio(ratio)

        return {
            "total_cells": total_cells,
            "abnormal_cells": abnormal_cells,
            "detected_boxes_raw": n_raw,
            "dedup_removed": n_raw - n_kept,
            "cells": cells,
            "highest_risk": highest_risk,
            "interpretable": True,
            "unreadable_reason": None,
            "risk_level": coarse_risk,
            "risk_scope": risk_scope_for(total_cells), 
            "max_uncertainty": round(image_uncertainty, 4),
            "uncertainty_metric": METRIC_NAME,
            "uncertainty_aggregation": f"p{UNCERTAINTY_PERCENTILE}",
            "uncertainty_thresholds": {
                "high": UNCERTAINTY_HIGH,
                "medium": UNCERTAINTY_MEDIUM,
            },
            "needs_clarification": image_uncertainty > UNCERTAINTY_HIGH,
            "error": None
        }

    except Exception as e:
        logger.error(f"模型分析失败: {e}", exc_info=True)
        empty_result["error"] = str(e)
        return empty_result

async def analyze_pil_image(
    image: Image.Image,
    conf: float = 0.25,
    iou: float = 0.45,
    mc_samples: int = MC_DROPOUT_SAMPLES
) -> Dict[str, Any]:
    """异步入口：将同步GPU推理丢到线程池，避免阻塞事件循环"""
    return await asyncio.to_thread(_analyze_pil_image_sync, image, conf, iou, mc_samples)

# === 内部调用入口 ===
async def analyze_image_from_path(file_path: str, **kwargs) -> Dict[str, Any]:
    """
       异步包装器：将耗时的同步图像分析任务放到线程池中执行。
    """
    try:
        with Image.open(file_path) as img:
            img = img.convert("RGB")
        return await analyze_pil_image(img, **kwargs)

    except Exception as e:
        logger.error(f"未能打开图片 {file_path}： {e}")
        return {
            "total_cells": 0,
            "abnormal_cells": 0,
            "cells": [],
            "highest_risk": "none",
            "interpretable": False,
            "unreadable_reason": None,
            "risk_level": None,
            "risk_scope": None,
            "max_uncertainty": 0.0,
            "uncertainty_metric": METRIC_NAME,
            "uncertainty_aggregation": f"p{UNCERTAINTY_PERCENTILE}",
            "uncertainty_thresholds": {"high": UNCERTAINTY_HIGH, "medium": UNCERTAINTY_MEDIUM},
            "needs_clarification": False,
            "error": str(e),
        }

def analyze_case_folder(folder_path: str, pattern: str = "*.jpg", **kwargs) -> Dict[str, Dict]:
    image_paths = sorted(glob.glob(os.path.join(folder_path, pattern)))
    if not image_paths:
        logger.warning(f"{folder_path}中没有带有{pattern}的图片")
        return {}
    results = {}
    for path in image_paths:
        filename = os.path.basename(path)
        logger.info(f"Analyzing {filename} ...")
        results[filename] = _analyze_pil_image_sync(Image.open(path).convert("RGB"), **kwargs)
    return results
