import io
import os
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
MULTI_WEIGHTS = os.getenv("MULTI_WEIGHTS", "outputs/models/best_resnet50_multiclass_model.pth")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MC_DROPOUT_SAMPLES = int(os.getenv("MC_DROPOUT_SAMPLES", "10"))
UNCERTAINTY_THRESHOLD = float(os.getenv("UNCERTAINTY_THRESHOLD", "0.15"))
ABNORMAL_CONF_THRESHOLD = float(os.getenv("ABNORMAL_CONF_THRESHOLD", "0.5"))

# 类别定义
BINARY_CLASSES = ["normal", "abnormal"]
MULTICLASS_CLASSES = ["Dyskeratotic", "Koilocytotic", "Metaplastic", "Parabasal", "SuperficialIntermediate"]

class ModelType(str, Enum):
    BINARY = "binary"
    MULTICLASS = "multiclass"

# 模型管理器
class DualModelManager:
    """统一管理模型(YOLO + 二分类 + 多分类)"""

    def __init__(self):
        self.detector: Optional[YOLO] = None
        self.binary_classifier: Optional[CervicalCancerModel] = None
        self.multi_classifier: Optional[CervicalCancerModel] = None
        self._load_all()

    def _load_all(self):
        try:
            # 加载 YOLO 检测模型
            logger.info(f"[DualModel] 从 {YOLO_WEIGHTS} 加载检测模型")
            self.detector = YOLO(YOLO_WEIGHTS)

            # 2. 二分类模型
            logger.info(f"[DualModel] 从 {BINARY_WEIGHTS} 加载二进制分类器")
            binary_state = torch.load(BINARY_WEIGHTS, map_location=DEVICE, weights_only=True)
            self.binary_classifier = self._build_classifier(binary_state, num_classes=2)
            self.binary_classifier.load_state_dict(binary_state)
            self.binary_classifier.to(DEVICE).eval()

            # 3. 多分类模型
            logger.info(f"[DualModel] 从 {MULTI_WEIGHTS} 加载多分类器")
            multi_state = torch.load(MULTI_WEIGHTS, map_location=DEVICE, weights_only=True)
            self.multi_classifier = self._build_classifier(multi_state, num_classes=len(MULTICLASS_CLASSES))
            self.multi_classifier.load_state_dict(multi_state)
            self.multi_classifier.to(DEVICE).eval()

            logger.info("三个模型全部顺利加载.")
        except Exception as e:
            logger.error(f"[Dualmodel] 模型加载失败: {e}", exc_info=True)
            raise RuntimeError(f"模型初始化失败: {e}")

    # 静态方法
    @staticmethod
    def _build_classifier(state_dict: dict, num_classes: int) -> CervicalCancerModel:
        """根据 state_dict 自动判断是否使用注意力机制，并构建 CervicalCancerModel"""
        keys = list(state_dict.keys())
        use_attention = any(k.startswith("attention.") or ".attention." in k for k in keys)
        attention_type = None
        if use_attention:
            if any("attention.ca." in k or "attention.sa." in k for k in keys):
                attention_type = "cbam"
            elif any("attention.fc." in k for k in keys):
                attention_type = "se"
        return CervicalCancerModel(
            num_classes=num_classes, pretrained=False, model_type="resnet50",
            use_attention=use_attention, attention_type=attention_type
        )

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


# def _enable_dropout_only(module: torch.nn.Module):
#     if isinstance(module, torch.nn.modules.dropout._DropoutNd):
#         module.train()
    # else:
    #     module.eval()


# def mc_dropout_predict(model: torch.nn.Module, inputs: torch.Tensor, n_samples: int = 10) -> Tuple[np.ndarray, np.ndarray]:
#     original_training = model.training
#     model.apply(_enable_dropout_only)
#     preds = []
#     with torch.no_grad():
#         for _ in range(n_samples):
#             out = model(inputs)
#             prob = torch.softmax(out, dim=1)
#             preds.append(prob.cpu().numpy())
#     preds = np.stack(preds, axis=0)
#     if not original_training:
#         model.eval()
#     return preds.mean(axis=0), preds.std(axis=0)

def _enable_dropout_only(module):
    if isinstance(module, torch.nn.modules.dropout._DropoutNd):
        module.train()

def mc_dropout_predict(model, inputs: torch.Tensor, n_samples: int = 10):
    """MC Dropout 推理: 多次前向传播获取预测分布"""
    was_training = model.training
    model.eval()
    model.apply(_enable_dropout_only)
    with torch.no_grad():
        preds = []
        for _ in range(n_samples):
            out = model(inputs)
            preds.append(torch.nn.functional.softmax(out, dim=1).cpu().numpy())
    model.train(was_training)
    preds = np.stack(preds)  # (n_samples, num_classes)
    return preds.mean(axis=0), preds.std(axis=0)

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
        "max_uncertainty": 0.0,
        "needs_clarification": False,
        "error": None
    }
    mgr = get_model_manager()
    try:
        results = mgr.detector(image, conf=conf, iou=iou)
        cells = []
        res0 = results[0]
        boxes = getattr(res0, "boxes", [])
        morph_tool = MorphologyTool()

        for box in boxes:
            xyxy = box.xyxy[0].tolist() if hasattr(box, "xyxy") else box.tolist()
            x1, y1, x2, y2 = map(int, xyxy)
            cell_img = image.crop((x1, y1, x2, y2)).convert("RGB")
            input_tensor = preprocess_image_pil(cell_img, mode="eval").unsqueeze(0).to(DEVICE)

            # ---- Step 1: 二分类筛查 ----
            bin_idx, bin_conf, bin_probs = mgr.predict_binary(input_tensor)
            is_abnormal = (bin_idx == 1) and (bin_conf >= ABNORMAL_CONF_THRESHOLD)

            # ---- Step 2: 条件触发多分类 ----
            if is_abnormal:
                multi_idx, multi_conf, multi_probs = mgr.predict_multi(input_tensor)
                final_class = MULTICLASS_CLASSES[multi_idx]
                final_conf = multi_conf
                final_probs_dict = {name: round(float(p), 4) for name, p in zip(MULTICLASS_CLASSES, multi_probs)}
                active_model_type = ModelType.MULTICLASS
                active_model = mgr.multi_classifier
            else:
                final_class = BINARY_CLASSES[bin_idx]
                final_conf = bin_conf
                final_probs_dict = {name: round(float(p), 4) for name, p in zip(BINARY_CLASSES, bin_probs)}
                active_model_type = ModelType.BINARY
                active_model = mgr.binary_classifier

            # ---- Step 3: MC Dropout (仅对活跃模型) ----
            _, std_prob = mc_dropout_predict(active_model, input_tensor, n_samples=mc_samples)
            uncertainty = float(np.max(std_prob))

            # ---- Step 4: 形态学 ----
            morph = morph_tool.run(cell_img)

            cells.append({
                "bbox": [x1, y1, x2, y2],
                "binary_result": {"class": BINARY_CLASSES[bin_idx], "confidence": round(bin_conf, 4)},
                "final_class": final_class,
                "final_confidence": round(final_conf, 4),
                "model_used": active_model_type.value,
                "uncertainty": round(uncertainty, 4),
                "probabilities": final_probs_dict,
                "morphology": morph
            })

        total_cells = len(cells)
        abnormal_cells = sum(1 for c in cells if c["binary_result"]["class"] == "abnormal")
        highest_risk = max(cells, key=lambda x: x["final_confidence"])["final_class"] if cells else "none"
        max_uncertainty = max((c["uncertainty"] for c in cells), default=0.0)

        return {
            "total_cells": total_cells,
            "abnormal_cells": abnormal_cells,
            "cells": cells,
            "highest_risk": highest_risk,
            "max_uncertainty": round(max_uncertainty, 4),
            "needs_clarification": max_uncertainty > UNCERTAINTY_THRESHOLD,
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
            "highest_risk": "unknown",
            "max_uncertainty": 0.0,
            "needs_clarification": False,
            "error": str(e)
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

