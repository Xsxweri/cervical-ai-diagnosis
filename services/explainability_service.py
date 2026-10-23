"""
Grad-CAM 生成器
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


def grad_cam_enabled() -> bool:
    """每次调用实时读取环境变量，便于测试中 monkeypatch。"""
    return os.getenv("ENABLE_GRADCAM", "false").strip().lower() in {"1", "true", "yes", "on"}


def compute_case_gradcam(
    image_path: Optional[str],
    cv_result: Optional[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    """为置信度最高的异常细胞生成 Grad-CAM。任何失败返回 None，不阻断报告生成。"""
    try:
        from PIL import Image

        from services.vision_service import ModelType, get_model_manager
        from tools.gradcam_tool import GradCAMTool

        cells = (cv_result or {}).get("cells") or []
        if not image_path or not cells:
            return None

        # 优先选异常细胞中置信度最高者，否则退回全体最高置信度细胞
        abnormal = [c for c in cells if (c.get("binary_result") or {}).get("class") == "abnormal"]
        target = max(abnormal or cells, key=lambda c: c.get("final_confidence", 0.0))
        bbox = target.get("bbox")
        if not bbox:
            return None

        with Image.open(image_path) as img:
            cell_img = img.convert("RGB").crop(tuple(bbox))

        mgr = get_model_manager()
        model = (
            mgr.multi_classifier
            if target.get("model_used") == ModelType.MULTICLASS.value
            else mgr.binary_classifier
        )
        return GradCAMTool(model).run(cell_img)
    except Exception as e:
        logger.warning("Grad-CAM 生成失败: %s", e)
        return None