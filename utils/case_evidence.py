"""
病例级证据聚合：把多细胞 CV 结果压成可写进报告的结构化摘要。
"""

from __future__ import annotations

from collections import Counter
from typing import Optional


def _stat(xs):
    if not xs:
        return None
    xs = sorted(xs)
    n = len(xs)
    return {
        "n": n,
        "mean": sum(xs) / n,
        "median": xs[n // 2],
        "min": xs[0],
        "max": xs[-1],
    }


def aggregate_case_evidence(cv_result: Optional[dict]) -> dict:
    """聚合 Case 级别信息。

    :param cv_result: 视觉服务输出结果
    """
    cells = (cv_result or {}).get("cells") or []
    total = len(cells)
    if total == 0:
        return {"total": 0}

    # 1) 类别分布（按 final_class）
    dist = Counter(c.get("final_class") for c in cells)

    # 2) 异常细胞的置信度 Top-N
    abnormal = [c for c in cells if (c.get("binary_result") or {}).get("class") == "abnormal"]
    abnormal.sort(key=lambda c: float(c.get("final_confidence") or 0.0), reverse=True)
    top = abnormal[:3]

    # 3) 形态学统计（仅取测量可靠的细胞，避免用噪声算均值）
    nc, circ = [], []
    unreliable = 0
    for c in cells:
        m = c.get("morphology") or {}
        if m.get("measurement_reliable"):
            if m.get("nuclear_cytoplasmic_ratio"):
                nc.append(float(m["nuclear_cytoplasmic_ratio"]))
            if m.get("circularity"):
                circ.append(float(m["circularity"]))
        else:
            unreliable += 1

    # 4) 冲突证据：异常细胞分型不止一种
    abnormal_types = {c.get("final_class") for c in abnormal}

    # 5) 不确定性分布
    us = [float(c.get("uncertainty") or 0.0) for c in cells]

    return {
        "total": total,
        "abnormal_count": len(abnormal),
        "class_distribution": dict(dist),
        "abnormal_types": sorted(abnormal_types),
        "has_conflict": len(abnormal_types) > 1,
        "top_abnormal": [
            {
                "class": c.get("final_class"),
                "confidence": float(c.get("final_confidence") or 0.0),
                "uncertainty": float(c.get("uncertainty") or 0.0),
            }
            for c in top
        ],
        "nc_ratio": _stat(nc),
        "circularity": _stat(circ),
        "morphology_unreliable": unreliable,
        "uncertainty_max": max(us) if us else 0.0,
        "uncertainty_median": sorted(us)[len(us) // 2] if us else 0.0,
    }