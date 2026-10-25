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

    # 2) 异常细胞（按二分类结果）
    abnormal = [c for c in cells if (c.get("binary_result") or {}).get("class") == "abnormal"]
    abnormal.sort(key=lambda c: float(c.get("final_confidence") or 0.0), reverse=True)

    # 3) 只有「当前标签空间下确属异常」的分型才进入 abnormal_types
    #    否则会出现"异常细胞分型：化生细胞 / 正常表层细胞"这类自相矛盾的表述
    abnormal_classes = set(cv_result.get("multiclass_abnormal_classes") or [])

    def _is_resolved(c) -> bool:
        cls = c.get("final_class")
        if not abnormal_classes:          # 空间未知 → 保守视为已定论
            return bool(cls)
        return cls in abnormal_classes

    resolved = [c for c in abnormal if _is_resolved(c)]
    # 二分类判异常、但分级/分型模型给出非异常类 → 结论冲突，转人工复核
    conflicted = [c for c in abnormal if not _is_resolved(c) or c.get("dual_conflict")]

    top = resolved[:3]

    # 4) 形态学统计（仅取测量可靠的细胞，避免用噪声算均值）
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

    # 5) 冲突证据：已定论异常细胞的分型不止一种
    abnormal_types = {c.get("final_class") for c in resolved}

    # 6) 不确定性分布
    us = [float(c.get("uncertainty") or 0.0) for c in cells]

    return {
        "total": total,
        "abnormal_count": len(abnormal),
        "class_distribution": dict(dist),
        "abnormal_types": sorted(abnormal_types),
        "has_conflict": len(abnormal_types) > 1,
        "conflict_count": len(conflicted),
        "conflict_ratio": (len(conflicted) / total) if total else 0.0,
        "review_required": bool(conflicted),
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