from __future__ import annotations

from typing import Any, Dict

from utils.case_evidence import aggregate_case_evidence
from utils.uncertainty import UNCERTAINTY_MEDIUM


def decision_router(state: Dict[str, Any]) -> str:
    """根据不确定性评估结果路由。"""
    return state.get("action", "direct_report")


def explainable_router(state: Dict[str, Any]) -> str:
    """判断是否需要生成可解释性报告（需要时才付一次 LLM 成本）。"""
    if state.get("risk_level") in ("high", "emergency", "pending", "unreadable"):
        return "explainable_report"

    # 冲突证据来自聚合结果，而非 cv_result 顶层键
    ev = aggregate_case_evidence(state.get("cv_result") or {})
    if float(state.get("max_uncertainty") or 0.0) > UNCERTAINTY_MEDIUM or ev.get("has_conflict"):
        return "explainable_report"
    return "skip_explainable"