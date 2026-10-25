"""
模型服务接口定义,请求/响应数据模型与结果构造
"""

from pydantic import BaseModel

from utils.uncertainty import (
    METRIC_NAME,
    UNCERTAINTY_HIGH,
    UNCERTAINTY_MEDIUM,
    UNCERTAINTY_PERCENTILE,
)


class FollowUpRequest(BaseModel):
    session_id: str
    message: str


class LoginRequest(BaseModel):
    username: str
    password: str


def _agent_payload(session_id: str, result: dict) -> dict:
    """统一 Agent 响应体：把不可判读原因、风险口径一并返回给前端。"""
    risk_level = result.get("risk_level")
    cv = result.get("cv_result") or {}
    if risk_level == "unreadable":
        status = "unreadable"
    elif result.get("is_complete"):
        status = "complete"
    else:
        status = "awaiting_clarification"
    return {
        "session_id": session_id,
        "status": status,
        "risk_level": risk_level,
        "risk_scope": result.get("risk_scope"),
        "decision_reason": result.get("decision_reason"),
        "clarification_rounds": result.get("clarification_rounds", 0),
        "recommendations": result.get("recommendations"),
        "report": result.get("diagnosis_report"),
        "questions": result.get("clarification_questions"),
        "error_message": result.get("error_message"),
        # CV 指标（来自会话状态里已保存的 cv_result 字段）
        "total_cells": cv.get("total_cells"),
        "abnormal_cells": cv.get("abnormal_cells"),
        "highest_risk": cv.get("highest_risk"),
        "label_space": cv.get("label_space"),
        "multiclass_abnormal_classes": cv.get("multiclass_abnormal_classes"),
        "grading_conflict_count": cv.get("grading_conflict_count", 0),
        "review_required": cv.get("review_required", False),
        "max_uncertainty": cv.get("max_uncertainty"),
        "uncertainty_thresholds": cv.get("uncertainty_thresholds"),
        "uncertainty_metric": cv.get("uncertainty_metric"),
        "uncertainty_aggregation": cv.get("uncertainty_aggregation"),
        "cells": cv.get("cells") or [],
    }

def _session_payload(session_id: str, state: dict) -> dict:
    """会话读取响应：在 Agent 响应基础上补细胞明细与证据链信息。"""
    payload = _agent_payload(session_id, state)
    payload.update(
        {
            "references": state.get("references") or [],
            "image_available": bool(state.get("image_path")),
        }
    )
    return payload

def _error_result(detail: str) -> dict:
    """单图失败时的占位结果，字段与视觉服务正常返回完全一致。"""
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
        "error": detail,
    }