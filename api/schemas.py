"""
模型服务接口定义,请求/响应数据模型与结果构造
"""

from datetime import datetime
from typing import Literal, Optional

from pydantic import BaseModel, Field

from utils.uncertainty import (
    METRIC_NAME,
    UNCERTAINTY_HIGH,
    UNCERTAINTY_MEDIUM,
    UNCERTAINTY_PERCENTILE,
)

# 用户
MIN_PASSWORD_LEN = 8
_USERNAME_RE = r"^[A-Za-z0-9_.-]{3,64}$"
AssignableRole = Literal["doctor", "admin"]

class FollowUpRequest(BaseModel):
    session_id: str
    message: str


class LoginRequest(BaseModel):
    username: str
    password: str

class UserCreate(BaseModel):
    username: str = Field(pattern=_USERNAME_RE)
    password: str = Field(min_length=MIN_PASSWORD_LEN, max_length=128)
    name: Optional[str] = Field(default=None, max_length=64)
    department: Optional[str] = Field(default=None, max_length=64)
    hospital: Optional[str] = Field(default=None, max_length=128)
    role: AssignableRole = "doctor"


class UserUpdate(BaseModel):
    name: Optional[str] = Field(default=None, max_length=64)
    department: Optional[str] = Field(default=None, max_length=64)
    hospital: Optional[str] = Field(default=None, max_length=128)
    role: Optional[AssignableRole] = None
    is_active: Optional[bool] = None


class PasswordResetRequest(BaseModel):
    password: str = Field(min_length=MIN_PASSWORD_LEN, max_length=128)


class SelfPasswordChangeRequest(BaseModel):
    old_password: str
    new_password: str = Field(min_length=MIN_PASSWORD_LEN, max_length=128)


class UserPublic(BaseModel):
    """响应模型：显式列字段，永不含 password_hash。"""
    id: int
    username: str
    name: Optional[str] = None
    department: Optional[str] = None
    hospital: Optional[str] = None
    role: str
    is_active: bool
    created_at: Optional[datetime] = None
    last_login_at: Optional[datetime] = None
    failed_attempts: int = 0
    locked_until: Optional[datetime] = None

class TaskAccepted(BaseModel):
    """上传接口 202 响应：任务已入队，结果需轮询 /api/agent/task/{session_id}"""
    session_id: str
    status: Literal["queued"] = "queued"
    queue: str
    poll_interval_ms: int = 1500

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
        "dual_conflict_count": cv.get("dual_conflict_count", 0),
        "conflict_count": cv.get("conflict_count", 0),
        "dual_mode": cv.get("dual_mode", False),
        "primary_label_space": cv.get("primary_label_space"),
        "aux_label_space": cv.get("aux_label_space"),
        "review_required": cv.get("review_required", False),
        "max_uncertainty": cv.get("max_uncertainty"),
        "uncertainty_thresholds": cv.get("uncertainty_thresholds"),
        "uncertainty_metric": cv.get("uncertainty_metric"),
        "uncertainty_aggregation": cv.get("uncertainty_aggregation"),
        "cells": cv.get("cells") or [],
        "gradcam_path": result.get("gradcam_path"),
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
    from services.vision_service import (  # 延迟导入：避免 api.schemas 拉入 torch
        AUX_SPACE, DUAL_MODE, MULTI_LABEL_SPACE, PRIMARY_SPACE,
        abnormal_classes_for,
    )
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
        # --- 标签空间与冲突（与正常返回对齐）---
        "label_space": MULTI_LABEL_SPACE,
        "multiclass_abnormal_classes": sorted(abnormal_classes_for(PRIMARY_SPACE)),
        "grading_conflict_count": 0,
        "dual_conflict_count": 0,
        "conflict_count": 0,
        "review_required": False,
        "dual_mode": DUAL_MODE,
        "primary_label_space": PRIMARY_SPACE,
        "aux_label_space": AUX_SPACE,
        # --- 不确定性 ---
        "uncertainty_metric": METRIC_NAME,
        "uncertainty_aggregation": f"p{UNCERTAINTY_PERCENTILE}",
        "uncertainty_thresholds": {"high": UNCERTAINTY_HIGH, "medium": UNCERTAINTY_MEDIUM},
        "needs_clarification": False,
        "detected_boxes_raw": 0,
        "dedup_removed": 0,
        "error": detail,
    }