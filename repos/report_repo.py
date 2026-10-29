"""
诊断报告持久化

隔离理由：避免 DB 恢复时 Redis 键过期，导致归属关系丢失。
"""

from __future__ import annotations
from typing import List, Optional

from core.db import acquire

# 只保留回看所需字段。
# probabilities 只在「当时那次决策」中有意义（回看时医生关心的是
# "哪个细胞被判成什么、多确信"），全量存会让库体积膨胀 10 倍以上。
_SLIM_KEYS = (
    "bbox", "binary_result", "final_class", "final_confidence",
    "uncertainty", "grading_conflict", "dual_conflict",
    "auxiliary", "patch_xy",
)


def _slim_cells(cells: List[dict]) -> List[dict]:
    return [{k: c[k] for k in _SLIM_KEYS if k in c} for c in cells]


async def persist(
    session_id: str,
    state: dict,
    operator_id: str,
    patient_ref: Optional[str] = None,
) -> None:
    """写入（幂等：同一 session 重复调用只更新可变字段）。

    只应在 `is_complete=True` 时调用 —— 澄清中断时 state 还不完整。
    """
    cv = state.get("cv_result") or {}
    async with acquire() as conn:
        await conn.execute(
            """
            INSERT INTO diagnosis_reports (
                session_id, operator_id, patient_ref, label_space,
                risk_level, risk_scope, review_required,
                conflict_count, grading_conflict_count, dual_conflict_count,
                total_cells, abnormal_cells, highest_risk, max_uncertainty,
                report_md, recommendations, references_list,
                cells_json, knowledge_json, gradcam_path
            ) VALUES (
                $1,$2,$3,$4,
                $5,$6,$7,
                $8,$9,$10,
                $11,$12,$13,$14,
                $15,$16,$17,
                $18,$19,$20
            )
            ON CONFLICT (session_id) DO UPDATE SET
                risk_level             = EXCLUDED.risk_level,
                risk_scope             = EXCLUDED.risk_scope,
                review_required        = EXCLUDED.review_required,
                conflict_count         = EXCLUDED.conflict_count,
                grading_conflict_count = EXCLUDED.grading_conflict_count,
                dual_conflict_count    = EXCLUDED.dual_conflict_count,
                report_md              = EXCLUDED.report_md,
                recommendations        = EXCLUDED.recommendations,
                references_list        = EXCLUDED.references_list,
                cells_json             = EXCLUDED.cells_json,
                knowledge_json         = EXCLUDED.knowledge_json,
                gradcam_path           = EXCLUDED.gradcam_path
            """,
            session_id,
            operator_id,
            patient_ref,
            cv.get("label_space") or "sipakmed",
            state.get("risk_level"),
            state.get("risk_scope"),
            bool(cv.get("review_required")),
            int(cv.get("conflict_count") or 0),
            int(cv.get("grading_conflict_count") or 0),
            int(cv.get("dual_conflict_count") or 0),
            cv.get("total_cells"),
            cv.get("abnormal_cells"),
            cv.get("highest_risk"),
            cv.get("max_uncertainty"),
            state.get("diagnosis_report"),
            state.get("recommendations") or [],
            state.get("references") or [],
            _slim_cells(cv.get("cells") or []),
            state.get("retrieved_knowledge") or [],
            state.get("gradcam_path"),
        )


async def update_gradcam(session_id: str, path: str) -> None:
    """按需生成 Grad-CAM 后回写路径。"""
    async with acquire() as conn:
        await conn.execute(
            "UPDATE diagnosis_reports SET gradcam_path = $2 WHERE session_id = $1",
            session_id, path,
        )


async def load(session_id: str) -> Optional[dict]:
    """从冷存储还原成 DiagnosisState 兼容的 dict，供 Redis 过期后回填。"""
    async with acquire() as conn:
        row = await conn.fetchrow(
            "SELECT * FROM diagnosis_reports WHERE session_id = $1", session_id
        )
    if not row:
        return None

    cells = row["cells_json"] or []
    return {
        # 原图可能已被 uploads 清理任务删除，如实置空
        "image_path": None,
        "cv_result": {
            "total_cells": row["total_cells"],
            "abnormal_cells": row["abnormal_cells"],
            "cells": cells,
            "highest_risk": row["highest_risk"],
            "max_uncertainty": row["max_uncertainty"],
            "risk_level": row["risk_level"],
            "risk_scope": row["risk_scope"],
            "label_space": row["label_space"],
            "review_required": row["review_required"],
            "conflict_count": row["conflict_count"],
            "grading_conflict_count": row["grading_conflict_count"],
            "dual_conflict_count": row["dual_conflict_count"],
            "interpretable": True,
            "needs_clarification": False,
            "error": None,
        },
        "risk_level": row["risk_level"],
        "risk_scope": row["risk_scope"],
        "diagnosis_report": row["report_md"],
        "recommendations": row["recommendations"] or [],
        "references": row["references_list"] or [],
        "retrieved_knowledge": row["knowledge_json"] or [],
        "gradcam_path": row["gradcam_path"],
        "clarification_rounds": 0,
        "is_complete": True,
        "current_step": "restored_from_db",
    }


async def get_operator_id(session_id: str) -> Optional[str]:
    """Redis 归属键过期后，从这里取权威归属（否则 DB 恢复永远 403）。"""
    async with acquire() as conn:
        return await conn.fetchval(
            "SELECT operator_id FROM diagnosis_reports WHERE session_id = $1",
            session_id,
        )