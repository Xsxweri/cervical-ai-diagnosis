"""节点1:不确定性评估。

依据图像级不确定性（P90 归一化熵）、细胞总数、已追问轮次与是否已检索，
决定下一步动作:clarify / deep_retrieve / direct_report / unreadable。
"""

from typing import Any, Dict, Optional

from Agents.state import DiagnosisState
from utils.uncertainty import (
    MAX_CLARIFICATION_ROUNDS,
    UNCERTAINTY_HIGH,
    UNCERTAINTY_MEDIUM,
    risk_level_from_ratio,
    risk_scope_for,
)

# 阈值常量（别名，保持与 utils.uncertainty 单一事实源同步）
HIGH_UNCERTAINTY = UNCERTAINTY_HIGH
MEDIUM_UNCERTAINTY = UNCERTAINTY_MEDIUM


def _decision(
    *,
    action: str,
    risk_level: str,
    reason: str,
    max_uncertainty: float,
    rounds: int,
    scope: Optional[str],
    needs_clarification: bool = False,
    error_message: Optional[str] = None,
) -> Dict[str, Any]:
    """
    统一构造节点输出
    输入：
    - action: 节点输出动作
    - risk_level: 评估结果风险等级
    - reason: 评估结果描述
    - max_uncertainty: 最大不确定性
    - rounds: 累计 Clarification 轮次
    - scope: 评估结果风险范围
    - needs_clarification: 是否需要 Clarification
    - error_message: 错误信息
    """
    out: Dict[str, Any] = {
        "max_uncertainty": max_uncertainty,
        "needs_clarification": needs_clarification,
        "clarification_rounds": rounds,
        "action": action,
        "risk_level": risk_level,
        "risk_scope": scope,
        "decision_reason": reason,
        "current_step": "uncertainty_assessment",
    }
    if error_message is not None:
        out["error_message"] = error_message
    return out

def uncertainty_assessment_agent(state: DiagnosisState) -> Dict[str, Any]:
    """利用MC Dropout结果评估不确定性"""
    assert UNCERTAINTY_HIGH >= UNCERTAINTY_MEDIUM, "不确定性阈值配置反了"
    cv_result = state.get("cv_result", {}) or {}
    max_uncertainty = float(cv_result.get("max_uncertainty", 0.0) or 0.0)
    total_cells = int(cv_result.get("total_cells", 0) or 0)
    rounds = int(state.get("clarification_rounds", 0) or 0)
    retrieval_done = bool(state.get("retrieval_done", False))
    scope = risk_scope_for(total_cells)

    common = {"max_uncertainty": max_uncertainty, "rounds": rounds}

    # 不可读性判断
    if total_cells == 0:
        return _decision(
            action="unreadable",
            risk_level="unreadable",
            scope=None,
            reason="未检出任何细胞，无法判读",
            error_message=(
                "未检测到可判读的宫颈细胞，无法生成诊断结论。"
                "请确认图像为细胞学视野、清晰度足够、且未为大面积空白或非细胞内容，然后重新上传。"
            ),
            **common,
        )

    # ---- 高不确定性 ----
    if max_uncertainty > HIGH_UNCERTAINTY:
        # 已达检索上限 + 追问上限 → 强制出报告并标注证据不足
        if rounds >= MAX_CLARIFICATION_ROUNDS and retrieval_done:
            return _decision(
                action="direct_report",
                risk_level="pending",
                scope=scope,
                reason=(
                    f"不确定性仍高({max_uncertainty:.3f})，已追问 {rounds} 轮且完成文献检索，"
                    f"不再继续追问，降级为证据不足报告"),
                error_message=(
                    f"已追问 {rounds} 轮并检索文献后，证据仍不足以给出明确分级。"
                    "建议人工镜检复核或重新采样。"),
                **common,
            )

        # 继续追问
        if rounds < MAX_CLARIFICATION_ROUNDS:
            return _decision(
                action="clarify",
                risk_level="pending",
                scope=scope,
                needs_clarification=True,
                reason=(
                    f"不确定性 {max_uncertainty:.3f} > {HIGH_UNCERTAINTY}，"
                    f"第 {rounds + 1}/{MAX_CLARIFICATION_ROUNDS} 轮追问"),
                **common,
            )
        # 检索更多知识
        return _decision(
            action="deep_retrieve",
            risk_level="medium",
            scope=scope,
            reason=f"不确定性 {max_uncertainty:.3f} 仍高，追问已达上限，转为文献检索辅助",
            **common,
        )

    # ---- 中等不确定性 ----
    if max_uncertainty > MEDIUM_UNCERTAINTY and not retrieval_done:
        return _decision(
            action="deep_retrieve",
            risk_level="medium",
            scope=scope,
            reason=f"不确定性 {max_uncertainty:.3f} 处于中等区间，先检索文献再评估",
            **common,
        )

    # ----  低不确定性（或中不确定性但已检索过）：直接出报告 ----
    abnormal_count = int(cv_result.get("abnormal_cells", 0) or 0)
    ratio = abnormal_count / total_cells
    risk = risk_level_from_ratio(ratio)
    return _decision(
        action="direct_report",
        risk_level=risk,
        scope=scope,
        reason=(
            f"不确定性 {max_uncertainty:.3f} ≤ {MEDIUM_UNCERTAINTY}，"
            f"直接出报告；异常占比 {ratio:.2f} → {risk}" if max_uncertainty <= MEDIUM_UNCERTAINTY
            else f"已检索文献，不确定性 {max_uncertainty:.3f} 不再降级追问，直接出报告"),
        **common,
    )