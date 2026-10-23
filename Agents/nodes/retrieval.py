"""节点4-5：RAG 检索与检索后重评。"""

import asyncio
from typing import Any, Dict

from Agents.knowledge import get_kb
from Agents.state import DiagnosisState
from utils.uncertainty import (
    UNCERTAINTY_HIGH,
    UNCERTAINTY_MEDIUM,
    risk_level_from_ratio,
    risk_scope_for,
)


async def deep_retrieval_agent(state: DiagnosisState) -> Dict[str, Any]:
    """中等不确定性时进行深度知识检索"""
    kb = get_kb()
    cv_result = state.get("cv_result", {}) or {}
    classification = cv_result.get("highest_risk", "")

    tasks = [kb.retrieve_by_classification(classification, k=5)]

    # 2. 基于形态学特征检索 + 3 个最 confident 的异常细胞
    cells = cv_result.get("cells") or []
    if cells:
        abnormal = [c for c in cells
                    if (c.get("binary_result") or {}).get("class") == "abnormal"]
        abnormal.sort(key=lambda c: float(c.get("final_confidence") or 0.0), reverse=True)
        for c in abnormal[:3]:
            morph = c.get("morphology") or {}
            if morph.get("measurement_reliable"):
                tasks.append(kb.retrieve_by_morphology(morph, k=2))
                break

    results = await asyncio.gather(*tasks)
    all_knowledge = [item for sub in results for item in sub]

    return {
        "retrieved_knowledge": all_knowledge,
        "retrieval_done": True,
        "current_step": "deep_retrieval"
    }


async def post_retrieval_assessment_agent(state: DiagnosisState) -> Dict[str, Any]:
    """检索后评估：把文献证据折回风险等级，避免 RAG 只是装饰。"""
    cv_result = state.get("cv_result", {}) or {}
    knowledge = state.get("retrieved_knowledge") or []
    total_cells = int(cv_result.get("total_cells", 0) or 0)
    scope = risk_scope_for(total_cells)

    raw_u = float(cv_result.get("max_uncertainty", 0.0) or 0.0)
    # 证据质量：文献条数越多、相似度越高，下调越明显（上限下调 40%）
    if knowledge:
        evidence_gain = min(0.4, 0.1 * len(knowledge))
        adjusted = raw_u * (1.0 - evidence_gain)
        evidence_note = f"文献 {len(knowledge)} 条，不确定性 {raw_u:.3f} → {adjusted:.3f}"
    else:
        adjusted = raw_u
        evidence_note = f"未检索到相关文献，不确定性维持 {raw_u:.3f}"

    if adjusted > UNCERTAINTY_HIGH:
        # 仍高：标注复核，但仍出报告（不再回到追问，避免回环）
        action = "direct_report"
        risk_level = "pending"
        needs_clarification = False
    elif adjusted > UNCERTAINTY_MEDIUM:
        action = "direct_report"
        risk_level = "medium"
        needs_clarification = False
    else:
        abnormal = int(cv_result.get("abnormal_cells", 0) or 0)
        ratio = abnormal / total_cells if total_cells else 0.0
        action = "direct_report"
        risk_level = risk_level_from_ratio(ratio)
        needs_clarification = False

    return {
        "max_uncertainty": round(adjusted, 4),
        "retrieval_done": True,
        "needs_clarification": needs_clarification,
        "action": action,
        "risk_level": risk_level,
        "risk_scope": scope,
        "decision_reason": f"检索后重评：{evidence_note} → {risk_level}",
        "current_step": "post_retrieval_assessment",
    }