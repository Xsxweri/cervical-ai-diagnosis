"""节点4-5：RAG 检索与检索后重评。"""

import asyncio
from typing import Any, Dict

from Agents.knowledge import get_kb
from Agents.state import DiagnosisState
from utils.label_mapping import max_risk, risk_floor_for
from utils.uncertainty import (
    EVIDENCE_FULL_COUNT,
    EVIDENCE_RELEVANT_COS,
    UNCERTAINTY_HIGH,
    risk_level_from_ratio,
    risk_scope_for,
)
from utils.trace import traced


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

    with traced(state, "deep_retrieval"):
        results = await asyncio.gather(*tasks)
    all_knowledge = [item for sub in results for item in sub]

    return {
        "retrieved_knowledge": all_knowledge,
        "retrieval_done": True,
        "current_step": "deep_retrieval"
    }

def _distance_to_cosine(d) -> float:
    """
    余弦相似度 = 1 - (欧氏距离^2)/2
    """
    dist = max(float(d or 0.0), 0.0)
    return round(1.0 - (dist * dist) / 2.0, 4)


def evidence_sufficiency_from(knowledge: list[dict]) -> tuple[float, int, float]:
    """
    计算文献证据的充分性指标：
        sufficiency: 充分性 ∈ [0,1]，文献条数越多，越能保证检索结果的可靠性
        count: 相关文献条数，用于计算文献质量
    """
    if not knowledge:
        return 0.0, 0, 0.0

    cosines = [_distance_to_cosine(k.get("score")) for k in knowledge]
    relevant = [c for c in cosines if c >= EVIDENCE_RELEVANT_COS]

    sufficiency = min(1.0, len(relevant) / EVIDENCE_FULL_COUNT)
    mean_cos = round(sum(relevant) / len(relevant), 4) if relevant else 0.0
    return round(sufficiency, 4), len(relevant), mean_cos

async def post_retrieval_assessment_agent(state: DiagnosisState) -> Dict[str, Any]:
    """
    检索后评估，对检索结果进行评估，并返回评估结果
    """
    cv_result = state.get("cv_result", {}) or {}
    knowledge = state.get("retrieved_knowledge") or []
    total_cells = int(cv_result.get("total_cells", 0) or 0)
    scope = risk_scope_for(total_cells)

    # ---- 维度 1：模型不确定性（只读，原样传递）----
    model_u = float(cv_result.get("max_uncertainty", 0.0) or 0.0)

    # ---- 维度 2：证据充分性（独立计算）----
    sufficiency, n_relevant, mean_cos = evidence_sufficiency_from(knowledge)

    # ---- 维度 3：临床风险（细胞证据 + 严重度下限）----
    abnormal = int(cv_result.get("abnormal_cells", 0) or 0)
    ratio = abnormal / total_cells if total_cells else 0.0
    label_space = cv_result.get("label_space") or ""
    clinical_risk = cv_result.get("risk_level") or max_risk(
        risk_level_from_ratio(ratio),
        *[risk_floor_for(c.get("final_class"), label_space)
          for c in (cv_result.get("cells") or [])
          if (c.get("binary_result") or {}).get("class") == "abnormal"
          and risk_floor_for(c.get("final_class"), label_space)],
    )

    # ---- 三个维度合成"下一步动作"，互不改写 ----
    if model_u > UNCERTAINTY_HIGH:
        risk_level = "pending"
        #   证据充分 → 模型能力边界（换图 / 人工镜检）
        #   证据不足 → 信息不足（补取材 / 追问病史）
        if sufficiency >= 1.0:
            closure = "文献已充分覆盖，但模型对图像本身不确定，属模型能力边界，建议人工镜检"
        else:
            closure = "文献亦不足以覆盖争议点，证据不足，建议人工镜检并补充取材或追问病史"
        reason = (
            f"模型不确定性仍高({model_u:.3f} > {UNCERTAINTY_HIGH})；"
            f"证据充分性 {sufficiency:.2f}（达标 {n_relevant}/{len(knowledge)} 条，"
            f"平均余弦 {mean_cos:.3f}）。{closure}"
        )
    else:
        risk_level = clinical_risk
        reason = (
            f"模型不确定性 {model_u:.3f}，证据充分性 {sufficiency:.2f}"
            f"（达标 {n_relevant}/{len(knowledge)} 条，平均余弦 {mean_cos:.3f}），"
            f"异常占比 {ratio:.2f} → {clinical_risk}"
        )
        if sufficiency < 1.0:
            reason += "；⚠️ 证据未达满分，建议人工复核"

    return {
        "model_uncertainty": round(model_u, 4),    # 只读；不返回 max_uncertainty
        "evidence_sufficiency": sufficiency,
        "evidence_relevant_count": n_relevant,
        "evidence_mean_cosine": mean_cos,
        "retrieval_done": True,
        "needs_clarification": False,
        "action": "direct_report",
        "risk_level": risk_level,
        "risk_scope": scope,
        "decision_reason": reason,
        "current_step": "post_retrieval_assessment",
    }