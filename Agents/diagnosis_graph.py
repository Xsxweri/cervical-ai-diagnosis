from typing import Dict, Any, List, Optional
from Agents.state import DiagnosisState
from services.rag_service import MedicalKnowledgeBase
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, END
from contextlib import AsyncExitStack
from core.redis_client import REDIS_URL
from langgraph.checkpoint.redis import AsyncRedisSaver
from langgraph.types import interrupt
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from Prompts import render, raw
from Prompts.clinical_content import (
    CLARIFICATION_FALLBACK_QUESTIONS,
    RECOMMENDATIONS_BY_RISK,
    RECOMMENDATION_DEFAULT,
    REPORT_FALLBACK,
    RECOMMENDATION_CELL_SCOPE
)
import os
import asyncio
from dotenv import load_dotenv
from utils.uncertainty import (
    UNCERTAINTY_HIGH,
    UNCERTAINTY_MEDIUM,
    MAX_CLARIFICATION_ROUNDS,
    risk_level_from_ratio,
    risk_scope_for,
)

load_dotenv()

class _UnavailableKnowledgeBase:
    """RAG 不可用时的空实现：检索一律返回空，避免上层调用崩溃。"""
    async def retrieve(self, query, k: int = 5):
        return []
    async def retrieve_by_classification(self, classification, k: int = 3):
        return []
    async def retrieve_by_morphology(self, morphology_features, k: int = 3):
        return []
    async def load_documents(self, pdf_paths, force: bool = False):
        return 0

# LLM
def get_llm():
    return ChatOpenAI(
        model=os.getenv("LLM_MODEL_ID"),
        temperature=0.3,
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("LLM_BASE_URL"),
        timeout=120.0,
        max_retries=3
    )

# 结构化输出模型（替代 json.loads）
class ClarificationResponse(BaseModel):
    questions: List[str] = Field(description="2-3个追问问题")
    reasoning: str = Field(default="", description="追问理由")

# 阈值常量
HIGH_UNCERTAINTY = UNCERTAINTY_HIGH
MEDIUM_UNCERTAINTY = UNCERTAINTY_MEDIUM

# 可解释性开关：开启后在可解释性节点为异常细胞生成 Grad-CAM（默认关闭，避免额外 GPU 开销）
ENABLE_GRADCAM = os.getenv("ENABLE_GRADCAM", "false").strip().lower() in {"1", "true", "yes", "on"}


def aggregate_case_evidence(cv_result: dict) -> dict:
    """聚合 Case 级别信息
    :param cv_result: 模型输出结果
    """
    from collections import Counter

    cells = cv_result.get("cells") or []
    total = len(cells)
    if total == 0:
        return {"total": 0}

    # 1) 类别分布（按 final_class）
    dist = Counter(c.get("final_class") for c in cells)

    # 2) 异常细胞的置信度 Top-N（已排序时取前 3）
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

    # 4) 冲突证据：异常细胞分型不止一种
    abnormal_types = {c.get("final_class") for c in abnormal}
    has_conflict = len(abnormal_types) > 1

    # 5) 不确定性分布
    us = [float(c.get("uncertainty") or 0.0) for c in cells]

    return {
        "total": total,
        "abnormal_count": len(abnormal),
        "class_distribution": dict(dist),
        "abnormal_types": sorted(abnormal_types),
        "has_conflict": has_conflict,
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

# ====节点1、不确定性评估====
def uncertainty_assessment_agent(state: DiagnosisState) -> Dict[str, Any]:
    """利用MC Dropout结果评估不确定性"""
    assert UNCERTAINTY_HIGH >= UNCERTAINTY_MEDIUM, "不确定性阈值配置反了"
    cv_result = state.get("cv_result", {}) or {}
    max_uncertainty = float(cv_result.get("max_uncertainty", 0.0) or 0.0)
    total_cells = int(cv_result.get("total_cells", 0) or 0)
    rounds = int(state.get("clarification_rounds", 0) or 0)
    retrieval_done = bool(state.get("retrieval_done", False))
    scope = risk_scope_for(total_cells)

    # 不可读性判断
    if total_cells == 0:
        return {
            "max_uncertainty": max_uncertainty,
            "needs_clarification": False,
            "clarification_rounds": rounds,
            "action": "unreadable",
            "risk_level": "unreadable",
            "risk_scope": None,
            "decision_reason": "未检出任何细胞，无法判读",
            "error_message": (
                "未检测到可判读的宫颈细胞，无法生成诊断结论。"
                "请确认图像为细胞学视野、清晰度足够、且未为大面积空白或非细胞内容，然后重新上传。"
            ),
            "current_step": "uncertainty_assessment",
        }

    # ---- 高不确定性 ----
    if max_uncertainty > HIGH_UNCERTAINTY:
        # 已达检索上限 + 追问上限 → 强制出报告并标注证据不足
        if rounds >= MAX_CLARIFICATION_ROUNDS and retrieval_done:
            return {
                "max_uncertainty": max_uncertainty,
                "needs_clarification": False,
                "clarification_rounds": rounds,
                "action": "direct_report",
                "risk_level": "pending",
                "risk_scope": scope,
                "decision_reason": (
                    f"不确定性仍高({max_uncertainty:.3f})，已追问 {rounds} 轮且完成文献检索，"
                    f"不再继续追问，降级为证据不足报告"),
                "error_message": (
                    f"已追问 {rounds} 轮并检索文献后，证据仍不足以给出明确分级。"
                    "建议人工镜检复核或重新采样。"),
                "current_step": "uncertainty_assessment",
            }
        
         # 继续追问
        if rounds < MAX_CLARIFICATION_ROUNDS:
            return {
                "max_uncertainty": max_uncertainty,
                "needs_clarification": True,
                "clarification_rounds": rounds,
                "action": "clarify",
                "risk_level": "pending",
                "risk_scope": scope,
                "decision_reason": (
                    f"不确定性 {max_uncertainty:.3f} > {HIGH_UNCERTAINTY}，"
                    f"第 {rounds + 1}/{MAX_CLARIFICATION_ROUNDS} 轮追问"),
                "current_step": "uncertainty_assessment",
            }
        # 检索更多知识
        return {
            "max_uncertainty": max_uncertainty,
            "needs_clarification": False,
            "clarification_rounds": rounds,
            "action": "deep_retrieve",
            "risk_level": "medium",
            "risk_scope": scope,
            "decision_reason": (
                f"不确定性 {max_uncertainty:.3f} 仍高，追问已达上限，转为文献检索辅助"),
            "current_step": "uncertainty_assessment",
        }

    # ---- 中等不确定性 ----
    if max_uncertainty > MEDIUM_UNCERTAINTY and not retrieval_done:
        return {
            "max_uncertainty": max_uncertainty,
            "needs_clarification": False,
            "clarification_rounds": rounds,
            "action": "deep_retrieve",
            "risk_level": "medium",
            "risk_scope": scope,
            "decision_reason": (
                f"不确定性 {max_uncertainty:.3f} 处于中等区间，先检索文献再评估"),
            "current_step": "uncertainty_assessment",
        }
    # ----  低不确定性（或中不确定性但已检索过）：直接出报告 ----
    abnormal_count = int(cv_result.get("abnormal_cells", 0) or 0)
    ratio = abnormal_count / total_cells
    risk = risk_level_from_ratio(ratio)
    return {
        "max_uncertainty": max_uncertainty,
        "needs_clarification": False,
        "clarification_rounds": rounds,
        "action": "direct_report",
        "risk_level": risk,
        "risk_scope": scope,
        "decision_reason": (
            f"不确定性 {max_uncertainty:.3f} ≤ {MEDIUM_UNCERTAINTY}，"
            f"直接出报告；异常占比 {ratio:.2f} → {risk}" if max_uncertainty <= MEDIUM_UNCERTAINTY
            else f"已检索文献，不确定性 {max_uncertainty:.3f} 不再降级追问，直接出报告"),
        "current_step": "uncertainty_assessment",
    }


# ====决策路由====
def decision_router(state: DiagnosisState) -> str:
    """根据不确定性评估结果路由"""
    return state.get("action","direct_report")

# ====主动追问====
async def prepare_clarification_agent(state: DiagnosisState) -> Dict[str, Any]:
    """高不确定性时主动追问关键信息"""
    cv_result = state.get("cv_result", {}) or {}
    llm = get_llm()
    structured_llm = llm.with_structured_output(ClarificationResponse)

    # 基于细胞特征生成追问问题
    prompt = render(
        "clarification_questions",
        uncertainty=f"{(state.get('max_uncertainty') or 0.0):.3f}",
        total_cells=cv_result.get("total_cells", 0),
        abnormal_cells=cv_result.get("abnormal_cells", 0),
        highest_risk=cv_result.get("highest_risk", "未知"),
    )

    try:
        result = await structured_llm.ainvoke(prompt)
        questions = result.questions or CLARIFICATION_FALLBACK_QUESTIONS
    except Exception as e:
        print(f"LLM生成追问问题失败: {e}")
        questions = CLARIFICATION_FALLBACK_QUESTIONS
    return {
        "clarification_questions": questions,
        "needs_clarification": True,
        "current_step": "clarification_prepared"
    }

async def clarification_agent(state: DiagnosisState) -> Dict[str, Any]:
    """通过 interrupt 暂停图执行，等待用户回答后恢复"""
    questions = state.get("clarification_questions") or []
    rounds = int(state.get("clarification_rounds", 0) or 0)

    user_response = interrupt({
        "questions": questions,
        "message": "检测到高不确定性，需要用户补充信息...",
        "round": rounds + 1,
        "max_rounds": MAX_CLARIFICATION_ROUNDS,
    })

    history = list(state.get("conversation_history") or [])
    history.append(HumanMessage(content=str(user_response)))

    return {
        "conversation_history": history,
        "user_query": str(user_response),
        "clarification_rounds": rounds + 1,   # ← 此处递增
        "needs_clarification": False,
        "action": "re_evaluate",
        "current_step": "clarification_resolved"
    }

# ====RAG检索====
async def deep_retrieval_agent(state: DiagnosisState) -> Dict[str, Any]:
    """中等不确定性时进行深度知识检索"""
    kb = get_kb()
    cv_result = state.get("cv_result", {}) or {}
    classification = cv_result.get("highest_risk", "")

    tasks =[kb.retrieve_by_classification(classification, k=5)]

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
    """
    检索后评估，对检索结果进行评估，并返回评估结果
    """
    cv_result = state.get("cv_result", {}) or {}
    knowledge = state.get("retrieved_knowledge") or []
    total_cells = int(cv_result.get("total_cells", 0) or 0)
    scope = risk_scope_for(total_cells)

    raw_u = float(cv_result.get("max_uncertainty", 0.0) or 0.0)
    # 证据质量：文献条数越多、相似度越高，下调越明显（上限下调 40%）
    if knowledge:
        top_scores = sorted((d.get("score") or 1.0 for d in knowledge))[:3]
        # Chroma 返回的是距离，越小越相关；这里只做保守的定性下调
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

# ==== 输出报告 ====
async def generate_report_agent(state: DiagnosisState) -> Dict[str, Any]:
    """生成诊断报告"""
    cv_results = state.get("cv_result") or {}
    ev = aggregate_case_evidence(cv_results)
    scope = state.get("risk_scope") or "cell"

    if scope == "case":
        scope_text = "病例级（基于异常细胞占比，样本中细胞数足以代表整体）"
    else:
        scope_text = (
            f"细胞级提示（仅检出 {ev.get('total', 0)} 个细胞，"
            f"占比不足以代表病例整体，不得作为病例级风险结论）")

    dist = ev.get("class_distribution") or {}
    dist_text = "、".join(f"{k}:{v}" for k, v in sorted(dist.items())) or "无"

    top_text = "、".join(
        f"{t['class']}(conf={t['confidence']:.2f})" for t in ev.get("top_abnormal") or []
    ) or "无"

    def _fmt(s):
        if not s:
            return "无可靠测量"
        return (f"均值 {s['mean']:.3f}（中位 {s['median']:.3f}，"
                f"范围 {s['min']:.3f}–{s['max']:.3f}，n={s['n']}）")

    conflict_text = (
        f"存在冲突证据：异常细胞分型不一致（{', '.join(ev.get('abnormal_types') or [])}）"
        if ev.get("has_conflict") else "无冲突证据（异常细胞分型一致）")

    # 构建诊断上下文
    context = f'''
    细胞检测结果（病例级聚合）:
    - 总细胞数: {ev.get('total', 0)}
    - 异常细胞数: {ev.get('abnormal_count', 0)}
    - 判定口径: {scope_text}
    - 风险等级: {state.get('risk_level')}
    - 决策依据: {state.get('decision_reason') or '未记录'}

    分类分布:
    - {dist_text}
    - 最高风险细胞（Top-{len(ev.get('top_abnormal') or [])}）: {top_text}
    - {conflict_text}

    形态学（仅统计测量可靠的细胞）:
    - 核质比: {_fmt(ev.get('nc_ratio'))}
    - 圆形度: {_fmt(ev.get('circularity'))}
    - 不可靠测量细胞数: {ev.get('morphology_unreliable', 0)}
    - 说明: 形态学为自动测量，不可靠时不得据此得出形态学结论

    不确定性:
    - 图像级(P90 归一化熵): {(state.get('max_uncertainty') or 0.0):.3f}
    - 细胞级 max / median: {ev.get('uncertainty_max', 0.0):.3f} / {ev.get('uncertainty_median', 0.0):.3f}
    - 口径: {cv_results.get('uncertainty_metric', 'normalized_entropy')}，聚合 {cv_results.get('uncertainty_aggregation', 'p90')}
    '''

    report_prompt = ChatPromptTemplate.from_messages([
        ("system", raw("diagnosis_report_system")),
        ("human", "{context}\n\n请生成面向医生的辅助诊断报告。")
    ])

    llm = get_llm()
    chain = report_prompt | llm

    try:
        response = await chain.ainvoke({"context": context})

        # 生成建议
        if (state.get("risk_scope") or "cell") == "case":
            recommendations = RECOMMENDATIONS_BY_RISK.get(
                state.get("risk_level"), RECOMMENDATION_DEFAULT
            )
        else:
            recommendations = RECOMMENDATION_CELL_SCOPE

        return {
            "diagnosis_report": response.content,
            "recommendations": recommendations,
            "current_step": "generate_report",
            "is_complete": True
        }
    except Exception as e:  # 捕获所有异常
        print(f"LLM生成报告失败: {e}")
        return {
            "diagnosis_report": REPORT_FALLBACK,
            "recommendations": RECOMMENDATION_CELL_SCOPE if scope != "case" else RECOMMENDATION_DEFAULT,
            "current_step": "generate_report",
            "is_complete": True
        }

# ====不可判读终止====
def unreadable_report_agent(state: DiagnosisState) -> Dict[str, Any]:
    """0细胞/无法判读：终止流程，把原因作为用户可见的输出。"""
    reason = state.get("error_message") or "未检测到可判读的宫颈细胞，无法生成诊断结论。"
    return {
        "diagnosis_report": f"## 无法判读\n\n{reason}",
        "recommendations": [
            "请重新采样或更换细胞学视野图像后重试",
            "如图像质量存疑，建议直接人工镜检复核",
        ],
        "error_message": reason,
        "is_complete": True,
        "current_step": "unreadable",
    }

# ====可解释性辅助：Grad-CAM====
def _compute_gradcam(state: DiagnosisState) -> Optional[Dict[str, Any]]:
    """为置信度最高的异常细胞生成 Grad-CAM"""
    try:
        from PIL import Image
        from services.vision_service import get_model_manager, ModelType
        from tools.gradcam_tool import GradCAMTool

        image_path = state.get("image_path")
        cv_result = state.get("cv_result") or {}
        cells = cv_result.get("cells") or []
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
        print(f"Grad-CAM 生成失败: {e}")
        return None

# ====可解释性报告====
async def explainable_report_agent(state: DiagnosisState) -> Dict[str, Any]:
    """
    生成带可解释性的综合报告
    结合形态学分析 + 不确定性 + (可选)RAG医学依据
    无论是否检索到知识都会生成解释，避免低不确定性病例缺失可解释性
    """
    cv_results = state.get("cv_result") or {}
    knowledge = state.get("retrieved_knowledge") or []

    # 构建可解释性上下文
    explain_context = f'''
    诊断结果: {state.get("diagnosis_report", "")}

    可解释性信息:
    1. 模型关注区域: AI主要关注细胞的核区域和细胞质边界
    2. 关键形态特征:
    '''

    for i, cell in enumerate((cv_results.get("cells") or [])[:3], 1):
        morph = cell.get("morphology") or {}
        explain_context += f'''
    细胞{i}:
    - 核质比: {(morph.get("nuclear_cytoplasmic_ratio") or 0.0):.3f}
    - 圆形度: {(morph.get("circularity") or 0.0):.3f}
    - 解读: {morph.get("interpretation", "")}
    '''

    explain_context += f'''
    3. 不确定性分析: {cv_results.get("uncertainty_metric", "normalized_entropy")}={(state.get("max_uncertainty") or 0.0):.3f}
    (图像级取 {cv_results.get("uncertainty_aggregation", "p90")} 聚合，范围 0~1, 越大越不确定)
    '''

    # 仅当检索到知识时追加医学依据
    if knowledge:
        explain_context += '\n  相关医学依据:\n'
        for doc in knowledge[:3]:
            explain_context += f'    - {(doc.get("content", "") or "")[:150]}...\n'

    # 可选：Grad-CAM 注意力可解释性（ENABLE_GRADCAM=true 时生成）
    gradcam_path = None
    if ENABLE_GRADCAM:
        gradcam = await asyncio.to_thread(_compute_gradcam, state)
        if gradcam:
            gradcam_path = gradcam.get("overlay_path")
            explain_context += f'''
       4. 模型注意力(Grad-CAM): {gradcam.get("attention_analysis", "")}
       '''

    explain_prompt = ChatPromptTemplate.from_messages([
        ("system", raw("explainability_system")),
        ("human", "{context}\n\n请生成可解释性报告。")
    ])

    # 从检索结果构建参考文献（按 source:page 去重，过滤未知来源）
    references = []
    seen = set()
    for doc in knowledge:
        source = doc.get("source", "")
        page = doc.get("page", 0)
        key = f"{source}:{page}"
        if key in seen or not source or source == "未知":
            continue
        seen.add(key)
        references.append(f"{source} (第{page}页)")

    try:
        llm = get_llm()
        chain = explain_prompt | llm
        response = await chain.ainvoke({"context": explain_context})
        full_report = (state.get("diagnosis_report") or "") + "\n\n---\n\n【AI诊断逻辑解释】\n" + response.content
    except Exception as e:
        print(f"LLM生成可解释性报告失败: {e}")
        full_report = state.get("diagnosis_report") or ""

    return {
        "diagnosis_report": full_report,
        "references": references,
        "gradcam_path": gradcam_path,
        "current_step": "explainable_report",
        "is_complete": True
    }

def explainable_router(state: DiagnosisState) -> str:
    """
    判断是否需要生成可解释性报告
    """
    if state.get("risk_level") in ("high", "emergency", "pending", "unreadable"):
        return "explainable_report"
    ev_conflict = bool((state.get("cv_result") or {}).get("has_conflict"))
    if float(state.get("max_uncertainty") or 0.0) > UNCERTAINTY_MEDIUM or ev_conflict:
        return "explainable_report"
    return "skip_explainable"

def skip_explainable_agent(state: DiagnosisState) -> Dict[str, Any]:
    """低风险清晰病例：跳过二次 LLM,只附加结构化摘要（零 LLM 成本）。"""
    ev = aggregate_case_evidence(state.get("cv_result") or {})
    dist = ev.get("class_distribution") or {}
    summary = (
        "\n\n---\n\n【结构化摘要(未调用 LLM)】\n"
        f"- 检出细胞 {ev.get('total', 0)} 个，异常 {ev.get('abnormal_count', 0)} 个\n"
        f"- 分类分布：{'、'.join(f'{k}:{v}' for k, v in sorted(dist.items())) or '无'}\n"
        f"- 不确定性(P90 归一化熵)：{(state.get('max_uncertainty') or 0.0):.3f}\n"
        f"- 风险等级：{state.get('risk_level')}(口径 {state.get('risk_scope')})"
    )
    return {
        "diagnosis_report": (state.get("diagnosis_report") or "") + summary,
        "current_step": "skip_explainable",
        "is_complete": True,
    }


# ====构建状态机图====
builder = StateGraph(DiagnosisState)

# 添加节点
builder.add_node("uncertainty_assessment", uncertainty_assessment_agent)
builder.add_node("prepare_clarification", prepare_clarification_agent)
builder.add_node("clarification", clarification_agent)
builder.add_node("deep_retrieval", deep_retrieval_agent)
builder.add_node("generate_report", generate_report_agent)
builder.add_node("unreadable_report", unreadable_report_agent)
builder.add_node("explainable_report", explainable_report_agent)
builder.add_node("post_retrieval_assessment", post_retrieval_assessment_agent)
builder.add_node("skip_explainable", skip_explainable_agent)


# 设置入口
builder.set_entry_point("uncertainty_assessment")

# 添加条件边
builder.add_conditional_edges(
    "uncertainty_assessment",
    decision_router,
    {
        "clarify": "prepare_clarification",
        "deep_retrieve": "deep_retrieval",
        "direct_report": "generate_report",
        "re_evaluate": "uncertainty_assessment",
        "unreadable": "unreadable_report"
    }
)

# 可解释性条件化
builder.add_conditional_edges(
    "generate_report",
    explainable_router,
    {
        "explainable_report": "explainable_report",
        "skip_explainable": "skip_explainable",
    },
)

# 问题生成后进入中断节点等待用户回答
builder.add_edge("prepare_clarification", "clarification")

# 用户回答后重新评估
builder.add_conditional_edges(
    "clarification",
    decision_router,
    {
        "re_evaluate": "uncertainty_assessment",
        "deep_retrieve": "deep_retrieval",
        "direct_report": "generate_report", 
        "unreadable": "unreadable_report"
    }
)

# 深度检索后进入报告生成
builder.add_edge("deep_retrieval", "post_retrieval_assessment")
builder.add_edge("post_retrieval_assessment", "generate_report")

# 可解释性报告后结束
builder.add_edge("explainable_report", END)
builder.add_edge("unreadable_report", END)
builder.add_edge("skip_explainable", END)

# 设置检查点后端
_checkpointer_cm = AsyncRedisSaver.from_conn_string(REDIS_URL)
_exit_stack = AsyncExitStack()
kb: MedicalKnowledgeBase = None
graph = None

def get_kb() -> MedicalKnowledgeBase:
    if kb is None:
        raise RuntimeError("知识库未初始化：请先调用init_kb()")
    return kb

async def init_kb():
    """构建/加载知识库。任何失败都降级为"无 RAG"，绝不阻断服务启动。"""
    global kb
    import glob
    try:
        # 嵌入模型加载是重同步 IO，必须放到线程里，否则阻塞事件循环
        kb = await asyncio.to_thread(MedicalKnowledgeBase, "./knowledge_db")
        pdf_files = glob.glob("./Cervical_cytology/*.pdf")
        if pdf_files:
            print(f"[RAG] 发现 {len(pdf_files)} 篇医学文献，正在检查知识库...")
            chunk_count = await kb.load_documents(pdf_files)
            print(f"[RAG] 知识库就绪，共 {chunk_count} 个文本块")
        else:
            print("[RAG] 未找到 PDF 文献，RAG 检索将返回空结果")
    except Exception as e:
        print(f"[RAG] 知识库初始化失败，RAG 降级为不可用: {e}")
        kb = _UnavailableKnowledgeBase()

async def init_checkpoint_store():
    """进入上下文打开连接 → 建索引 → 编译图（幂等性由 lifespan 保证只调一次）"""
    global graph
    checkpointer = await _exit_stack.enter_async_context(_checkpointer_cm)
    await checkpointer.asetup()
    graph = builder.compile(checkpointer=checkpointer)

async def close_checkpoint_store():
    await _exit_stack.aclose()

def get_graph():
    if graph is None:
        raise RuntimeError("图未初始化：请先在 lifespan 中调用 init_checkpoint_store()")
    return graph