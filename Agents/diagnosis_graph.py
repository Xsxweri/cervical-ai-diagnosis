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

# ====节点1、不确定性评估====
def uncertainty_assessment_agent(state: DiagnosisState) -> Dict[str, Any]:
    """利用MC Dropout结果评估不确定性"""
    assert UNCERTAINTY_HIGH >= UNCERTAINTY_MEDIUM, "不确定性阈值配置反了"
    cv_result = state.get("cv_result", {}) or {}
    max_uncertainty = cv_result.get("max_uncertainty", 0.0)
    total_cells = int(cv_result.get("total_cells", 0) or 0)
    if total_cells == 0:
        return {
            "max_uncertainty": max_uncertainty,
            "needs_clarification": True,
            "action": "clarify",
            "risk_level": "pending",
            "risk_scope": "cell",
            "error_message": (
                "未检测到可判读的宫颈细胞，无法生成诊断结论。"
                "请确认图像为细胞学视野、清晰度足够、且未为大面积空白或非细胞内容，然后重新上传。"
            ),
            "current_step": "uncertainty_assessment",
        }

    if state.get("user_query") and max_uncertainty > HIGH_UNCERTAINTY:
        print("检测到用户补充信息，忽略高不确定性，强制进入深度检索以辅助诊断")
        return {
            "max_uncertainty": max_uncertainty,
            "needs_clarification": False,
            "action": "deep_retrieve",  # 跳过直接报告，去查文献辅助
            "risk_level": "medium",
            "current_step": "uncertainty_assessment"
        }

    # 不确定性分级决策
    if max_uncertainty > HIGH_UNCERTAINTY:
        # 高不确定性: 必须追问
        return {
            "max_uncertainty": max_uncertainty,
            "needs_clarification": True,
            "action": "clarify",
            "risk_level": "pending",
            "risk_scope": risk_scope_for(total_cells),
            "current_step": "uncertainty_assessment"
        }
    elif max_uncertainty > MEDIUM_UNCERTAINTY:
        # 中等不确定性: 检索更多知识
        return {
            "max_uncertainty": max_uncertainty,
            "needs_clarification": False,
            "action": "deep_retrieve",
            "risk_level": "medium",
            "risk_scope": risk_scope_for(total_cells),
            "current_step": "uncertainty_assessment"
        }
    else:
        # 低不确定性: 直接生成报告
        abnormal_count = int(cv_result.get("abnormal_cells", 0) or 0)
        ratio = abnormal_count / total_cells
        risk = risk_level_from_ratio(ratio)

        return {
            "max_uncertainty": max_uncertainty,
            "needs_clarification": False,
            "action": "direct_report",
            "risk_level": risk,
            "risk_scope": risk_scope_for(total_cells),
            "current_step": "uncertainty_assessment"
        }
# ====节点2、决策路由====
def decision_router(state: DiagnosisState) -> str:
    """根据不确定性评估结果路由"""
    return state.get("action","direct_report")

# ====节点3、主动追问====
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

    user_response = interrupt({
        "questions": questions,
        "message": "检测到高不确定性，需要用户补充信息..."
    })

    # 将用户回答注入历史和状态
    history = list(state.get("conversation_history") or [])
    history.append(HumanMessage(content=str(user_response)))

    return {
        "conversation_history": history,
        "user_query": str(user_response),
        "needs_clarification": False,
        "action": "re_evaluate",
        "current_step": "clarification_resolved"
    }

# ====节点4、RAG检索====
async def deep_retrieval_agent(state: DiagnosisState) -> Dict[str, Any]:
    """中等不确定性时进行深度知识检索"""
    kb = get_kb()
    cv_result = state.get("cv_result", {}) or {}
    classification = cv_result.get("highest_risk", "")

    tasks =[kb.retrieve_by_classification(classification, k=5)]

    # 2. 基于形态学特征检索 (利用现有cell_analyzer)
    cells = cv_result.get("cells") or []
    if cells:
        morph = cells[0].get("morphology") or {}
        # 形态学不可靠时不作为检索条件，避免用错误特征污染检索结果
        if morph.get("measurement_reliable"):
            tasks.append(kb.retrieve_by_morphology(morph, k=3))

    results = await asyncio.gather(*tasks)
    all_knowledge = [item for sub in results for item in sub]

    return {
        "retrieved_knowledge": all_knowledge,
        "current_step": "deep_retrieval"
    }

# ====节点5、直接报告====
async def generate_report_agent(state: DiagnosisState) -> Dict[str, Any]:
    """低不确定性时直接生成诊断报告"""
    cv_results = state.get("cv_result") or {}
    cells = cv_results.get("cells") or []
    first_cell = cells[0] if cells else {}
    morphology=first_cell.get("morphology") or {}

    if morphology.get("measurement_reliable"):
        nc_line = f"{(morphology.get('nuclear_cytoplasmic_ratio') or 0.0):.3f}"
        morph_note = ""
    else:
        nc_line = "不可靠（自动分割失败，忽略此项）"
        morph_note = "\n    - 说明: 形态学自动测量不可靠，不应据此得出形态学结论"

    scope = state.get("risk_scope") or "cell"
    risk_scope_text = (
        "病例级（基于异常细胞占比）"
        if scope == "case"
        else f"细胞级提示（仅检出 {state.get('cv_result', {}).get('total_cells', 0)} 个细胞，"
             f"占比不足以代表病例整体，不得作为病例级风险结论）"
    )

    # 构建诊断上下文
    context = f'''
    细胞检测结果:
    - 总细胞数: {cv_results.get("total_cells", 0)}
    - 异常细胞数: {cv_results.get("abnormal_cells", 0)}
    - 最高风险分类: {cv_results.get("highest_risk", "")}
    - 模型置信度: {(first_cell.get("final_confidence") or 0.0):.3f}
    - 不确定性: {(state.get("max_uncertainty") or 0.0):.3f}
    - 风险口径: {risk_scope_text}

    形态学分析:
    - 核质比: {nc_line}
    - 圆形度: {(morphology.get('circularity') or 0.0):.3f}{morph_note}
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

# ====节点6、可解释性报告====
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
    3. 不确定性分析: MC Dropout标准差={(state.get("max_uncertainty") or 0.0):.3f}
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


# ====节点7、构建状态机图====
builder = StateGraph(DiagnosisState)

# 添加节点
builder.add_node("uncertainty_assessment", uncertainty_assessment_agent)
builder.add_node("prepare_clarification", prepare_clarification_agent)
builder.add_node("clarification", clarification_agent)
builder.add_node("deep_retrieval", deep_retrieval_agent)
builder.add_node("generate_report", generate_report_agent)
builder.add_node("explainable_report", explainable_report_agent)

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
        "re_evaluate": "uncertainty_assessment"
    }
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
        "direct_report": "generate_report"
    }
)

# 深度检索后进入报告生成
builder.add_edge("deep_retrieval", "generate_report")

# 直接报告后进入可解释性增强
builder.add_edge("generate_report", "explainable_report")

# 可解释性报告后结束
builder.add_edge("explainable_report", END)

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