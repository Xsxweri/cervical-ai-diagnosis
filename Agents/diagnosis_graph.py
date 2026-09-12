from typing import Dict, Any, List
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
import os
import asyncio
from dotenv import load_dotenv

load_dotenv()

# LLM
def get_llm():
    return ChatOpenAI(
        model=os.getenv("LLM_MODEL_ID"),
        temperature=0.3,
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("LLM_BASE_URL"),
        timeout=30.0
    )

# 结构化输出模型（替代 json.loads）
class ClarificationResponse(BaseModel):
    questions: List[str] = Field(description="2-3个追问问题")
    reasoning: str = Field(description="追问理由")

# 阈值常量
HIGH_UNCERTAINTY = 0.3
MEDIUM_UNCERTAINTY = 0.15

# ====节点1、不确定性评估====
def uncertainty_assessment_agent(state: DiagnosisState) -> Dict[str, Any]:
    '''利用MC Dropout结果评估不确定性'''
    cv_result = state.get("cv_result", {}) or {}
    max_uncertainty = cv_result.get("max_uncertainty", 0.0)

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
            "current_step": "uncertainty_assessment"
        }
    elif max_uncertainty > MEDIUM_UNCERTAINTY:
        # 中等不确定性: 检索更多知识
        return {
            "max_uncertainty": max_uncertainty,
            "needs_clarification": False,
            "action": "deep_retrieve",
            "risk_level": "medium",
            "current_step": "uncertainty_assessment"
        }
    else:
        # 低不确定性: 直接生成报告
        abnormal_count = cv_result.get("abnormal_cells", 0)
        total_count = max(cv_result.get("total_cells", 1),1)
        ratio = abnormal_count / total_count

        if ratio == 0:
            risk = "low"
        elif ratio < 0.3:
            risk = "medium"
        elif ratio < 0.6:
            risk = "high"
        else:
            risk = "emergency"

        return {
            "max_uncertainty": max_uncertainty,
            "needs_clarification": False,
            "action": "direct_report",
            "risk_level": risk,
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
    prompt = f'''
    你是一位病理科AI助手。检测到以下细胞图像的不确定性较高(uncertainty= {(state.get("max_uncertainty") or 0.0):.3f}),
    需要向医生/患者追问关键信息以辅助诊断。

    检测结果:
    - 总细胞数: {cv_result.get("total_cells", 0)}
    - 异常细胞数: {cv_result.get("abnormal_cells", 0)}
    - 最高风险: {cv_result.get("highest_risk", "未知")}

    请生成2-3个最关键的追问问题, 帮助降低诊断不确定性。
    '''

    result = await structured_llm.ainvoke(prompt)
    return {
         "clarification_questions": result.questions,
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
       tasks.append(kb.retrieve_by_morphology(cells[0].get("morphology") or {}, k=3))

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

    # 构建诊断上下文
    context = f'''
    细胞检测结果:
    - 总细胞数: {cv_results.get("total_cells", 0)}
    - 异常细胞数: {cv_results.get("abnormal_cells", 0)}
    - 最高风险分类: {cv_results.get("highest_risk", "")}
    - 模型置信度: {(first_cell.get("final_confidence") or 0.0):.3f}
    - 不确定性: {(state.get("max_uncertainty") or 0.0):.3f}

    形态学分析:
    - 核质比: {(morphology.get('nuclear_cytoplasmic_ratio') or 0.0):.3f}
    - 圆形度: {(morphology.get('circularity') or 0.0):.3f}
    '''

    report_prompt = ChatPromptTemplate.from_messages([
        ("system", '''你是一位资深病理科医生。请根据AI分析结果生成诊断报告。
        报告要求:
        1. 说明AI检测到的细胞数量和异常比例
        2. 解释风险等级判断依据
        3. 给出明确的后续建议 (复查时间、进一步检查项目)
        4. 说明模型置信度, 建议高不确定性病例人工复核
        5. 语气专业、客观、安抚患者'''),
        ("human", "{context}\n\n请生成诊断报告。")
    ])

    llm = get_llm()
    chain = report_prompt | llm

    try:
        response = await chain.ainvoke({"context": context})

        # 生成建议
        recommendations = {
            "low": ["建议1年后常规复查", "保持健康生活方式"],
            "medium": ["建议6个月后复查细胞学", "可考虑HPV检测"],
            "high": ["建议3个月内阴道镜检查", "HPV分型检测,必要时活检"],
            "emergency": ["建议立即阴道镜检查", "尽快预约妇科肿瘤专科, 准备组织病理学检查"]
        }.get(state.get("risk_level"), ["建议咨询专业医生"])

        return {
            "diagnosis_report": response.content,
            "recommendations": recommendations,
            "current_step": "generate_report",
            "is_complete": True
        }
    except Exception as e:  # 捕获所有异常
        print(f"LLM生成报告失败: {e}")
        return {
            "diagnosis_report": "抱歉，AI医生暂时无法生成详细报告，请稍后重试。",
            "recommendations": ["建议咨询专业医生"],
            "current_step": "generate_report",
            "is_complete": True
        }

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

    explain_prompt = ChatPromptTemplate.from_messages([
        ("system", '''你是一位病理科AI解释专家。请用通俗语言解释AI的诊断逻辑。
        要求:
        1. 解释模型"看到了什么"导致这个判断
        2. 指出哪些形态特征是关键证据
        3. 说明不确定性来源
        4. 引用医学文献支持诊断
        5. 让患者理解但不过度担忧'''),
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

## 设置 Redis Checkpoint
_checkpointer_cm = AsyncRedisSaver.from_conn_string(REDIS_URL)
_exit_stack = AsyncExitStack()
kb: MedicalKnowledgeBase = None
graph = None

def get_kb() -> MedicalKnowledgeBase:
    if kb is None:
        raise RuntimeError("知识库未初始化：请先调用init_kb()")
    return kb

async def init_kb():
    """延迟构建知识库（加载嵌入模型 + PDF），只在应用启动时调用一次"""
    global kb
    import glob
    kb = MedicalKnowledgeBase(persist_dir="./knowledge_db")
    pdf_files = glob.glob("./Cervical_cytology/*.pdf")
    if pdf_files:
        print(f"[RAG] 发现 {len(pdf_files)} 篇医学文献，正在构建知识库...")
        try:
            chunk_count = await kb.load_documents(pdf_files)
            print(f"[RAG] 知识库构建完成，共 {chunk_count} 个文本块")
        except Exception as e:
            print(f"[RAG] 知识库加载失败: {e}")
    else:
        print("[RAG] 未找到 PDF 文献，RAG 检索将返回空结果")

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