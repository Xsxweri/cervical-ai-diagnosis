from typing import Dict, Any, Literal,List
from functools import partial
from Agents.state import DiagnosisState
from services.rag_service import MedicalKnowledgeBase
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph,END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import interrupt, Command
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage,SystemMessage
from langchain_core.prompts import ChatPromptTemplate
import os
from langchain_core.runnables import RunnableConfig
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
    '''根据不确定性评估结果路由'''
    return state.get("action","direct_report")

# ====节点3、主动追问====
def clarification_agent(state: DiagnosisState) -> Dict[str, Any]:
    '''高不确定性时主动追问关键信息'''
    cv_result = state.get("cv_result", {}) or {}
    llm = get_llm()
    structured_llm = llm.with_structured_output(ClarificationResponse)

    # 基于细胞特征生成追问问题
    prompt = f'''
    你是一位病理科AI助手。检测到以下细胞图像的不确定性较高(uncertainty={state["max_uncertainty"]:.3f}),
    需要向医生/患者追问关键信息以辅助诊断。

    检测结果:
    - 总细胞数: {cv_result.get("total_cells", 0)}
    - 异常细胞数: {cv_result.get("abnormal_cells", 0)}
    - 最高风险: {cv_result.get("highest_risk", "未知")}

    请生成2-3个最关键的追问问题, 帮助降低诊断不确定性。
    '''

    result = structured_llm.invoke(prompt)

    # 使用 interrupt 暂停图的执行，并将问题发送给前端/用户
    # 当用户回复后，图会从这里恢复，并将用户的回复作为 interrupt 的返回值
    user_response = interrupt({
        "questions": result.questions,
        "reasoning": result.reasoning,
        "message": "诊断不确定性较高，需要补充信息..."
    })

    # 用户回复后，将用户的回答追加到对话历史中
    history = state.get("conversation_history", [])
    history.append(HumanMessage(content=user_response))

    # 返回更新后的状态，并重置 action 为重新评估
    return {
        "conversation_history": history,
        "user_query": user_response,
        "action": "deep_retrieve",
        "current_step": "clarification_resolved"
    }


# ====节点4、RAG检索====
def deep_retrieval_agent(state: DiagnosisState,kb: MedicalKnowledgeBase) -> Dict[str, Any]:
    '''中等不确定性时进行深度知识检索'''
    cv_result = state.get("cv_result", {}) or {}
    classification = cv_result.get("highest_risk", "")

    # 1. 基于分类检索
    class_knowledge = kb.retrieve_by_classification(classification, k=5)

    # 2. 基于形态学特征检索 (利用现有cell_analyzer)
    cells = cv_result.get("cells", [])
    if cells:
        morphology = cells[0].get("morphology", {})
        morph_knowledge = kb.retrieve_by_morphology(morphology, k=3)
    else:
        morph_knowledge = []

    all_knowledge = class_knowledge + morph_knowledge

    return {
        **state,
        "retrieved_knowledge": all_knowledge,
        "current_step": "deep_retrieval"
    }

# ====节点5、直接报告====
def generate_report_agent(state: DiagnosisState) -> Dict[str, Any]:
    '''低不确定性时直接生成诊断报告'''
    cv_results = state.get("cv_result", {})
    cells = cv_results.get("cells", [])
    first_cell = cells[0] if cells else {}
    morphology=first_cell.get("morphology", {})

    # 构建诊断上下文
    context = f'''
    细胞检测结果:
    - 总细胞数: {cv_results.get("total_cells", 0)}
    - 异常细胞数: {cv_results.get("abnormal_cells", 0)}
    - 最高风险分类: {cv_results.get("highest_risk", "")}
    - 模型置信度: {first_cell.get("confidence", 0):.3f}
    - 不确定性: {state.get("max_uncertainty", 0):.3f}

    形态学分析:
    - 核质比: {morphology.get('nuclear_cytoplasmic_ratio', 0):.3f}
    - 圆形度: {morphology.get('circularity', 0):.3f}
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
        response = chain.invoke({"context": context})

        # 生成建议
        recommendations = {
            "low": ["建议1年后常规复查", "保持健康生活方式"],
            "medium": ["建议6个月后复查细胞学", "可考虑HPV检测"],
            "high": ["建议3个月内阴道镜检查", "HPV分型检测,必要时活检"],
            "emergency": ["建议立即阴道镜检查", "尽快预约妇科肿瘤专科, 准备组织病理学检查"]
        }.get(state["risk_level"], ["建议咨询专业医生"])

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
def explainable_report_agent(state: DiagnosisState) -> Dict[str, Any]:
    '''
    生成带可解释性的综合报告
    结合Grad-CAM + 形态学分析 + LLM解释
    '''
    if not state.get("retrieved_knowledge"):
        return {
            "current_step": "explainable_report",
            # "is_complete": False
        }

    cv_results = state.get("cv_result", {})

    # 构建可解释性上下文
    explain_context = f'''
    诊断结果: {state.get("diagnosis_report", "")}

    可解释性信息:
    1. 模型关注区域: AI主要关注细胞的核区域和细胞质边界
    2. 关键形态特征:
    '''

    for i, cell in enumerate(cv_results.get("cells", [])[:3], 1):
        morph = cell.get("morphology", {})
        explain_context += f'''
    细胞{i}:
    - 核质比: {morph.get("nuclear_cytoplasmic_ratio", 0):.3f}
    - 圆形度: {morph.get("circularity", 0):.3f}
    - 解读: {morph.get("interpretation", "")}
    '''

    explain_context += f'''
    3. 不确定性分析: MC Dropout标准差={state.get("max_uncertainty", 0):.3f}

    4. 医学依据:
    '''

    for doc in state.get("retrieved_knowledge", [])[:3]:
        explain_context += f'- {doc.get("content","")[:150]}...\n'

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

    llm = get_llm()
    chain = explain_prompt | llm
    response = chain.invoke({"context": explain_context})

    # 在原有报告基础上追加可解释性
    full_report = state.get("diagnosis_report", "") + "\n\n---\n\n【AI诊断逻辑解释】\n" + response.content

    return {
        **state,
        "diagnosis_report": full_report,
        "current_step": "explainable_report",
        "is_complete": True
    }


# ====节点7、构建状态机图====
kb = MedicalKnowledgeBase(persist_dir="./knowledge_db")

import glob
pdf_dir = "./Cervical_cytology"
pdf_files = glob.glob(f"{pdf_dir}/*.pdf")
if pdf_files:
    print(f"[RAG] 发现 {len(pdf_files)} 篇医学文献，正在构建知识库...")
    try:
        chunk_count = kb.load_documents(pdf_files)
        print(f"[RAG] 知识库构建完成，共 {chunk_count} 个文本块")
    except Exception as e:
        print(f"[RAG] 知识库加载失败: {e}")
else:
    print(f"[RAG] 未在 {pdf_dir}/ 中找到 PDF 文献，RAG 检索将返回空结果")

builder = StateGraph(DiagnosisState)

# 添加节点
builder.add_node("uncertainty_assessment", uncertainty_assessment_agent)
builder.add_node("clarification", clarification_agent)
builder.add_node("deep_retrieval", partial(deep_retrieval_agent,kb=kb))
builder.add_node("generate_report", generate_report_agent)
builder.add_node("explainable_report", explainable_report_agent)

# 设置入口
builder.set_entry_point("uncertainty_assessment")

# 添加条件边
builder.add_conditional_edges(
    "uncertainty_assessment",
    decision_router,
    {
        "clarify": "clarification",
        "deep_retrieve": "deep_retrieval",
        "direct_report": "generate_report",
        "re_evaluate": "uncertainty_assessment" # 用户回答后，重新进入评估节点
    }
)

# 深度检索后进入报告生成
builder.add_edge("deep_retrieval", "generate_report")

# 直接报告后进入可解释性增强
builder.add_edge("generate_report", "explainable_report")

# # 澄清后结束 (等待用户输入)
# builder.add_edge("clarification", END)

# 可解释性报告后结束
builder.add_edge("explainable_report", END)

# 必须使用 MemorySaver 来支持多轮对话的持久化
memory = MemorySaver()
graph = builder.compile(checkpointer= memory)