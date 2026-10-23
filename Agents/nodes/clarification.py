"""节点2-3:主动追问与 interrupt 恢复。"""

from typing import Any, Dict

from langchain_core.messages import HumanMessage
from langgraph.types import interrupt

from Agents.llm import ClarificationResponse, get_llm
from Agents.state import DiagnosisState
from Prompts import render
from Prompts.clinical_content import CLARIFICATION_FALLBACK_QUESTIONS
from utils.uncertainty import MAX_CLARIFICATION_ROUNDS


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