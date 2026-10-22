from typing import List, Optional, Dict, Any, Literal, TypedDict
from pydantic import BaseModel, Field
from langchain_core.messages import BaseMessage

"""
定义Agent状态机数据结构
"""
class RetrievedKnowledgeItem(BaseModel):
    """知识库条目"""
    content: str = Field(..., description="知识片段内容")
    source: str = Field(..., description="来源文档")
    page: int = Field(..., description="知识所在页码")
    score: float = Field(..., description="知识相关性评分")

class DiagnosisState(TypedDict, total=False):
    """
    诊断流程状态
    """

    # --- 输入层 ---
    image_path: Optional[str]  #图像文件路径
    user_query: Optional[str]  #用户输入的查询/澄清回答
    messages: List[BaseMessage]  #预留消息通道
    conversation_history: List[BaseMessage]  # 多轮对话历史（会话持久化对象）

    # --- 感知层 (CV) ---
    cv_result: Optional[Dict[str, Any]]  #计算机视觉分析结果
    max_uncertainty: Optional[float]  #最大不确定性分数 (0-1)
    needs_clarification: bool  # 是否处于等待用户澄清状态

    # --- 认知层 (RAG) ---
    retrieved_knowledge: List[Dict[str, Any]]  # 检索到的相关知识

    # --- 决策层 (Agent) ---
    risk_level: Optional[Literal["pending", "unreadable", "low", "medium", "high", "emergency"]]  #风险等级
    risk_scope: Optional[Literal["cell", "case"]]  # 风险口径：单细胞提示 / 病例级
    action: Optional[Literal["direct_report", "clarify", "deep_retrieve","re_evaluate", "unreadable"]]  #下一步动作
    clarification_questions: List[str]  # 生成的追问问题列表
    clarification_rounds: int  # 已追问轮次（用于硬性上限，防止无限回环）
    retrieval_done: bool  # RAG检索是否完成（防检索回环）
    decision_reason: Optional[str]  # 决策理由

    # --- 可解释性 ---
    gradcam_path: Optional[str]  #Grad-CAM热力图路径
    morphology_analysis: Optional[Dict[str, Any]]  #形态学特征分析

    # --- 输出层 ---
    diagnosis_report: Optional[str]  #最终诊断报告
    recommendations: List[str]  #建议措施
    references: List[str]  #参考文献/指南

    # --- 流程控制 ---
    current_step: str  #当前步骤标识
    is_complete: bool  #流程是否结束
    error_message: Optional[str]  #错误信息