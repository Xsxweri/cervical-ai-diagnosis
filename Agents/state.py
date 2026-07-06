"""
定义Agent状态机数据结构
"""

from typing import TypedDict,List,Optional,Dict,Any,Literal, Annotated
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
# from operator import add

class RetrievedKnowledgeItem(TypedDict):
    """知识库条目"""
    content: str
    source: str
    page: int
    score: float

class DiagnosisState(TypedDict, total=False):
    """诊断流程状态"""

    #输入
    image_path: str  # 图像路径
    user_query: Optional[str]  # 用户输入的查询
    conversation_history: Annotated[List[BaseMessage], add_messages]  # 对话历史

    #cv分析结果
    cv_result: Optional[Dict[str, Any]]  # 计算机视觉分析结果

    #不确定性量化(MC Dropout)
    max_uncertainty: Optional[float]  # 最大不确定性
    needs_clarification: bool # 是否需要澄清问题

    # RAG检索结果
    retrieved_knowledge: Optional[List[RetrievedKnowledgeItem]] # 检索到的知识列表，每个知识包含标题、内容、来源等信息

    # Agent决策
    risk_level: Optional[Literal["low", "medium", "high", "emergency"]]  # 风险等级
    action: Optional[Literal["direct_report", "clarify", "deep_retrieve"]]  # 代理动作
    clarification_questions: Optional[List[str]] # 追问问题列表

    # 可解释性
    gradcam_path: Optional[str] # Grad-CAM可视化图像路径
    morphology_analysis: Optional[Dict[str, Any]] # 形态学分析结果

    # 输出
    diagnosis_report: Optional[str] # 诊断报告文本
    recommendations: Optional[List[str]] # 推荐措施列表
    references: Optional[List[str]] # 参考文献列表
    error_message: Optional[str]  # 全局异常兜底信息

    # 流程控制
    # messages: Annotated[List[Any], add_messages]  # 对话消息
    current_step: str # 当前步骤
    is_complete: bool # 是否完成诊断流程