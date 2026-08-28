from typing import List, Optional, Dict, Any, Literal
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

class DiagnosisState(BaseModel):
    """
    诊断流程状态
    """

    # --- 输入层 ---
    image_path: Optional[str] = Field(None, description="图像文件路径")
    user_query: Optional[str] = Field(None, description="用户输入的查询")

    # LangGraph 会自动处理消息的追加，但我们需要一个初始空列表
    messages: List[BaseMessage] = Field(default_factory=list)

    # --- 感知层 (CV) ---
    cv_result: Optional[Dict[str, Any]] = Field(None, description="计算机视觉分析结果 (JSON)")
    max_uncertainty: Optional[float] = Field(None, description="最大不确定性分数 (0-1)")
    needs_clarification: bool = Field(False, description="标记是否需要向用户追问")

    # --- 认知层 (RAG) ---
    retrieved_knowledge: List[RetrievedKnowledgeItem] = Field(default_factory=list, description="检索到的相关知识")

    # --- 决策层 (Agent) ---
    risk_level: Optional[Literal["low", "medium", "high", "emergency"]] = Field(None, description="风险等级")
    action: Optional[Literal["direct_report", "clarify", "deep_retrieve"]] = Field(None, description="下一步动作")
    clarification_questions: List[str] = Field(default_factory=list, description="生成的追问问题列表")

    # --- 可解释性 ---
    gradcam_path: Optional[str] = Field(None, description="Grad-CAM热力图路径")
    morphology_analysis: Optional[Dict[str, Any]] = Field(None, description="形态学特征分析")

    # --- 输出层 ---
    diagnosis_report: Optional[str] = Field(None, description="最终诊断报告")
    recommendations: List[str] = Field(default_factory=list, description="建议措施")
    references: List[str] = Field(default_factory=list, description="参考文献/指南")

    # --- 流程控制 ---
    current_step: str = Field("start", description="当前步骤标识")
    is_complete: bool = Field(False, description="流程是否结束")
    error_message: Optional[str] = Field(None, description="错误信息")

    # --- 辅助方法 (可选) ---
    def add_message(self, message: BaseMessage):
        """辅助方法：添加消息"""
        self.messages.append(message)