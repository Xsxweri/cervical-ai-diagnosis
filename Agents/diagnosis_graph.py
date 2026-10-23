from functools import lru_cache

from langgraph.graph import END, StateGraph

from Agents.nodes import (
    clarification_agent,
    deep_retrieval_agent,
    explainable_report_agent,
    generate_report_agent,
    post_retrieval_assessment_agent,
    prepare_clarification_agent,
    skip_explainable_agent,
    uncertainty_assessment_agent,
    unreadable_report_agent,
)
from Agents.routing import decision_router, explainable_router
from Agents.state import DiagnosisState


# # 阈值常量
# HIGH_UNCERTAINTY = UNCERTAINTY_HIGH
# MEDIUM_UNCERTAINTY = UNCERTAINTY_MEDIUM

# 可解释性开关：开启后在可解释性节点为异常细胞生成 Grad-CAM（默认关闭，避免额外 GPU 开销）
# ENABLE_GRADCAM = os.getenv("ENABLE_GRADCAM", "false").strip().lower() in {"1", "true", "yes", "on"}

# ====构建状态机图====
@lru_cache(maxsize=1)
def get_builder() -> StateGraph:
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
    return builder

# ====兼容门面====
# 以下符号仅做再导出，供 agent_api.py / tests 维持原有 import 路径，
# 本模块内部并不使用，故显式忽略 F401。
# 必须放在文件末尾：checkpoint.init_checkpoint_store() 内部延迟导入
# get_builder()，置于顶部会形成循环 import。
from Agents.checkpoint import (  # noqa: E402, F401
    close_checkpoint_store,
    get_graph,
    init_checkpoint_store,
)
from Agents.knowledge import init_kb  # noqa: E402, F401
from Agents.nodes import HIGH_UNCERTAINTY, MEDIUM_UNCERTAINTY  # noqa: E402, F401
from utils.case_evidence import aggregate_case_evidence  # noqa: E402, F401