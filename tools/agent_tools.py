"""
Agent 可调用的工具集

设计约束：
  * 全部**只读** —— 不允许写库、不允许改 state；
  * 返回**给 LLM 读的字符串**，不是结构化对象（LLM 需要可读事实，不是 schema）；
  * 用闭包绑定 state：LangChain tool 本身无状态，state 在创建时注入；
  * 每个工具都有明确的成本标签，供 gate 做预算控制。
"""

from __future__ import annotations

from typing import Any, Dict, List

from langchain_core.tools import tool


def make_cell_tool(state: Dict[str, Any]):
    @tool
    def get_cell_detail(cell_index: int) -> str:
        """
        查看指定细胞的完整分析明细。

        cell_index 从 0 开始（对应报告"细胞明细"表的行号）。
        返回二分类结果、多分类分型与 Top-3 概率、不确定性、
        形态学特征（含可靠性）、分级冲突标记。
        """
        cells = (state.get("cv_result") or {}).get("cells") or []
        if not 0 <= cell_index < len(cells):
            return f"索引越界：共 {len(cells)} 个细胞，有效范围 0–{max(len(cells) - 1, 0)}"

        c = cells[cell_index]
        binary = c.get("binary_result") or {}
        probs = c.get("probabilities") or {}
        top = sorted(probs.items(), key=lambda kv: -kv[1])[:3]
        morph = c.get("morphology") or {}
        reliable = morph.get("measurement_reliable")

        parts = [
            f"细胞 #{cell_index}",
            f"- 二分类: {binary.get('class')} (conf={float(binary.get('confidence') or 0):.3f})",
            f"- 最终分型: {c.get('final_class')} (conf={float(c.get('final_confidence') or 0):.3f})",
            f"- 使用模型: {c.get('model_used')}",
            f"- 不确定性: {float(c.get('uncertainty') or 0):.3f}",
            "- Top-3 概率: " + ("、".join(f"{k}={v:.3f}" for k, v in top) or "无"),
            f"- 核质比: {morph.get('nuclear_cytoplasmic_ratio', 'N/A')}"
            f"（{'可靠' if reliable else '不可靠，不得据此下结论'}）",
            f"- 圆形度: {morph.get('circularity', 'N/A')}",
            f"- 分级冲突: {'是' if c.get('grading_conflict') else '否'}",
        ]
        if c.get("dual_conflict"):
            parts.append("- 双空间冲突: 是（主/辅标签空间严重度不一致）")
        if c.get("auxiliary"):
            parts.append(f"- 辅助空间结果: {c['auxiliary']}")
        return "\n".join(parts)

    return get_cell_detail


def make_guideline_tool(state: Dict[str, Any]):
    @tool
    async def search_guidelines(query: str) -> str:
        """
        按自由文本检索临床指南与文献。

        用于回答"这个特征在指南里怎么定义""ASC-H 的处理原则"等问题。
        返回最多 5 条，含来源与页码。
        """
        from Agents.knowledge import get_kb

        try:
            kb = get_kb()
            # retrieve 接受自由文本，比 retrieve_by_classification 的固定映射更灵活
            items = await kb.retrieve(query, k=5)
        except Exception as e:
            return f"知识库不可用：{e}。请基于已有信息回答，并说明文献未能检索。"

        if not items:
            return "未检索到相关文献。"
        return "\n\n".join(
            f"[{i + 1}] {(it.get('content') or '')[:600]}\n"
            f"    来源: {it.get('source')} 第 {it.get('page')} 页"
            f" (score={float(it.get('score') or 0):.4f})"
            for i, it in enumerate(items)
        )

    return search_guidelines


def make_conflict_tool(state: Dict[str, Any]):
    @tool
    def get_conflict_stats() -> str:
        """
        查看本病例的冲突统计与人工复核判据。

        当二分类判异常但多分类归为良性、或双标签空间严重度不一致时，
        用于解释"为什么要转人工复核"。
        """
        from utils.case_evidence import aggregate_case_evidence

        cv = state.get("cv_result") or {}
        ev = aggregate_case_evidence(cv)
        return (
            f"- 总细胞: {ev.get('total', 0)}\n"
            f"- 异常细胞: {ev.get('abnormal_count', 0)}\n"
            f"- 异常分型: {', '.join(ev.get('abnormal_types') or []) or '无'}\n"
            f"- 分级冲突数: {cv.get('grading_conflict_count', 0)}\n"
            f"- 双空间冲突数: {cv.get('dual_conflict_count', 0)}\n"
            f"- 是否存在冲突证据: {'是' if ev.get('has_conflict') else '否'}\n"
            f"- 需人工复核: {'是' if cv.get('review_required') else '否'}"
        )

    return get_conflict_stats


def make_evidence_tool(state: Dict[str, Any]):
    @tool
    def get_evidence_status() -> str:
        """
        查看本轮诊断的证据充分性与模型不确定性。

        用于回答"为什么说证据不足""这个结论有多可靠"。
        注意：模型不确定性与证据充分性是**两个独立维度**，
        文献多寡不能改变模型对图像的测量。
        """
        cv = state.get("cv_result") or {}
        return (
            f"- 模型不确定性（P90 归一化熵）: "
            f"{float(state.get('model_uncertainty') or cv.get('max_uncertainty') or 0):.3f}\n"
            f"- 证据充分性: "
            f"{state.get('evidence_sufficiency') if state.get('evidence_sufficiency') is not None else '未测量'}\n"
            f"- 达标文献数: {state.get('evidence_relevant_count') or 0}\n"
            f"- 风险等级: {state.get('risk_level')}\n"
            f"- 风险口径: {state.get('risk_scope')}"
        )

    return get_evidence_status


# 工具预算表：Planner 的 gate 据此做拦截
TOOL_SPECS: Dict[str, Dict[str, Any]] = {
    "get_cell_detail":     {"budget": 3, "cost": "cpu"},
    "search_guidelines":   {"budget": 2, "cost": "cpu"},
    "get_conflict_stats":  {"budget": 1, "cost": "cpu"},
    "get_evidence_status": {"budget": 1, "cost": "cpu"},
}


def build_followup_tools(state: Dict[str, Any]) -> List:
    """追问场景的工具集。全部只读 —— 追问不得改变诊断状态。"""
    return [
        make_cell_tool(state),
        make_guideline_tool(state),
        make_conflict_tool(state),
        make_evidence_tool(state),
    ]