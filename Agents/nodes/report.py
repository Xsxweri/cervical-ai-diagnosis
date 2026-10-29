"""节点6-8:报告生成、不可判读终止、无 LLM 结构化摘要。"""

from typing import Any, Dict

from langchain_core.prompts import ChatPromptTemplate

from Agents.llm import get_llm
from Agents.state import DiagnosisState
from Prompts import raw
from Prompts.clinical_content import (
    RECOMMENDATION_CELL_SCOPE,
    RECOMMENDATION_DEFAULT,
    RECOMMENDATIONS_BY_RISK,
    REPORT_FALLBACK,
)
from utils.case_evidence import aggregate_case_evidence


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

    
    try:
        llm = get_llm()
        chain = report_prompt | llm
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


def skip_explainable_agent(state: DiagnosisState) -> Dict[str, Any]:
    """低风险清晰病例：跳过二次 LLM,只附加结构化摘要（零 LLM 成本）。"""
    ev = aggregate_case_evidence(state.get("cv_result") or {})
    dist = ev.get("class_distribution") or {}
    summary = (
        "\n\n## 结构化摘要\n\n"
        f"- 检出细胞 {ev.get('total', 0)} 个，异常 {ev.get('abnormal_count', 0)} 个\n"
        f"- 分类分布：{'、'.join(f'{k}:{v}' for k, v in sorted(dist.items())) or '无'}\n"
        f"- 不确定性(P90 归一化熵): {(state.get('max_uncertainty') or 0.0):.3f}\n"
        f"- 风险等级：{state.get('risk_level')}(口径 {state.get('risk_scope')})"
    )
    return {
        "diagnosis_report": (state.get("diagnosis_report") or "") + summary,
        "current_step": "skip_explainable",
        "is_complete": True,
    }