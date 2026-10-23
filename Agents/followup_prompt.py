"""
Followup Prompt 构造

- 输入：会话状态
- 输出：追问 Prompt
"""

from typing import Any, Dict

from Prompts import raw, render

_DEFAULT_TIME_LOCATION = "<time_location>\n当前时间：未知\n当前地点：未知\n</time_location>"

def build_followup_system() -> str:
    """取系统提示词"""
    return raw("followup_system")

def build_followup_prompt(
    state: Dict[str, Any],
    user_input: str,
    time_location: str = "",
) -> str:
    """
    会话状态压成追问 Prompt
    """
    ctx_time_location = time_location or _DEFAULT_TIME_LOCATION

    report_snippet = (state.get("diagnosis_report") or "")[:1200]
    recommendations = "\n".join(f"- {r}" for r in (state.get("recommendations") or []))
    risk_level = state.get("risk_level") or "未知"
    risk_scope = state.get("risk_scope") or "cell"

    # 细胞级证据
    cv = state.get("cv_result") or {}
    cells = cv.get("cells") or []
    n_total = int(cv.get("total_cells", 0) or 0)
    n_abn = int(cv.get("abnormal_cells", 0) or 0)

    class_dist: Dict[str, int] = {}
    for c in cells:
        k = c.get("final_class")
        if k is None:
            continue
        class_dist[k] = class_dist.get(k, 0) + 1
    dist_text = "、".join(f"{k}:{v}" for k, v in sorted(class_dist.items())) or "无"

    # 文献证据
    knowledge = state.get("retrieved_knowledge") or []
    kb_text = "\n".join(
        f"- {(d.get('content') or '')[:150]}({d.get('source', '未知')} 第{d.get('page', 0)}页)"
        for d in knowledge[:3]
    ) or "（本次未检索到文献依据）"

    return render(
        "followup_user",
        time_location=ctx_time_location,
        risk_level=risk_level,
        risk_scope=risk_scope,
        report_snippet=report_snippet,
        recommendations=recommendations,
        total_cells=n_total,
        abnormal_cells=n_abn,
        class_distribution=dist_text,
        uncertainty=f"{float(state.get('max_uncertainty') or 0.0):.3f}",
        knowledge=kb_text,
        user_input=user_input,
    )