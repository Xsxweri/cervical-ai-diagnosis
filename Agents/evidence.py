"""有界证据收集 Agent。

与"全自主 ReAct"的三处关键差异：

  1. LLM 不决定终止 —— `_should_stop()` 是代码判据（冲突细胞都看过 / 步数上限）。
     "LLM 觉得够了"不是终止条件。
  2. LLM 不决定路由 —— Planner 只能选"继续查"，不能选"出报告/追问/转复核"。
     那些仍由 `risk_level` + `conflict` + `rounds` 决定（见 diagnosis_graph）。
  3. 证据只追加进报告 prompt, **不改写** risk_level / cv_result。
     「Agent 可以多查证据，但不能改写模型对图像的测量」是规则。

预算控制：
    TOOL_SPECS 为每个工具设了 budget。超预算的调用被 gate 拒绝并记入 trace,
    不抛异常 —— 否则一次越界就让整份报告回退成模板。
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field

from Agents.llm import get_llm
from tools.agent_tools import TOOL_SPECS, build_followup_tools
from utils.trace import traced

logger = logging.getLogger("evidence_agent")

# 由 Phase 0 实测决定：P90 total_ms = 20.1s，单步增量 ~1.3s → 3 步后约 24s
MAX_STEPS = int(os.getenv("EVIDENCE_AGENT_MAX_STEPS", "3"))

# 触发阈值：只在中/高不确定或存在冲突时启用。
# 低风险快路径保持确定性 —— Agent 不该为"已经确定的结论"付额外成本。
ENABLE_BELOW_UNCERTAINTY = float(os.getenv("EVIDENCE_AGENT_UNCERTAINTY", "0.30"))


class CollectAction(BaseModel):
    """LLM 只能选"继续收集"，不能选"结束"或"出报告"。"""

    tool: Literal[
        "get_cell_detail",
        "search_guidelines",
        "get_conflict_stats",
        "get_evidence_status",
    ] = Field(description="要调用的工具名")
    args: Dict[str, Any] = Field(
        default_factory=dict,
        description="工具参数。get_cell_detail 需 {'cell_index': int}；"
                    "search_guidelines 需 {'query': str}；其余传 {}",
    )
    reason: str = Field(default="", description="为什么查这个（一句话）")


class PlannerOutput(BaseModel):
    """done=True 表示"我收集完了"，但不代表可以出报告。"""

    done: bool = Field(description="证据是否已足够，不必再查")
    action: Optional[CollectAction] = Field(
        default=None, description="done=False 时必须给出下一步动作"
    )


def needs_evidence_agent(state: Dict[str, Any]) -> bool:
    """是否启用 Agent。灰色地带才启用，低风险快路径跳过。"""
    cv = state.get("cv_result") or {}
    u = float(state.get("model_uncertainty") or cv.get("max_uncertainty") or 0.0)
    conflict = int(cv.get("conflict_count") or 0)
    return u > ENABLE_BELOW_UNCERTAINTY or conflict > 0


def _conflict_cell_indices(cv: Dict[str, Any]) -> List[int]:
    """存在分级/双空间冲突的细胞索引。用于判断"是否已看过所有冲突细胞"。"""
    return [
        i for i, c in enumerate(cv.get("cells") or [])
        if c.get("grading_conflict") or c.get("dual_conflict")
    ]


def _should_stop(state: Dict[str, Any], calls: Dict[str, int], step: int) -> bool:
    """代码判据。**LLM 无权决定终止**。

    两个终止条件：
      1. 达到步数上限（预算硬限）
      2. 已看过全部冲突细胞（最多 2 个）+ 已查冲突统计
         对于无冲突样本，第 2 条永不成立，靠第 1 条兜底。
    """
    if step >= MAX_STEPS:
        return True

    conflict_idx = _conflict_cell_indices(state.get("cv_result") or {})
    if not conflict_idx:
        return False

    cells_seen = calls.get("get_cell_detail", 0)
    return (
        cells_seen >= min(len(conflict_idx), 2)
        and calls.get("get_conflict_stats", 0) >= 1
    )


def _build_planner_prompt(state: Dict[str, Any], calls: Dict[str, int]) -> str:
    cv = state.get("cv_result") or {}
    cells = cv.get("cells") or []
    conflict_idx = _conflict_cell_indices(cv)

    # 只列出"还没看过"的细胞索引，避免 LLM 重复查同一个
    seen_note = f"已调用 {calls}" if calls else "尚未调用任何工具"
    return f"""你是宫颈细胞学诊断的证据收集助手。

## 当前病例
- 风险等级：{state.get("risk_level")}
- 风险口径：{state.get("risk_scope")}（cell=单细胞提示，不代表病例整体）
- 细胞数：{len(cells)}，异常 {cv.get("abnormal_cells", 0)} 个
- 模型不确定性（P90 归一化熵）：{float(state.get("model_uncertainty") or cv.get("max_uncertainty") or 0):.3f}
- 证据充分性：{state.get("evidence_sufficiency") if state.get("evidence_sufficiency") is not None else "未测量"}
- 冲突细胞索引：{conflict_idx or "无"}

## 可用工具
- get_cell_detail(cell_index)：查看单个细胞的完整明细（分型/概率/不确定性/形态学可靠性/冲突标记）
- search_guidelines(query)：按自由文本检索临床指南与文献
- get_conflict_stats()：查看冲突统计与人工复核判据
- get_evidence_status()：查看三个维度的当前取值

## 状态
{seen_note}

## 你的任务
如果现有信息已足够支撑报告，返回 done=true。
否则选一个工具继续查 —— 优先查冲突细胞的明细，或检索与当前分型相关的指南。

注意：你**不能**改变风险等级，也不能直接产出诊断结论。
你只负责补齐事实，最终结论由系统规则决定。

## 输出格式（必须是合法 JSON）
只返回一个 JSON 对象，不要有任何其它文字、不要用 Markdown 代码块包裹：

{{"done": false, "action": {{"tool": "get_cell_detail", "args": {{"cell_index": 0}}, "reason": "查看冲突细胞明细"}}}}

- done=true 表示不必再查，此时 action 可为 null
- done=false 时 action 必须给出
- tool 只能是: get_cell_detail / search_guidelines / get_conflict_stats / get_evidence_status
- get_cell_detail 需要 args={{"cell_index": 整数}}; search_guidelines 需要 args={{"query": "检索词"}}; 其余工具 args 传 {{}}
"""


async def collect_evidence(state: Dict[str, Any]) -> Dict[str, Any]:
    """有界收集证据。返回 {"items": [...], "steps": n, "stopped_by": str}。

    全程只读 —— 不改 state 的任何诊断字段。
    """
    if not needs_evidence_agent(state):
        return {"items": [], "steps": 0, "stopped_by": "skipped"}

    tools = {t.name: t for t in build_followup_tools(state)}
    calls: Dict[str, int] = {}
    items: List[Dict[str, Any]] = []
    tr = state.get("_trace")
    stopped_by = "max_steps"

    try:
        planner = get_llm().with_structured_output(PlannerOutput, method="function_calling")
    except Exception as e:
        logger.error("Planner 初始化失败: %s", e)
        return {"items": [], "steps": 0, "stopped_by": "init_error"}

    for step in range(MAX_STEPS):
        if _should_stop(state, calls, step):
            stopped_by = "satisfied" if step < MAX_STEPS else "max_steps"
            break

        try:
            with traced(state, f"planner_{step}", llm=True):
                plan: PlannerOutput = await planner.ainvoke(
                    _build_planner_prompt(state, calls)
                )
        except Exception as e:
            logger.warning("Planner 第 %d 步失败，提前结束: %s", step, e)
            stopped_by = "planner_error"
            break

        if plan.done or plan.action is None:
            stopped_by = "planner_done"
            break

        name = plan.action.tool
        spec = TOOL_SPECS.get(name)
        if spec is None:
            logger.warning("拒绝未注册工具: %s", name)
            continue
        if calls.get(name, 0) >= spec["budget"]:
            # 超预算不抛异常 —— 记一条 system 项，让 LLM 下一步换工具
            items.append({
                "tool": name,
                "args": plan.action.args,
                "reason": plan.action.reason,
                "result": f"[gate] 工具预算用尽（上限 {spec['budget']} 次），已跳过",
            })
            continue

        try:
            with traced(state, f"tool_{name}"):
                if tr is not None:
                    tr.tool()
                result = await tools[name].ainvoke(plan.action.args)
        except Exception as e:
            result = f"工具执行失败: {type(e).__name__}: {e}"

        calls[name] = calls.get(name, 0) + 1
        items.append({
            "tool": name,
            "args": plan.action.args,
            "reason": plan.action.reason,
            "result": str(result)[:1000],   # 截断：控制注入 prompt 的长度
        })

    logger.info(
        "证据收集结束：step=%d calls=%s stopped_by=%s",
        len(items), calls, stopped_by,
    )
    return {"items": items, "steps": len(items), "stopped_by": stopped_by}


def format_evidence_block(items: List[Dict[str, Any]]) -> str:
    """把收集到的证据格式化成报告 prompt 可用的文本块。

    刻意**不带 E#/C# 这类引用标记**：
        validator 的 _CITE_RE 只认 [ECMGH] 前缀，这里用 [S1] 形式
        既不干扰引用检查，也避免 LLM 把工具结果当成"文献引用"。
    """
    if not items:
        return ""
    parts = ["## 补充证据（证据收集 Agent，仅供解释，不改变系统结论）"]
    for i, it in enumerate(items, 1):
        parts.append(f"[S{i}] {it['tool']}({it['args']}) — {it['reason'] or '无说明'}\n{it['result']}")
    parts.append(
        "\n以上证据只用于解释已有结论。**不得据此修改风险等级或作出新的诊断判断。**"
    )
    return "\n\n".join(parts)