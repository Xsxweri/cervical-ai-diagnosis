"""轻量 trace：记录节点耗时与 LLM 调用次数。

为什么需要：
    Agent 化的核心风险是"多轮 LLM 调用让延迟/成本失控"。
    没有基线就无法判断 Agent 是改善还是恶化。

设计约束：
    * 不引入外部依赖（不用 OpenTelemetry），单进程内足够；
    * 挂在 state["_trace"]，随 state 传递，不改变任何函数签名；
    * 以 "_" 开头的键（内部用）在 as_dict() 时被过滤。
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Any, Dict, Iterator, List, Optional


class Trace:
    def __init__(self) -> None:
        self.steps: List[Dict[str, Any]] = []
        self.llm_calls = 0
        self.tool_calls = 0
        self._t0 = time.perf_counter()

    @contextmanager
    def node(self, name: str) -> Iterator[None]:
        """记录一个节点的耗时。"""
        entry: Dict[str, Any] = {"node": name, "_t": time.perf_counter()}
        self.steps.append(entry)
        try:
            yield
        finally:
            entry["ms"] = round((time.perf_counter() - entry["_t"]) * 1000, 1)

    def llm(self) -> None:
        self.llm_calls += 1

    def tool(self) -> None:
        self.tool_calls += 1

    def as_dict(self) -> Dict[str, Any]:
        return {
            "total_ms": round((time.perf_counter() - self._t0) * 1000, 1),
            "llm_calls": self.llm_calls,
            "tool_calls": self.tool_calls,
            "steps": [
                {k: v for k, v in s.items() if not k.startswith("_")}
                for s in self.steps
            ],
        }


@contextmanager
def traced(state: Dict[str, Any], name: str, *, llm: bool = False) -> Iterator[None]:
    """便捷包装：state 里有 _trace 就记录，没有就空转（不影响生产路径）。

    用法：
        with traced(state, "generate_report", llm=True):
            response = await chain.ainvoke(...)
    """
    tr: Optional[Trace] = state.get("_trace")
    if tr is None:
        yield
        return
    with tr.node(name):
        if llm:
            tr.llm()
        yield