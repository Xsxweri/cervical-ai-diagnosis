"""证据收集 Agent 单测。Planner 用 fake 替代，无需 GPU / 真实 LLM。"""

import pytest

import Agents.evidence as ea
from Agents.evidence import (
    CollectAction,
    PlannerOutput,
    _should_stop,
    format_evidence_block,
    needs_evidence_agent,
)

BASE_STATE = {
    "risk_level": "medium",
    "risk_scope": "cell",
    "model_uncertainty": 0.45,
    "evidence_sufficiency": 0.67,
    "evidence_relevant_count": 2,
    "cv_result": {
        "total_cells": 2,
        "abnormal_cells": 1,
        "conflict_count": 1,
        "cells": [
            {"final_class": "light_dysplastic", "grading_conflict": True},
            {"final_class": "normal_superficial"},
        ],
    },
}


# ---------------- 启用条件 ----------------

def test_skip_low_uncertainty_no_conflict():
    """低不确定 + 无冲突 → 不启用（快路径保持确定性）。"""
    st = {**BASE_STATE, "model_uncertainty": 0.1,
          "cv_result": {**BASE_STATE["cv_result"], "conflict_count": 0}}
    assert not needs_evidence_agent(st)


def test_enable_on_high_uncertainty():
    assert needs_evidence_agent({**BASE_STATE, "model_uncertainty": 0.75})


def test_enable_on_conflict():
    st = {**BASE_STATE, "model_uncertainty": 0.1}
    assert needs_evidence_agent(st)        # conflict_count=1


# ---------------- 终止判据（代码，不是 LLM）----------------

def test_stop_at_max_steps(monkeypatch):
    monkeypatch.setattr(ea, "MAX_STEPS", 2)
    assert _should_stop(BASE_STATE, {}, 2)


def test_not_stop_when_conflict_cells_unseen():
    """有冲突细胞但还没查 → 不应停。"""
    assert not _should_stop(BASE_STATE, {}, 0)


def test_stop_after_conflict_cells_and_stats(monkeypatch):
    monkeypatch.setattr(ea, "MAX_STEPS", 5)
    calls = {"get_cell_detail": 1, "get_conflict_stats": 1}
    assert _should_stop(BASE_STATE, calls, 1)


def test_no_conflict_never_satisfied_early(monkeypatch):
    """无冲突样本：第 2 条终止永不成立，只能靠步数上限。"""
    monkeypatch.setattr(ea, "MAX_STEPS", 3)
    st = {**BASE_STATE, "cv_result": {**BASE_STATE["cv_result"], "conflict_count": 0}}
    assert not _should_stop(st, {"get_cell_detail": 2}, 1)


# ---------------- 预算 gate ----------------

@pytest.mark.asyncio
async def test_budget_exceeded_does_not_raise(monkeypatch):
    """超预算应被 gate 拒绝并记录，不抛异常。

    否则一次越界会让整份报告回退成模板。
    """
    monkeypatch.setattr(ea, "MAX_STEPS", 5)
    # conflict 为空 → 只靠 MAX_STEPS 终止
    st = {**BASE_STATE, "cv_result": {**BASE_STATE["cv_result"],
                                      "conflict_count": 1}}

    class FakePlanner:
        async def ainvoke(self, _prompt):
            return PlannerOutput(
                done=False,
                action=CollectAction(tool="get_evidence_status", args={}),
            )

    class FakeLLM:
        def with_structured_output(self, _model, **_kwargs):
            return FakePlanner()

    monkeypatch.setattr(ea, "get_llm", lambda: FakeLLM())
    out = await ea.collect_evidence(st)

    # get_evidence_status 的 budget 是 1 → 第 2 次起被 gate 拦
    gated = [i for i in out["items"] if "[gate]" in i["result"]]
    assert gated, "超预算调用应被记录为 [gate]"


# ---------------- 格式化 ----------------

def test_format_block_empty():
    assert format_evidence_block([]) == ""


def test_format_block_has_no_citation_prefix():
    """工具结果用 [S#]，不参与 validator 的 [ECMGH]# 引用检查。"""
    block = format_evidence_block([
        {"tool": "get_conflict_stats", "args": {}, "reason": "看冲突", "result": "冲突 1 个"}
    ])
    assert "[S1]" in block
    assert "不得据此修改风险等级" in block


# ---------------- Agent 不改诊断字段 ----------------

@pytest.mark.asyncio
async def test_agent_does_not_mutate_state(monkeypatch):
    """Agent 全程只读：调用后 state 的诊断字段不变。"""
    snapshot = {
        k: v for k, v in BASE_STATE.items() if k != "cv_result"
    }

    class FakePlanner:
        async def ainvoke(self, _prompt):
            return PlannerOutput(done=True)

    class FakeLLM:
        def with_structured_output(self, _model, **_kwargs):
            return FakePlanner()

    monkeypatch.setattr(ea, "get_llm", lambda: FakeLLM())
    out = await ea.collect_evidence(BASE_STATE)
    assert out["stopped_by"] == "planner_done", out

    for k, v in snapshot.items():
        assert BASE_STATE[k] == v, f"字段被改写: {k}"