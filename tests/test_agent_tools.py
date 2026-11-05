"""工具层单测。纯函数，无需 GPU / LLM / PG。"""

from tools.agent_tools import (
    TOOL_SPECS,
    build_followup_tools,
    make_cell_tool,
    make_conflict_tool,
    make_evidence_tool,
)

STATE = {
    "risk_level": "medium",
    "risk_scope": "case",
    "model_uncertainty": 0.42,
    "evidence_sufficiency": 0.67,
    "evidence_relevant_count": 2,
    "cv_result": {
        "total_cells": 2,
        "abnormal_cells": 1,
        "grading_conflict_count": 1,
        "dual_conflict_count": 0,
        "review_required": True,
        "cells": [
            {
                "binary_result": {"class": "abnormal", "confidence": 0.91},
                "final_class": "light_dysplastic",
                "final_confidence": 0.88,
                "model_used": "multiclass",
                "uncertainty": 0.31,
                "probabilities": {"light_dysplastic": 0.88, "normal": 0.08},
                "morphology": {
                    "nuclear_cytoplasmic_ratio": 0.52,
                    "circularity": 0.77,
                    "measurement_reliable": True,
                },
                "grading_conflict": True,
            },
            {
                "binary_result": {"class": "normal", "confidence": 0.85},
                "final_class": "normal_superficial",
                "final_confidence": 0.85,
                "model_used": "multiclass",
                "uncertainty": 0.12,
                "probabilities": {},
                "morphology": {"measurement_reliable": False},
            },
        ],
    },
}


# ---------------- get_cell_detail ----------------

def test_cell_detail_normal():
    t = make_cell_tool(STATE)
    out = t.invoke({"cell_index": 0})
    assert "light_dysplastic" in out
    assert "0.910" in out          # 二分类置信度
    assert "可靠" in out
    assert "分级冲突: 是" in out


def test_cell_detail_unreliable_morphology():
    t = make_cell_tool(STATE)
    out = t.invoke({"cell_index": 1})
    assert "不可靠" in out


def test_cell_detail_out_of_range():
    t = make_cell_tool(STATE)
    out = t.invoke({"cell_index": 99})
    assert "越界" in out
    assert "有效范围 0–1" in out


# ---------------- get_conflict_stats ----------------

def test_conflict_stats():
    t = make_conflict_tool(STATE)
    out = t.invoke({})
    assert "分级冲突数: 1" in out
    assert "需人工复核: 是" in out


# ---------------- get_evidence_status ----------------

def test_evidence_status():
    t = make_evidence_tool(STATE)
    out = t.invoke({})
    assert "0.420" in out
    assert "0.67" in out
    assert "medium" in out


# ---------------- 注册完整性 ----------------

def test_tool_specs_cover_all_tools():
    """TOOL_SPECS 必须覆盖 build_followup_tools 返回的每个工具。

    否则 Planner 的 gate 会因为 spec 缺失而拒绝调用。
    """
    names = {t.name for t in build_followup_tools(STATE)}
    missing = names - TOOL_SPECS.keys()
    assert not missing, f"缺少预算配置: {missing}"


def test_all_tools_have_description():
    """工具必须带 description，否则模型无法判断何时调用。"""
    for t in build_followup_tools(STATE):
        assert t.description and len(t.description.strip()) > 10, t.name