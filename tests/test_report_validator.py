"""报告校验器单测。纯函数，无需 PG / GPU / Redis。"""

import pytest

from utils.evidence_ledger import EvidenceLedger
from utils.report_validator import has_blocker, validate_report


def _ledger() -> EvidenceLedger:
    led = EvidenceLedger()
    led.add("guideline", "HSIL 需阴道镜评估", "ASCCP p12", score=0.78)
    led.add("guideline", "CIN2 建议锥切", "指南 p5", score=0.82)
    return led


BASE_CTX = {
    "total_cells": 10,
    "risk_level": "low",
    "conflict_count": 0,
    "review_required": False,
    "evidence_sufficiency": 1.0,
    "cells": [],
}


# ---------------- 引用完整性 ----------------

def test_unknown_citation_blocked():
    issues = validate_report("根据 E9 可判断为低风险。", _ledger(), BASE_CTX)
    assert any(i.kind == "unknown_citation" for i in issues)
    assert has_blocker(issues)


def test_known_citation_passes():
    issues = validate_report("根据 E1 与 E2，建议随访。", _ledger(), BASE_CTX)
    assert not any(i.kind == "unknown_citation" for i in issues)


# ---------------- 口径 ----------------

def test_case_scope_allowed_when_enough_cells():
    issues = validate_report("该患者整体风险较低。", _ledger(), BASE_CTX)
    assert not any(i.kind == "scope_violation" for i in issues)


# ---------------- 冲突 ----------------

def test_conflict_must_be_declared():
    ctx = {**BASE_CTX, "conflict_count": 1}
    issues = validate_report("未见异常细胞。", _ledger(), ctx)
    assert any(i.kind == "missing_conflict" for i in issues)


def test_conflict_declared_passes():
    ctx = {**BASE_CTX, "conflict_count": 1}
    issues = validate_report("存在分型冲突，建议人工复核。", _ledger(), ctx)
    assert not any(i.kind == "missing_conflict" for i in issues)


# ---------------- 复核 ----------------

def test_high_risk_needs_review_text():
    ctx = {**BASE_CTX, "risk_level": "high"}
    issues = validate_report("低风险，建议常规随访。", _ledger(), ctx)
    assert any(i.kind == "missing_review" for i in issues)


def test_high_risk_with_review_passes():
    ctx = {**BASE_CTX, "risk_level": "high"}
    issues = validate_report("建议人工镜检复核。", _ledger(), ctx)
    assert not any(i.kind == "missing_review" for i in issues)


# ---------------- 形态学 ----------------

def test_unreliable_morphology():
    ctx = {
        **BASE_CTX,
        "cells": [
            {"morphology": {"measurement_reliable": False}},
            {"morphology": {"measurement_reliable": False}},
            {"morphology": {"measurement_reliable": True}},
        ],
    }
    issues = validate_report("核质比偏高提示恶性倾向。", _ledger(), ctx)
    assert any(i.kind == "unreliable_morphology" for i in issues)


# ---------------- 证据披露（warning）----------------

def test_undisclosed_insufficiency_is_warning_only():
    ctx = {**BASE_CTX, "evidence_sufficiency": 0.0}
    issues = validate_report("低风险。", _ledger(), ctx)
    kinds = {i.kind: i.severity for i in issues}
    assert kinds.get("undisclosed_insufficiency") == "warning"
    assert not has_blocker([i for i in issues
                            if i.kind == "undisclosed_insufficiency"])


# ---------------- 合规报告应通过 ----------------

def test_clean_report_passes():
    report = "检出 1 个高级别细胞（E1、E2），建议人工镜检复核并 3 个月复查。"
    ctx = {**BASE_CTX, "risk_level": "high", "evidence_sufficiency": 1.0}
    assert validate_report(report, _ledger(), ctx) == []


# ---------------- 账本序列化往返 ----------------

def test_ledger_roundtrip():
    led = _ledger()
    back = EvidenceLedger.from_json(led.to_json())
    assert back.ids == led.ids
    assert back.count("guideline") == 2

# ============ 口径检查：三级判定 ============

@pytest.mark.parametrize("text", [
    "该患者属低风险。",
    "该患者整体风险较低。",
    "本病例判定为阴性。",
    "本病例为高风险，需立即处理。",
    "病例整体评估为高风险。",
    "本例患者诊断为高风险。",
])
def test_scope_blocker(text):
    """无局限词 + 病例级主语 + 正向断言 → blocker（真越界）。"""
    ctx = {**BASE_CTX, "total_cells": 1}
    issues = validate_report(text, _ledger(), ctx)
    assert any(
        i.kind == "scope_violation" and i.severity == "blocker" for i in issues
    ), f"漏拦: {text}"


@pytest.mark.parametrize("text", [
    "本病例仅检出 1 个细胞，属低风险，建议常规随访。",
    "该患者仅送检 1 个细胞，病变程度为轻度。",
    "本病例细胞数不足，但整体风险仍为高。",
    "本病例仅检出 1 个细胞，形态学符合良性，该细胞风险等级评估为 low。",
])
def test_scope_warning(text):
    """有局限词但仍有正向断言 → 只提示，不阻断。"""
    ctx = {**BASE_CTX, "total_cells": 1}
    issues = validate_report(text, _ledger(), ctx)
    assert not [i for i in issues if i.severity == "blocker"], f"不应阻断: {text}"
    assert any(i.kind == "scope_wording" for i in issues), f"应提示: {text}"


@pytest.mark.parametrize("text", [
    "本病例仅检出 1 个细胞，属细胞级提示，不足以代表病例整体。",
    "该患者送检视野仅含 1 个细胞，无法代表整体。",
    "本例患者样本细胞数不足，不作病例级判断。",
    "本病例仅检出 1 个细胞，属于极度稀薄样本，严禁据此推断恶性或病例级风险。",
    "因本例仅检出 1 个细胞，样本量严重不足，故不提供病例级 TBS 分级结论。",
])
def test_scope_clean(text):
    """局限声明（断言被否定）→ 不产生任何 scope issue。"""
    ctx = {**BASE_CTX, "total_cells": 1}
    issues = validate_report(text, _ledger(), ctx)
    assert not [i for i in issues if i.kind.startswith("scope_")], \
        f"误伤: {text} → {[i.kind for i in issues]}"


# ---------------- 节过滤：免责声明节不参与口径检查 ----------------

def test_disclaimer_section_not_checked():
    """「局限性与人工复核提示」节里的病例级主语不算越界。

    实测 LLM 会把 prompt 里的口径约束原文抄进这一节
    （"禁止以「该患者」名义得出……"），被关键词规则判成真越界。
    """
    report = (
        "## 关键发现\n"
        "- 风险等级：低（Low）\n"
        "- 该细胞形态符合良性改变。\n\n"
        "## 局限性与人工复核提示\n"
        "- 根据口径约束，禁止以「该患者」或「本病例整体」名义得出确定的风险分级结论。\n"
    )
    issues = validate_report(report, _ledger(), {**BASE_CTX, "total_cells": 1})
    assert not any(i.severity == "blocker" for i in issues), \
        f"误伤免责声明节: {[i.kind for i in issues]}"


def test_conclusion_section_still_checked():
    """「关键发现」节里的真违规仍必须拦。"""
    report = (
        "## 关键发现\n"
        "- 该患者属低风险。\n\n"
        "## 局限性与人工复核提示\n"
        "- 本报告仅供参考。\n"
    )
    issues = validate_report(report, _ledger(), {**BASE_CTX, "total_cells": 1})
    assert any(i.kind == "scope_violation" and i.severity == "blocker" for i in issues)