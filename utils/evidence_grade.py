"""
证据等级标定（CEBM 简化版）。

映射来源：doc_type 由 configs/corpus_manifest.yaml 人工标注（可用 RAG_CORPUS_MANIFEST 环境变量覆盖路径）。
本模块只做「文档类型 → 证据等级」的确定性映射，无 IO、无 LLM。

## 为什么需要

当前 evidence_sufficiency 只数条数（n_relevant / 3）：
    3 篇 CNN 分类研究论文  →  1.00
    3 篇国家筛查指南       →  1.00
两者在临床上的可信度完全不同。报告因此会对「只有原始研究支撑」的结论
宣称"证据充分"，属过度自信。

加权后指标才与临床语义一致，且能直接用于报告的证据分级展示。
"""

from __future__ import annotations

# doc_type → (等级, 描述, 权重)
#
# 权重用于 evidence_sufficiency_from 的加权求和：
#   I   级 指南 / 标准 / 术语体系  → 1.0
#   II  级 综述                  → 0.7
#   III 级 原始研究               → 0.5
#
# 未标注（None 或未知类型）→ 1.0：不惩罚历史语料与单测夹具，
# 保证「接入 grade 前后，未标注部分的数值不变」。
_GRADE: dict[str, tuple[str, str, float]] = {
    "national_guideline": ("I", "国家指南", 1.0),
    "intl_guideline":     ("I", "国际指南", 1.0),
    "standard":           ("I", "标准/术语体系", 1.0),
    "review":             ("II", "综述", 0.7),
    "research":           ("III", "原始研究", 0.5),
}

_UNKNOWN = ("V", "未定级", 1.0)


def grade_of(doc_type: str | None) -> tuple[str, str, float]:
    """返回 (等级, 描述, 权重)。未知类型按 1.0，不惩罚。"""
    return _GRADE.get(doc_type or "", _UNKNOWN)


def grade_label(doc_type: str | None) -> str:
    """报告展示用，如 "I 级 国家指南"。"""
    lvl, desc, _ = grade_of(doc_type)
    return f"{lvl} 级 {desc}"


def grade_rank(doc_type: str | None) -> int:
    """排序用：I=1 … V=5，越小越权威。"""
    return {"I": 1, "II": 2, "III": 3, "IV": 4, "V": 5}.get(grade_of(doc_type)[0], 5)