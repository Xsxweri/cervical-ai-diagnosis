"""
双轨风险对比：
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

from utils.label_mapping import max_risk

# 报告声明风险等级的常见写法
_DECL_RE = re.compile(
    r"风险等级[^。；\n]{0,6}?[：:\s]\s*\**\s*"
    r"(紧急|高危|高|中等|中|低风险|低|待定|待评估|pending|emergency|high|medium|low)",
    re.IGNORECASE,
)

_ALIAS = {
    "紧急": "emergency", "emergency": "emergency",
    "高危": "high", "高": "high", "high": "high",
    "中等": "medium", "中": "medium", "medium": "medium",
    "低风险": "low", "低": "low", "low": "low",
    "待定": "pending", "待评估": "pending", "pending": "pending",
}

# 非风险等级：不能参与 max 比较
TERMINAL = {"pending", "unreadable"}


@dataclass
class DualTrackResult:
    code_risk: str                       # 轨道 A（可能被降级为终止态）
    declared_risk: Optional[str]         # 轨道 B；报告未声明时为 None
    final_risk: str                      # 合并后的最终等级
    diverged: bool                       # 两轨是否分歧
    review_required: bool                # 是否因分歧/终止态强制复核


def extract_declared_risk(report: str) -> Optional[str]:
    """从报告文本提取声明的风险等级。提取不到返回 None（不视为分歧）。"""
    m = _DECL_RE.search(report or "")
    if not m:
        return None
    token = m.group(1).lower()
    return _ALIAS.get(token) or _ALIAS.get(m.group(1))


def resolve(code_risk: str, declared: Optional[str]) -> DualTrackResult:
    """合并两条轨道。**终止态不参与数值比较**（见模块 docstring）。"""
    code = (code_risk or "").lower() or "low"

    # 未声明 → 无从比较，保留代码轨
    if declared is None:
        return DualTrackResult(
            code_risk=code, declared_risk=None,
            final_risk=code, diverged=False,
            review_required=code in TERMINAL,
        )

    dyn = declared.lower()

    # 任一为终止态 → 取终止态，不做数值比较
    # （max_risk("pending","low") == "low" 会把"证据不足"降成"低风险"）
    for t in (code, dyn):
        if t in TERMINAL:
            return DualTrackResult(
                code_risk=code, declared_risk=dyn,
                final_risk=t, diverged=code != dyn,
                review_required=True,
            )

    # 都是数值等级 → 保守取严
    merged = max_risk(code, dyn)
    return DualTrackResult(
        code_risk=code, declared_risk=dyn,
        final_risk=merged, diverged=code != dyn,
        review_required=code != dyn,
    )