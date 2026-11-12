"""
报告机械校验。

设计原则：反思必须有**可验证的检查项**，而不是再问一次 LLM"你觉得对吗"。
每条规则都对应一个曾经真实发生过的缺陷：

    unknown_citation        → 报告"根据文献"却无法定位到具体哪一条
    scope_violation         → 细胞级证据被写成病例级结论（risk_scope 曾只在类型里）
    missing_conflict        → 有冲突却未声明（case_evidence._is_resolved 修过同类问题）
    missing_review          → 高风险/需复核却无复核建议（HERLEV_RISK_FLOOR 配套）
    unreliable_morphology   → 形态学测量不可靠却据此推断（"核在中央"伪断言已删，防回退）
    undisclosed_insufficiency → 证据不足未披露（Phase 1 新增维度）

全部为正则/集合运算，无 LLM、无 IO。
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List

from utils.evidence_ledger import EvidenceLedger
from utils.uncertainty import CASE_RISK_MIN_CELLS
from utils.dual_track import extract_declared_risk

# 报告里出现的、形如 E1 / C3 / M2 的引用标记
_CITE_RE = re.compile(r"\[([ECMGH])(\d+)\]")

# 病例级表述（细胞级证据不得使用）
_CASE_SUBJECT_RE = re.compile(
    r"该患者|本例患者|本病例|该病例|该患者整体|患者整体|病例整体"
)

# 句子切分（用于定位上下文，不改变匹配逻辑）
_SENT_RE = re.compile(r"[^。；\n]+")

# 样本局限表述：出现在同一句时，说明在描述样本量而非下病例级结论
_LIMITATION_RE = re.compile(
    r"仅检出|仅见|仅|不足|少于|样本量|细胞数|样本|"
    r"不代表|无法代表|不能代表|难以代表"
)

# 风险断言：同一句里出现这些，说明确实在下结论
_RISK_ASSERT_RE = re.compile(
    r"(?:低|中|高|严重|紧急)\s*(?:风险|级别)"
    r"|风险(?:等级)?[^。；\n]{0,8}?(?:为|是|属|达|较)\s*(?:低|中|高|严重|紧急)"
    r"|风险等级"
    r"|恶性|癌|病变|分级"
    r"|ASC-US|ASC-H|LSIL|HSIL"
    r"|阴性|阳性"
)

# 否定/禁止词。出现在断言词之前时，说明该断言被否定，不算违规。
_NEGATION_RE = re.compile(
    r"不得|严禁|禁止|不应|不可|不能|无法|不足以|切勿|避免|不作|不予|"
    r"不代表|不排除|不提供|未作|未给出|无需|不作出|不下|不予以|"
    r"疑似|疑为|待定|不确定"
)

# 冲突必须显式声明
_CONFLICT_RE = re.compile(r"冲突|不一致|需.{0,6}复核|人工复核")

# 复核建议
_REVIEW_RE = re.compile(r"复核|镜检|复诊|重新采样")

# 形态学推断措辞
_MORPH_INFER_RE = re.compile(r"核质比.{0,12}(提示|表明|支持|说明|符合)")

# 证据不足披露
_INSUFFICIENT_RE = re.compile(r"证据不足|证据有限|文献不足")

_CONCLUSION_SECTION_KW = ("关键发现", "细胞学判读")

_NEGATION_WINDOW = 20



@dataclass
class Issue:
    kind: str
    detail: str
    severity: str = "blocker"      # blocker | warning

def _conclusion_sections(text: str) -> str:
    """
    返回关键结论部分。
    """
    lines = (text or "").splitlines()
    if not any(ln.startswith("## ") for ln in lines):
        return text or ""

    out: list[str] = []
    keep = False
    for ln in lines:
        if ln.startswith("## "):
            keep = any(k in ln for k in _CONCLUSION_SECTION_KW)
        if keep:
            out.append(ln)
    return "\n".join(out)

def _has_positive_risk_assertion(sent: str) -> bool:
    """
    句子中是否存在正向风险断言（非否定/禁止词修饰）。
    """
    for m in _RISK_ASSERT_RE.finditer(sent):
        window = sent[max(0, m.start() - _NEGATION_WINDOW):m.start()]
        if not _NEGATION_RE.search(window):
            return True
    return False

def _scope_issue(text: str) -> tuple[str, str] | None:
    """判定报告是否存在口径问题。返回 (severity, 命中句) 或 None。

    判据（按顺序）：
      1. 无病例级主语 → 不是口径问题，跳过。
      2. 有局限词（仅检出/不足/样本量…）且断言全被否定 → 正确的局限声明，干净。
      3. 有局限词但有正向断言 → 措辞不规范（在描述样本量时夹带结论）→ warning。
      4. 无局限词却有正向断言 → 直接下病例级结论 → blocker。

    为什么不用「距离窗口 + 语序」做严格正则：
        实测两种真实越界都漏拦 ——
          「该患者整体风险较低」是"风险+较+低"（语序反向）
          「本病例判定为阴性」不含"风险"二字
        而窗口放宽就会把「本病例仅检出 1 个细胞，属低风险」这类
        局限声明一起捞进来。「有无局限词」比「距离多远」更能区分意图。
    """
    for sent in _SENT_RE.findall(text or ""):
        if not _CASE_SUBJECT_RE.search(sent):
            continue
        positive = _has_positive_risk_assertion(sent)
        limited = bool(_LIMITATION_RE.search(sent))

        if limited and not positive:
            continue                                   # 局限声明，干净
        if limited:
            return ("warning", sent.strip())           # 措辞问题
        if positive:
            return ("blocker", sent.strip())           # 真越界
    return None

def validate_report(
    report: str,
    ledger: EvidenceLedger,
    ctx: dict,
) -> List[Issue]:
    """返回问题列表；空列表 = 通过。

    Args:
        report: LLM 生成的报告正文
        ledger: 本轮诊断构建的证据账本
        ctx: 上下文，需含 total_cells / risk_level / conflict_count /
             review_required / evidence_sufficiency / cells
    """
    issues: List[Issue] = []
    text = report or ""

    total_cells = int(ctx.get("total_cells", 0) or 0)
    risk = ctx.get("risk_level") or ""

    # ---- 1) 引用完整性 ----
    cited = {f"{m.group(1)}{m.group(2)}" for m in _CITE_RE.finditer(text)}
    if unknown := cited - ledger.ids:
        issues.append(Issue(
            "unknown_citation",
            f"引用了不存在的证据: {sorted(unknown)}（账本内共 {len(ledger.ids)} 条）",
            severity="warning",
        ))

    # ---- 2) 口径：细胞级不得写成病例级 ----
    if total_cells < CASE_RISK_MIN_CELLS:
        found = _scope_issue(_conclusion_sections(text))
        if found:
            severity, sentence = found
            if severity == "blocker":
                issues.append(Issue(
                    "scope_violation",
                    f"仅 {total_cells} 个细胞（<{CASE_RISK_MIN_CELLS}），"
                    f"却给出病例级结论「{sentence[:60]}」",
                ))
            else:
                issues.append(Issue(
                    "scope_wording",
                    f"细胞级报告夹带病例级结论：「{sentence[:60]}」"
                    f"（描述样本量时请勿同时下病例级判断）",
                    severity="warning",
                ))

    # ---- 3) 冲突必须声明 ----
    has_conflict = bool(ctx.get("conflict_count") or ctx.get("review_required"))
    if has_conflict and not _CONFLICT_RE.search(text):
        issues.append(Issue(
            "missing_conflict",
            f"存在冲突证据（conflict_count={ctx.get('conflict_count', 0)}）但报告未提及",
        ))

    # ---- 4) 高风险 / 需复核必须有复核建议 ----
    if (risk in {"high", "emergency"} or ctx.get("review_required")) \
            and not _REVIEW_RE.search(text):
        issues.append(Issue(
            "missing_review",
            f"风险={risk}，review_required={bool(ctx.get('review_required'))}，"
            "但报告缺人工复核建议",
        ))

    # ---- 5) 形态学不可靠时不得据此推断 ----
    cells = ctx.get("cells") or []
    inspected = len(cells)
    unreliable = sum(
        1 for c in cells
        if not ((c.get("morphology") or {}).get("measurement_reliable"))
    )
    if inspected and unreliable / inspected > 0.5 and _MORPH_INFER_RE.search(text):
        issues.append(Issue(
            "unreliable_morphology",
            f"{unreliable}/{inspected} 个细胞形态学测量不可靠，却据核质比下结论",
        ))

    # ---- 6) 证据不足必须披露（warning，不阻断）----
    suff = ctx.get("evidence_sufficiency")
    if suff is not None and float(suff) < 0.5 and not _INSUFFICIENT_RE.search(text):
        issues.append(Issue(
            "undisclosed_insufficiency",
            f"证据充分性仅 {float(suff):.2f}，报告未声明证据不足",
            severity="warning",
        ))

    # ---- 7) 报告声明的风险等级与代码轨是否一致 ----
    declared = extract_declared_risk(text)
    if declared and risk and declared != risk.lower():
        issues.append(Issue(
            "risk_level_mismatch",
            f"报告声明风险等级={declared}，代码判定={risk}；已标记需人工复核",
            severity="warning",
        ))

    # ---- 8) 证据利用率：充分性满分却零引用 ----
    if suff is not None and float(suff) >= 1.0 and ledger.count("guideline") > 0:
        if not any(c.startswith("E") for c in cited):
            issues.append(Issue(
                "unused_evidence",
                f"证据充分性 1.00（账本内 {ledger.count('guideline')} 条文献），"
                f"但报告未引用任何文献编号",
                severity="warning",
            ))

    return issues


def has_blocker(issues: List[Issue]) -> bool:
    return any(i.severity == "blocker" for i in issues)