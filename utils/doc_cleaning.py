"""
文档清洗：参考文献剥离等纯文本处理。
"""

from __future__ import annotations

import re
from typing import List, Optional, Tuple

# 参考文献标题：独占一行。中文允许字间空格（排版产物）、全角/半角括号。
_REF_HEAD_RE = re.compile(
    r"^[ \t\u3000\r]*(?:[【\[]?[ \t\u3000]*参[ \t\u3000]*考[ \t\u3000]*文[ \t\u3000]*献"
    r"[ \t\u3000]*[】\]]?|references?|bibliography)"
    r"[ \t\u3000\r]*[:：]?[ \t\u3000\r]*$",
    re.MULTILINE | re.IGNORECASE,
)
_REF_ITEM_RE = re.compile(r"^[ \t\u3000]*[\[\(]?\d{1,3}[\]\)\.、]\s*\S", re.MULTILINE)

_CONFIRM_CHARS = 600
_CONFIRM_ITEMS = 2
_MIN_KEEP_RATIO = 0.30
_MIN_POSITION = 0.40


def strip_references_pages(pages: List[str]) -> Tuple[List[str], Optional[str]]:
    """按页剥离参考文献段。

    Returns:
        (保留的页文本, 说明)。说明为 None 表示未剥离。
        **纯函数、无副作用**：由调用方决定是否打印。

    保守失败：任何判据不满足都原样返回。
    漏剥离只是留噪音，误剥离会永久丢失正文证据。
    """
    n = len(pages)
    if n < 3:
        return list(pages), None

    cut = None
    for i in range(n - 1, -1, -1):
        text = pages[i] or ""
        if not text:
            continue
        if n >= 4 and i < n * _MIN_POSITION:
            continue
        matches = list(_REF_HEAD_RE.finditer(text))
        if matches:
            cut = (i, matches[-1].start())
            break

    if cut is None:
        return list(pages), None

    i, off = cut
    window = (pages[i] or "")[off: off + _CONFIRM_CHARS]
    if len(_REF_ITEM_RE.findall(window)) < _CONFIRM_ITEMS:
        return list(pages), "命中参考文献标题但未见编号条目，疑似误判，保留原文"

    kept_tail = (pages[i] or "")[:off].rstrip()
    head = list(pages[:i])
    before = sum(len(p or "") for p in pages)
    after = sum(len(p or "") for p in head) + len(kept_tail)
    if not before or after / before < _MIN_KEEP_RATIO:
        return list(pages), f"剥离后仅剩 {after / before:.0%}，疑似误判，保留原文"

    out = head + ([kept_tail] if kept_tail.strip() else [])
    return out, f"剥离参考文献：{before} → {after} 字符（{n} → {len(out)} 页）"