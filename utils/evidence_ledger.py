"""
证据账本：给每条证据分配稳定 ID, 供报告引用与机械校验。

设计约束：
    * 纯数据结构，无 IO、无 LLM —— 可以在任何环境（含单测）里构造；
    * ID 前缀区分来源，便于 validator 做分类检查；
    * to_json() 直接落 PG, 供事后审计与回放。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

# 前缀 → 证据类型。用不同前缀是为了让 validator 能分辨"引用了文献"还是"引用了细胞"
_PREFIX = {
    "guideline": "E",   # 文献/指南
    "cell": "C",        # 细胞明细
    "morphology": "M",  # 形态学测量
    "gradcam": "G",     # 热力图
    "history": "H",     # 历史病例
    "system": "S",      # 系统事件（工具失败、预算用尽、双轨分歧）
}


@dataclass
class EvidenceItem:
    id: str
    kind: str
    content: str
    source: str
    score: Optional[float] = None
    reliability: str = "ok"          # ok | low
    grade: str = ""                  # CEBM 等级标签，如 "I 级 国家指南"


@dataclass
class EvidenceLedger:
    items: List[EvidenceItem] = field(default_factory=list)
    _seq: Dict[str, int] = field(default_factory=dict)

    def add(
        self,
        kind: str,
        content: str,
        source: str,
        score: Optional[float] = None,
        reliability: str = "ok",
        grade: str = "",
    ) -> EvidenceItem:
        """追加一条证据并分配 ID。"""
        prefix = _PREFIX.get(kind, "X")
        n = self._seq.get(prefix, 0) + 1
        self._seq[prefix] = n
        item = EvidenceItem(f"{prefix}{n}", kind, content, source, score, reliability, grade)
        self.items.append(item)
        return item

    # ---------------- 查询 ----------------

    @property
    def ids(self) -> set[str]:
        return {i.id for i in self.items}

    def of_kind(self, kind: str) -> List[EvidenceItem]:
        return [i for i in self.items if i.kind == kind]

    def count(self, kind: str) -> int:
        return sum(1 for i in self.items if i.kind == kind)

    # ---------------- 序列化 ----------------

    def as_prompt_block(self) -> str:
        """注入报告的格式化文本。每条带 ID，LLM 被要求用 ID 引用。"""
        if not self.items:
            return "（无外部证据）"
        return "\n\n".join(
            f"[{i.id}] ({i.kind}{', ' + i.grade if i.grade else ''}, {i.source}) {i.content}"
            for i in self.items
        )

    def to_json(self) -> List[dict]:
        return [item.__dict__ for item in self.items]

    @classmethod
    def from_json(cls, data: Optional[List[dict]]) -> "EvidenceLedger":
        """从 PG 回读（report_repo.load 用）。"""
        led = cls()
        for d in data or []:
            idv = str(d.get("id") or "")
            prefix = idv[:1] if idv else "X"
            n = int(idv[1:]) if idv[1:].isdigit() else 0
            led._seq[prefix] = max(led._seq.get(prefix, 0), n)
            led.items.append(
                EvidenceItem(
                    id=idv,
                    kind=d.get("kind", ""),
                    content=d.get("content", ""),
                    source=d.get("source", ""),
                    score=d.get("score"),
                    reliability=d.get("reliability", "ok"),
                )
            )
        return led

    # ---------------- 便捷构造 ----------------

    @classmethod
    def from_knowledge(cls, knowledge: List[dict], max_chars: int = 800) -> "EvidenceLedger":
        """从既有 retrieved_knowledge 构建。

        Phase 1.5 的兼容入口：不改变现有检索流程，只是给已有文献补上 ID。
        max_chars 截断是为了控制 prompt 长度 —— 报告里只需要引用要点，
        不需要把整页 PDF 塞进上下文。
        """
        led = cls()
        for k in knowledge or []:
            from utils.evidence_grade import grade_label
            led.add(
                "guideline",
                (k.get("content") or "")[:max_chars],
                f"{(k.get('source') or '未知').split(chr(92))[-1]} p{k.get('page', 0)}",
                score=k.get("score"),
                grade=grade_label(k.get("doc_type")),
            )
        return led