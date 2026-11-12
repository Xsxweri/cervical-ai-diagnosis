"""
检索融合原语: 分词、BM25、RRF、MMR、查询扩展、距离/余弦换算。
"""

from __future__ import annotations

import math
import re
import unicodedata
from collections import Counter

_TOKEN_RE = re.compile(r"[A-Za-z]+\d*|\d+|[\u4e00-\u9fff]")

_CJK = "\u4e00-\u9fff"

# 缩写 → 中文全称。医学文本里中英混写，BM25 对不上同义表述，
# 这里做最小扩展（不引入同义词库）。
_ABBREV = {
    "hsil": "高级别鳞状上皮内病变",
    "lsil": "低级别鳞状上皮内病变",
    "asc-us": "意义未明的不典型鳞状细胞",
    "asc-h": "不除外高级别病变的不典型鳞状细胞",
    "ascus": "意义未明的不典型鳞状细胞",
    "nilm": "未见上皮内病变或恶性",
    "cin1": "宫颈上皮内瘤变I级",
    "cin2": "宫颈上皮内瘤变II级",
    "cin3": "宫颈上皮内瘤变III级",
    "hpv": "人乳头瘤病毒",
    "tbs": "TBS分类",
    "cis": "原位癌",
}

_SEG_SPLIT_RE = re.compile(r"[^A-Za-z0-9\u4e00-\u9fff]+")
_CJK_CHAR_RE = re.compile(f"[{_CJK}]")

_UNKNOWN_SOURCE = (None, "", "未知")


def _rank_key(it: dict) -> float:
    """
    一排序键：越小越优。
    优先级：cos_rank（含 doc_type 权威性加成）> -rrf > l2 距离。
    """
    if it.get("cos_rank") is not None:
        return -float(it["cos_rank"])
    if it.get("rrf") is not None:
        return -float(it.get("rrf") or 0.0)
    return float(it.get("score") or 0.0)


def dedupe_key(it: dict, key: tuple, idx: int):
    """去重主键。source 缺失 → 内容指纹；内容也空 → 用下标保底，保证互不合并。"""
    if it.get("source") in _UNKNOWN_SOURCE:
        body = (it.get("content") or "").strip()[:200]
        return ("content", body) if body else ("idx", idx)
    return tuple(it.get(f) for f in key)


def dedupe_knowledge(items, key=("source", "page"), sort_by_score=True):
    """跨检索来源合并时去重，同一 key 保留更优的一条。

    保留位置取首次出现，避免结果顺序随输入顺序抖动。
    """
    best = {}
    order = []
    for idx, it in enumerate(items or []):
        k = dedupe_key(it, key, idx)
        cur = best.get(k)
        if cur is None:
            best[k] = it
            order.append(k)
        elif _rank_key(it) < _rank_key(cur):
            best[k] = it
    out = [best[k] for k in order]
    if sort_by_score:
        out.sort(key=_rank_key)
    return out

def normalize_query(query: str) -> str:
    """NFKC(全角→半角)+ 折叠空白。"""
    q = unicodedata.normalize("NFKC", query or "")
    return re.sub(r"\s+", " ", q).strip()

def tokenize(text: str) -> list[str]:
    """先按标点/空白分段，再在段内做 CJK 字 bigram，拉丁/数字整词。"""
    out: list[str] = []
    for seg in _SEG_SPLIT_RE.split((text or "").lower()):
        if not seg:
            continue
        prev_cjk: str | None = None
        for tok in _TOKEN_RE.findall(seg):
            if len(tok) == 1 and _CJK_CHAR_RE.match(tok):
                if prev_cjk is not None:
                    out.append(prev_cjk + tok)
                prev_cjk = tok
            else:
                out.append(tok)
                prev_cjk = None
    return out

_ABBREV_RE = {
    abbr: re.compile(rf"(?<![A-Za-z0-9]){re.escape(abbr)}(?![A-Za-z0-9])", re.IGNORECASE)
    for abbr in _ABBREV
}

def expand_query(query: str) -> str:
    """把英文缩写补上中文全称，提升 BM25 与向量两侧的召回一致性。"""
    q = normalize_query(query)
    if not q:
        return q
    extra = [zh for abbr, zh in _ABBREV.items() if _ABBREV_RE[abbr].search(q)]
    extra = list(dict.fromkeys(extra))
    return f"{q} {' '.join(extra)}" if extra else q


class BM25:
    """Okapi BM25，纯 Python 实现（语料百级～万级足够）。"""

    def __init__(self, docs_tokens: list[list[str]], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.n = len(docs_tokens)
        self.dl = [len(d) for d in docs_tokens]
        self.avgdl = (sum(self.dl) / self.n) if self.n else 0.0
        self.tf = [Counter(d) for d in docs_tokens]
        df: Counter = Counter()
        for toks in docs_tokens:
            df.update(set(toks))
        # 带 +0.5 平滑的 IDF，恒为正
        self.idf = {t: math.log(1 + (self.n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def search(self, query_tokens: list[str], k: int) -> list[tuple[float, int]]:
        q = [t for t in query_tokens if t in self.idf]
        if not q or not self.n:
            return []
        scored: list[tuple[float, int]] = []
        for i, tf in enumerate(self.tf):
            norm = self.k1 * (1 - self.b + self.b * (self.dl[i] or 1) / (self.avgdl or 1))
            s = 0.0
            for t in q:
                f = tf.get(t, 0)
                if f:
                    s += self.idf[t] * f * (self.k1 + 1) / (f + norm)
            if s > 0:
                scored.append((s, i))
        scored.sort(key=lambda x: -x[0])
        return scored[:k]


def rrf_fuse(
    ranked_id_lists: list[list[int]], k: int = 60, top_k: int = 10, weights: list[float] | None = None
) -> list[tuple[int, float]]:
    """Reciprocal Rank Fusion。输入按相关性从高到低的 id 列表。"""
    if weights is None:
        weights = [1.0] * len(ranked_id_lists)
    agg: dict[int, float] = {}
    for ids, w in zip(ranked_id_lists, weights):
        for rank, doc_id in enumerate(ids, start=1):
            agg[doc_id] = agg.get(doc_id, 0.0) + w / (k + rank)
    return sorted(agg.items(), key=lambda x: -x[1])[:top_k]


def mmr_select(query_vec, cand_matrix, k: int, lambda_mult: float = 0.7) -> list[int]:
    """
    最大边际相关性。返回被选中候选在 cand_matrix 中的行号（按选择顺序）。

    query_vec: (d,)，已归一化
    cand_matrix: (n, d)，每行已归一化
    """
    n = len(cand_matrix)
    if not n:
        return []
    sim_q = cand_matrix @ query_vec
    selected: list[int] = []
    remaining = list(range(n))
    while remaining and len(selected) < k:
        best_i, best_s = remaining[0], -math.inf
        for i in remaining:
            redundancy = (
                max(float(cand_matrix[i] @ cand_matrix[j]) for j in selected)
                if selected else 0.0
            )
            s = lambda_mult * float(sim_q[i]) - (1 - lambda_mult) * redundancy
            if s > best_s:
                best_i, best_s = i, s
        selected.append(best_i)
        remaining.remove(best_i)
    return selected


def cosine_to_distance(cos: float) -> float:
    """余弦 → 等价的 l2 距离(嵌入已归一化): d = sqrt(2 - 2cos)。"""
    return math.sqrt(max(0.0, 2.0 * (1.0 - max(-1.0, min(1.0, float(cos))))))


def distance_to_cosine(dist: float) -> float:
    """l2 距离 → 余弦: cos = 1 - d²/2。与 Agents/nodes/retrieval.py 原实现一致。"""
    d = max(float(dist or 0.0), 0.0)
    return round(1.0 - (d * d) / 2.0, 4)