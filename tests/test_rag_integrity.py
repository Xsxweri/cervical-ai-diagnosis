"""RAG 完整性回归：跨来源去重 + 可用性真实性。

不加载嵌入模型、不读写 knowledge_db/ 与 Cervical_cytology/。
单独运行：pytest tests/test_rag_integrity.py -q
"""

import pytest
import numpy as np
from collections import OrderedDict
from Agents import knowledge as knowledge_mod
from Agents.knowledge import UnavailableKnowledgeBase
from Agents.nodes.retrieval import evidence_sufficiency_from
from services.rag_service import MedicalKnowledgeBase

from utils.rag_index import rebuild_reason
from utils.retrieval_fusion import (
    BM25, cosine_to_distance, distance_to_cosine, dedupe_knowledge,
    expand_query, mmr_select, rrf_fuse, tokenize, normalize_query
)
from utils.query_rewrite import (
    detect_intents, expand_queries, matched_terms, rule_rewrite, validate_terms,
)


# ---------------- 跨来源去重 ----------------

def test_dedupe_keeps_best_score_per_source_page():
    items = [
        {"content": "A", "source": "a.pdf", "page": 3, "score": 0.9},
        {"content": "B", "source": "b.pdf", "page": 1, "score": 0.4},
        {"content": "A2", "source": "a.pdf", "page": 3, "score": 0.2},
    ]
    out = dedupe_knowledge(items)
    assert [i["source"] for i in out] == ["a.pdf", "b.pdf"]   # 已按距离升序
    assert out[0]["content"] == "A2"                          # 保留更近的一条


def test_dedupe_keeps_multiple_unknown_sources():
    """来源缺失时退化为内容指纹，不得把所有条目并成一条。"""
    items = [
        {"content": "段落一", "source": "未知", "page": 0, "score": 0.3},
        {"content": "段落二", "source": "未知", "page": 0, "score": 0.5},
    ]
    assert len(dedupe_knowledge(items)) == 2


def test_dedupe_handles_empty_input():
    assert dedupe_knowledge([]) == []
    assert dedupe_knowledge(None) == []


def test_duplicate_source_page_would_inflate_sufficiency():
    """回归护栏：证明去重是必要的，未去重会把 1 条文献灌成满分。"""
    raw = [{"content": "x", "source": "a.pdf", "page": 3, "score": 0.3}] * 5
    suff_raw, n_raw, _ = evidence_sufficiency_from(raw)
    suff_dedup, n_dedup, _ = evidence_sufficiency_from(dedupe_knowledge(raw))

    assert n_raw == 5 and n_dedup == 1
    assert suff_raw == pytest.approx(1.0)          # 虚高
    assert suff_dedup == pytest.approx(1 / 3, abs=1e-4)     # 降级


# ---------------- 可用性真实性 ----------------

class _FakeKb:
    def __init__(self, n: int):
        self.chunk_count = n


def test_is_available_false_without_vectorstore():
    kb = MedicalKnowledgeBase.__new__(MedicalKnowledgeBase)   # 不触发嵌入模型加载
    kb.vectorstore = None
    kb._count_cache = None
    kb.embeddings = None
    kb.embedding_error = None
    assert kb.chunk_count == -1
    assert kb.is_available is False


def test_kb_status_uninitialized(monkeypatch):
    monkeypatch.setattr(knowledge_mod, "kb", None)
    assert knowledge_mod.kb_status() == {
        "available": False, "chunks": 0, "reason": "知识库未初始化"
    }


def test_kb_status_degraded_carries_reason(monkeypatch):
    monkeypatch.setattr(knowledge_mod, "kb", UnavailableKnowledgeBase())
    st = knowledge_mod.kb_status()
    assert st["available"] is False and st["reason"]


def test_kb_status_empty_library_is_unavailable(monkeypatch):
    """旧实现只查 isinstance，空库会被误判为可用。"""
    monkeypatch.setattr(knowledge_mod, "kb", _FakeKb(0))
    st = knowledge_mod.kb_status()
    assert st["available"] is False
    assert "空" in st["reason"]


def test_kb_status_unreadable_count_is_unavailable(monkeypatch):
    monkeypatch.setattr(knowledge_mod, "kb", _FakeKb(-1))
    assert knowledge_mod.kb_status()["available"] is False


def test_kb_status_ready(monkeypatch):
    monkeypatch.setattr(knowledge_mod, "kb", _FakeKb(255))
    assert knowledge_mod.kb_status() == {
        "available": True, "chunks": 255, "reason": ""
    }

# ---------------- manifest 缺失 ----------------

def test_rebuild_when_manifest_missing_but_library_populated():
    """回归护栏：manifest 被删而库非空 → 必须重建，否则重复灌入。"""
    r = rebuild_reason(None, "fp", actual=255)
    assert r and "manifest 缺失" in r


def test_rebuild_when_fingerprint_changed():
    assert rebuild_reason({"fingerprint": "old", "chunks": 10}, "new", 10)


def test_rebuild_when_count_mismatch():
    assert rebuild_reason({"fingerprint": "fp", "chunks": 255}, "fp", 510)


def test_no_rebuild_when_consistent():
    assert rebuild_reason({"fingerprint": "fp", "chunks": 255}, "fp", 255) is None


def test_first_index_when_empty():
    assert "首次索引" in rebuild_reason(None, "fp", actual=0)


# ---------------- 混合检索原语 ----------------

def test_tokenize_cjk_bigrams_and_latin_words():
    toks = tokenize("HSIL 与 CIN2")
    assert "hsil" in toks and "cin2" in toks
    assert tokenize("宫颈细胞学") == ["宫颈", "颈细", "细胞", "胞学"]


def test_bm25_ranks_matching_doc_first():
    docs = [tokenize("宫颈 细胞学 筛查"), tokenize("Python 编程 教程")]
    hits = BM25(docs).search(tokenize("细胞学 筛查"), k=2)
    assert hits and hits[0][1] == 0


def test_rrf_prefers_document_in_both_lists():
    fused = dict(rrf_fuse([[1, 2, 3], [2, 9]], top_k=3))
    assert max(fused, key=fused.get) == 2


def test_mmr_reduces_redundancy():
    q = np.array([1.0, 0.0], dtype="float32")
    cand = np.array([
        [0.98, 0.199],   # 与 q 最相关 → 第一条被选中
        [0.98, 0.199],   # 与第 0 条几乎重复
        [0.20, 0.980],   # 相关性较低但方向不同
    ], dtype="float32")
    cand /= np.linalg.norm(cand, axis=1, keepdims=True)
    picked = mmr_select(q, cand, k=2, lambda_mult=0.3)
    assert picked[0] == 0
    assert 2 in picked and 1 not in picked        # 跳过与 0 近乎重复的 1


def test_expand_query_maps_abbreviation():
    assert "高级别鳞状上皮内病变" in expand_query("HSIL 处理原则")
    assert expand_query("挖空细胞") == "挖空细胞"   # 无缩写时原样返回


def test_distance_cosine_roundtrip():
    for cos in (0.0, 0.5, 0.9, 1.0):
        d = cosine_to_distance(cos)
        assert abs(distance_to_cosine(d) - cos) < 1e-3

def test_normalize_query_fullwidth():
    """全角/全角空格必须归一化成半角。"""
    assert normalize_query("ＡＳＣ－ＵＳ　处理") == "ASC-US 处理"


def test_expand_query_normalizes_before_matching():
    """expand_query 内部要归一化，否则全角缩写查不到。"""
    assert "意义未明的不典型鳞状细胞" in expand_query("ＡＳＣ－ＵＳ 处理")


def test_classification_queries_cover_all_labels():
    """回归护栏：新增标签空间时 query_map 必须同步。"""
    from utils.label_queries import missing_classification_queries
    assert missing_classification_queries() == []


def test_query_cache_returns_copy(monkeypatch):
    """缓存命中必须返回副本，外部改写不得污染缓存。"""
    kb = MedicalKnowledgeBase.__new__(MedicalKnowledgeBase)
    kb._query_cache = OrderedDict()
    kb.QUERY_CACHE_MAX = 8
    kb.vectorstore = object()
    kb._count_cache = 1
    kb.DOC_WEIGHTED = False
    fake = [{"content": "x", "source": "a.pdf", "page": 1, "score": 0.3}]

    async def _fake_index():
        return {"matrix": None}

    monkeypatch.setattr(kb, "_ensure_index", _fake_index)
    monkeypatch.setattr(kb, "_hybrid_search_sync", lambda *a, **kw: list(fake))

    import asyncio
    first = asyncio.run(kb.retrieve("宫颈癌", k=1))
    second = asyncio.run(kb.retrieve("宫颈癌", k=1))
    assert first is not second and first == second


def test_evidence_sufficiency_prefers_reported_cosine():
    """有 cosine 字段时不得从 score 反算（避免二次舍入翻转阈值）。"""
    from Agents.nodes.retrieval import evidence_sufficiency_from

    # 反算得 0.4955(<0.5 不算相关)，显式 cosine 0.5001(算相关) → 应采信后者
    items = [{"score": 1.0045, "cosine": 0.5001, "source": "a.pdf", "page": 1}]
    _, n_rel, _ = evidence_sufficiency_from(items)
    assert n_rel == 1

def test_rule_rewrite_maps_colloquial_to_canonical():
    extra = rule_rewrite("细胞核大了一圈，核挤在一起")
    assert "核增大" in extra and "核拥挤重叠" in extra


def test_rule_rewrite_does_not_duplicate_present_terms():
    """规范术语已在 query 中时不重复追加。"""
    assert "核增大" not in rule_rewrite("核增大伴核深染")


def test_rule_rewrite_adds_latin_alias():
    assert "HSIL" in rule_rewrite("高级别鳞状上皮内病变怎么处理")


def test_detect_intents_by_wording():
    assert detect_intents("HSIL 需要做阴道镜吗") == ["处理原则", "临床管理"]
    assert detect_intents("怎么和反应性改变区分") == ["鉴别诊断", "鉴别要点"]
    assert detect_intents("挖空细胞") == ["形态学特征", "诊断标准"]


def test_expand_queries_caps_and_keeps_original_first():
    vs = expand_queries("HSIL 怎么处理", max_variants=3)
    assert vs[0] == "HSIL 怎么处理"
    assert len(vs) <= 3


def test_validate_terms_drops_hallucinations():
    """不在词表内的术语必须被丢弃。"""
    kept = validate_terms(["核增大", "高度可疑恶性肿瘤", "挖空细胞"])
    assert kept == ["核增大", "挖空细胞"]


def test_llm_rewrite_disabled_by_default(monkeypatch):
    """默认关闭，不得发起任何 LLM 调用。"""
    import Agents.query_rewrite_llm as m
    monkeypatch.setattr(m, "ENABLED", False)
    import asyncio
    assert asyncio.run(m.rewrite_query_llm("核很大", "核增大")) == []


def test_llm_rewrite_failure_falls_back_to_empty(monkeypatch):
    """LLM 抛异常 → 返回空，调用方退化到规则层。"""
    import Agents.query_rewrite_llm as m
    monkeypatch.setattr(m, "ENABLED", True)

    class _Boom:
        def with_structured_output(self, _model, **_kwargs): return self
        async def ainvoke(self, _): raise RuntimeError("timeout")

    monkeypatch.setattr(m, "get_llm", lambda: _Boom())
    import asyncio
    assert asyncio.run(m.rewrite_query_llm("核很大", "核增大")) == []


def test_unavailable_kb_has_nl_methods():
    """降级桩必须覆盖新增入口，否则工具调用 AttributeError。"""
    from Agents.knowledge import UnavailableKnowledgeBase
    kb = UnavailableKnowledgeBase()
    import asyncio
    assert asyncio.run(kb.retrieve_nl("x", k=1)) == []
    assert asyncio.run(kb.retrieve_multi(["x"], k=1)) == []

def test_rag_terms_yaml_is_valid():
    """审校护栏：yaml 可解析，且所有 alias target 都在 canonical 内。"""
    from pathlib import Path
    import yaml
    root = Path(__file__).resolve().parent.parent
    with open(root / "configs" / "rag_terms.yaml", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    flat = {t for g in cfg["canonical"].values() for t in g}
    bad = {a: t for a, t in cfg["alias"].items() if t not in flat}
    assert not bad, f"alias 指向不存在的规范术语：{bad}"


def test_terms_fallback_is_subset_of_yaml():
    """内置兜底必须是 YAML 的子集，且同名词的映射一致。"""
    from utils import query_rewrite as qr
    qr.load_terms(force=True)

    # ① canonical：兜底每个组必须存在，且词是该组 YAML 的子集
    for group, terms in qr._FALLBACK_CANONICAL.items():
        assert group in qr.CANONICAL_TERMS, (
            f"YAML 缺少兜底里的 canonical 组「{group}」"
        )
        missing = terms - qr.CANONICAL_TERMS[group]
        assert not missing, (
            f"canonical「{group}」中这些兜底术语在 YAML 里不存在（疑似改名/删除）：{missing}"
        )

    # ② alias：兜底里的每个映射，YAML 中必须存在且指向同一个 target
    mismatched = {
        k: (v, qr._ALIAS.get(k))
        for k, v in qr._FALLBACK_ALIAS.items()
        if qr._ALIAS.get(k) != v
    }
    assert not mismatched, (
        f"alias 两边不一致（key: 兜底 target, YAML target）：{mismatched}"
    )


def test_matched_terms_detects_canonical():
    """retrieve_nl 用 matched_terms 判断"是否已是规范术语"，防止误触发 LLM。"""
    assert "核增大" in matched_terms("核增大的细胞")


def test_rule_rewrite_hit_count_not_doubled():
    """expand_queries 传入 extra 时不得再次累加计数。"""
    from utils.query_rewrite import REWRITE_HITS
    REWRITE_HITS.clear()
    extra = rule_rewrite("核很大")
    expand_queries("核很大", max_variants=3, extra=extra)
    assert sum(REWRITE_HITS.values()) == len(extra)

def test_dedupe_prefers_higher_rrf():
    """混合检索下按 rrf 保留更优项，而非距离。"""
    items = [
        {"source": "a.pdf", "page": 1, "score": 0.1, "rrf": 0.010},
        {"source": "a.pdf", "page": 1, "score": 0.9, "rrf": 0.030},
    ]
    out = dedupe_knowledge(items)
    assert len(out) == 1 and out[0]["rrf"] == 0.030


def test_dedupe_empty_content_uses_index_fallback():
    """内容为空时用下标保底，互不合并。"""
    items = [{"source": "未知", "page": 0, "content": ""} for _ in range(3)]
    assert len(dedupe_knowledge(items)) == 3

def test_unknown_classification_falls_back_to_raw_label():
    from utils.label_queries import get_classification_query
    q = get_classification_query("herlev2_something")
    assert "herlev2_something" in q


def test_known_classification_uses_mapped_query():
    from utils.label_queries import CLASSIFICATION_QUERIES, get_classification_query
    assert get_classification_query("light_dysplastic") == \
        CLASSIFICATION_QUERIES["light_dysplastic"]

def test_doc_bonus_applies_authority_bias():
    """权威加成作用在归一化基准上：0.04 ≈ 相对差距 4%。

    归一化后 base = cosine / max(cosine)，故 bonus 只能跨越约 4% 的相对差距。
    余弦差在 4% 以内时，高权威文档可以反超。
    """
    from services.rag_service import _apply_doc_bonus
    items = [
        {"cosine": 0.5000, "doc_type": "research", "source": "a.pdf", "page": 1},
        {"cosine": 0.4900, "doc_type": "review", "source": "b.pdf", "page": 2},
    ]
    out = _apply_doc_bonus(items, weighted=True)
    # research base=1.000 ; review base=0.980 + 0.04 = 1.020 → 反超
    assert out[0]["doc_type"] == "review"


def test_doc_bonus_cannot_override_large_cosine_gap():
    """归一化后 top-1 不会被 bonus 推翻（差距 > bonus 时）。

    这是归一化的已知副作用：为了让 fused_rank(0.01-0.03) 不被 bonus 支配，
    统一归一化到 [0,1]，代价是 cosine 路径的权威加成从
    "绝对 0.04" 退化为 "4% 相对差距"。
    待评估：是否只对 fused_rank 归一化（见 docs/RAG.md）。
    """
    from services.rag_service import _apply_doc_bonus
    items = [
        {"cosine": 0.7800, "doc_type": "research"},
        {"cosine": 0.7200, "doc_type": "review"},
    ]
    out = _apply_doc_bonus(items, weighted=True)
    assert out[0]["doc_type"] == "research"       # 差距 7.7% > 4%，无法反超


def test_doc_bonus_disabled_has_no_authority_bias():
    """DOC_WEIGHT=false 时 bonus=0，但排序仍按归一化基准重排。

    注意：不是"完全不碰"（cos_rank 仍会写入，且按 base 重排），
    因为变体检索依赖这一行为保持顺序一致性。
    """
    from services.rag_service import _apply_doc_bonus
    items = [
        {"cosine": 0.40, "doc_type": "review"},
        {"cosine": 0.50, "doc_type": "research"},
    ]
    out = _apply_doc_bonus(items, weighted=False)
    assert all("cos_rank" in it for it in out)
    assert out[0]["doc_type"] == "research"          # 0.50 的 base 更高
    assert abs(out[0]["cos_rank"] - 1.0) < 1e-9      # 归一化后 top 恒为 1.0


def test_rank_key_prefers_cos_rank_over_rrf():
    """cos_rank 必须优先，否则 dedupe 会撤销权威性加成。"""
    from utils.retrieval_fusion import _rank_key
    assert _rank_key({"cos_rank": 0.60, "rrf": 0.01, "score": 0.9}) == -0.60
    assert _rank_key({"rrf": 0.01, "score": 0.9}) == -0.01


def test_evidence_sufficiency_uses_cos_rank():
    """充分性必须与检索排序一致，否则权威文档被误判不相关。"""
    from Agents.nodes.retrieval import evidence_sufficiency_from
    items = [{"cosine": 0.40, "cos_rank": 0.54}] * 3
    suff, n, _ = evidence_sufficiency_from(items)
    assert n == 3 and suff == 1.0

def test_corpus_manifest_matches_files():
    """回归护栏：清单必须覆盖所有实际语料文件，且无失效条目。

    历史事故：
      1) 清单 file 值带 .pdf 后缀 → MinerU 的 source 不带后缀 → 全部降级 research
      2) 新增语料未登记 → doc_type 静默变 research，检索质量下降无告警
    """
    import glob
    from pathlib import Path
    import yaml
    from utils.mineru_loader import discover_mineru_json, source_name_from

    root = Path(__file__).resolve().parent.parent
    corpus = root / "Cervical_cytology"

    with open(root / "configs" / "corpus_manifest.yaml", encoding="utf-8") as f:
        listed = {d["file"] for d in (yaml.safe_load(f) or {}).get("docs") or []}

    actual = {Path(p).name for p in glob.glob(str(corpus / "*.pdf"))}
    actual |= {source_name_from(f) for f in discover_mineru_json(str(corpus))}

    missing = sorted(actual - listed)
    stale = sorted(listed - actual)
    assert not missing, f"语料未登记到清单（会降级为 research）：{missing}"
    assert not stale, f"清单条目无对应文件（疑似改名/删除后未同步）：{stale}"