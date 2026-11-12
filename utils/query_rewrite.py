"""
查询改写：规则优先的术语标准化 + 多查询扩展。

设计约束：
  * 只做"追加/归一"，不改写原 query（与 expand_query 一致，便于回溯）；
  * 不做语义推断，只做词表映射 —— 确定性、可审计；
"""

from __future__ import annotations
import logging
import os
import re
from collections import Counter
from functools import lru_cache

import yaml

from utils.retrieval_fusion import normalize_query

logger = logging.getLogger("query_rewrite")

REWRITE_HITS: Counter = Counter()

# 1. 规范术语词表（白名单）, 本兜底必须与 configs/rag_terms.yaml 保持一致, 新增 YAML 词条时请同步此处。
_FALLBACK_CANONICAL: dict[str, set[str]] = {
    "正常": {"正常鳞状上皮", "NILM", "未见上皮内病变"},
    "感染": {"HPV感染", "挖空细胞", "人乳头瘤病毒", "非HPV16/18亚型"},
    "低级别": {"LSIL", "低级别鳞状上皮内病变", "轻度异型增生", "CIN1", "CIN"},
    "高级别": {"HSIL", "高级别鳞状上皮内病变", "中度异型增生", "重度异型增生", "CIN2", "CIN3"},
    "癌": {"原位癌", "CIS", "鳞状细胞癌", "浸润癌"},
    "不典型": {"ASC-US", "ASC-H", "意义未明的不典型鳞状细胞", "不典型鳞状细胞"},
    "形态": {"核增大", "核深染", "核质比增高", "核膜不规则", "核拥挤重叠", "染色质颗粒状"},
    "处理": {"阴道镜指征", "锥切", "活检", "随访间隔", "HPV分型检测", "处理原则"},
    "检查": {"细胞学检查", "阴道镜检查", "HPV核酸检测", "联合筛查"},
    "随访": {"12个月", "随访管理", "定期复查"},
}

# 2. 口语/变体 → 规范术语
_FALLBACK_ALIAS: dict[str, str] = {
    # 形态描述
    "核大": "核增大",
    "细胞核大": "核增大",
    "核变大": "核增大",
    "核黑": "核深染",
    "颜色深": "核深染",
    "挤在一起": "核拥挤重叠",
    "重叠": "核拥挤重叠",
    "核不规则": "核膜不规则",
    "核形不规则": "核膜不规则",
    # 病变名称变体
    "癌前病变": "高级别鳞状上皮内病变",
    "上皮内瘤变": "CIN",
    "挖空": "挖空细胞",
    "空泡细胞": "挖空细胞",
    "不典型细胞": "不典型鳞状细胞",
    "意义不明": "ASC-US",
    # 临床动作
    "怎么治": "处理原则",
    "怎么处理": "处理原则",
    "要不要做阴道镜": "阴道镜指征",
    "多久复查": "随访间隔",
     # 临床术语与口语
    "TCT": "细胞学检查",
    "tct": "细胞学检查",
    "等一年": "12个月",
    "过一年": "12个月",
    "HPV52": "非HPV16/18亚型",
    "其他高危型HPV": "非HPV16/18亚型",
    "查阴道镜": "阴道镜检查",
}

# 规范术语 → 英文缩写（反向补充，语料中英混写）
_FALLBACK_TO_LATIN = {
    "高级别鳞状上皮内病变": "HSIL",
    "低级别鳞状上皮内病变": "LSIL",
    "人乳头瘤病毒": "HPV",
    "宫颈上皮内瘤变": "CIN",
    "原位癌": "CIS",
}

# 意图 → 后缀模板。默认不扩展，靠 regex 命中才加。
_FALLBACK_INTENT_RULES: list[tuple[re.Pattern, list[str]]] = [
    (re.compile(r"首选|推荐|应选|如何选|怎么选|哪个更好|还是"),["推荐意见", "推荐级别"]),
    (re.compile(r"处理|治疗|怎么办|方案|指征|处置|阴道镜|锥切|活检"), ["处理原则", "临床管理"]),
    (re.compile(r"随访|复查|间隔|追踪"), ["随访管理", "复查间隔"]),
    (re.compile(r"鉴别|区分|区别|误诊|不像"), ["鉴别诊断", "鉴别要点"]),
    (re.compile(r"诊断|标准|判读|分级|定义"), ["诊断标准", "分级判读"]),
    (re.compile(r"机制|原因|病理|为什么|发生"), ["病理机制", "病因学"]),
]

_QUESTION_RE = re.compile(
    r"是什么|意思是|含义|定义|什么意思|指什么|什么叫"
    r"|包括哪些|包含哪些|有哪些|包含什么|包括什么"
)

_FALLBACK_INTENTS = ["形态学特征", "诊断标准"]

_TERMS_FILE = os.getenv("RAG_TERMS_FILE", "configs/rag_terms.yaml")

def _compile_intents(rows) -> list[tuple[re.Pattern, list[str]]]:
    """yaml 的 intent 段 → (编译后正则, 术语) 列表。非法正则跳过并告警。"""
    out: list[tuple[re.Pattern, list[str]]] = []
    for row in rows or []:
        pat, terms = (row or {}).get("pattern"), (row or {}).get("terms")
        if not pat or not terms:
            continue
        try:
            out.append((re.compile(pat), list(terms)))
        except re.error as e:
            logger.warning("意图规则正则非法，已跳过：%s (%s)", pat, e)
    return out


@lru_cache(maxsize=1)
def _load_cached(mtime: float):
    """mtime 仅作为缓存键；RAG_TERMS_RELOAD=true 时随文件变化重载。"""
    try:
        with open(_TERMS_FILE, encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
    except (OSError, yaml.YAMLError) as e:
        logger.warning("术语词表加载失败（%s），使用内置默认值：%s", _TERMS_FILE, e)
        return ({g: set(ts) for g, ts in _FALLBACK_CANONICAL.items()},
                dict(_FALLBACK_ALIAS), dict(_FALLBACK_TO_LATIN),
                list(_FALLBACK_INTENT_RULES), list(_FALLBACK_INTENTS))

    # yaml 里 canonical 是 list，代码需要 set
    canonical = {g: set(ts or []) for g, ts in (cfg.get("canonical") or {}).items()}
    canonical = {g: ts for g, ts in canonical.items() if ts} or _FALLBACK_CANONICAL

    # 关键校验：alias 的 target 必须在白名单内，否则丢弃（防审校手滑）
    flat = {t for ts in canonical.values() for t in ts}
    raw_alias = dict(cfg.get("alias") or {})
    bad = {a: t for a, t in raw_alias.items() if t not in flat}
    if bad:
        logger.warning("rag_terms.yaml 有 %d 条 alias 指向非法术语，已丢弃：%s",
                       len(bad), bad)
    alias = {a: t for a, t in raw_alias.items() if t in flat}

    return (
        canonical,
        alias,
        dict(cfg.get("to_latin") or _FALLBACK_TO_LATIN),
        _compile_intents(cfg.get("intent")) or _FALLBACK_INTENT_RULES,
        list(cfg.get("default_intents") or _FALLBACK_INTENTS),
    )


def load_terms(path: str | None = None, force: bool = False):
    """返回 (canonical, alias, to_latin, intent_rules, default_intents)。"""
    global _TERMS_FILE
    if path:
        _TERMS_FILE = path
        _load_cached.cache_clear()
    if force:
        _load_cached.cache_clear()
    mtime = 0.0
    if os.getenv("RAG_TERMS_RELOAD", "false").strip().lower() == "true":
        try:
            mtime = os.stat(_TERMS_FILE).st_mtime
        except OSError:
            mtime = 0.0
    return _load_cached(mtime)

CANONICAL_TERMS, _ALIAS, _TO_LATIN, _INTENT_RULES, _DEFAULT_INTENTS = load_terms()
_CANON_FLAT = {t for ts in CANONICAL_TERMS.values() for t in ts}

def matched_terms(query: str) -> list[str]:
    """query 中已命中的规范术语。"""
    q = normalize_query(query)
    return sorted(t for t in _CANON_FLAT if t in q)


def rule_rewrite(query: str) -> list[str]:
    """L1：把口语/变体归一到规范术语，返回需要追加的术语（去重、保序）。"""
    q = normalize_query(query)
    if not q:
        return []

    extra: list[str] = []

    # 先看规范术语是否已在 query 里，已存在的不重复追加
    present = {t for t in _CANON_FLAT if t in q}

    # 别名：长串优先，避免 "核大" 先命中导致 "核增大" 判断失效
    for alias in sorted(_ALIAS, key=len, reverse=True):
        target = _ALIAS[alias]
        if alias in q and target not in present:
            extra.append(target)
            present.add(target)
            REWRITE_HITS[(alias, target)] += 1

    # 规范术语 → 拉丁缩写，补 BM25 召回
    for zh, en in _TO_LATIN.items():
        if zh in present and en.lower() not in q.lower():
            extra.append(en)

    return list(dict.fromkeys(extra))

def rewrite_stats(top: int = 20) -> list[dict]:
    """命中频次，供 /api/system/info 展示。"""
    return [
        {"alias": a, "target": t, "hits": n}
        for (a, t), n in REWRITE_HITS.most_common(top)
    ]

def detect_intents(query: str) -> list[str]:
    """L2：从 query 措辞判断检索意图。"""
    q = normalize_query(query)
    if not q:
        return []

    if _QUESTION_RE.search(q):
        core = _QUESTION_RE.sub("", q).strip()
        if len(core) <= 16:          # 去掉疑问词后只剩术语本身
            return []

    for pattern, intents in _INTENT_RULES:
        if pattern.search(q):
            return intents
    return _DEFAULT_INTENTS


def expand_queries(query: str, max_variants: int = 3, extra: list[str] | None = None) -> list[str]:
    """L2：生成多查询变体。第一个永远是原始 query。"""
    q = normalize_query(query)
    if not q:
        return []

    if extra is None:
        extra = rule_rewrite(q)
    
    variants = [q]
    if extra:
        variants.append(f"{q} {' '.join(extra)}")

    for intent in detect_intents(q):
        variants.append(f"{q} {intent}")
        if len(variants) >= max_variants:
            break

    return list(dict.fromkeys(variants))[:max_variants]


def validate_terms(terms: list[str], max_terms: int = 3) -> list[str]:
    """**幻觉闸门**：只保留词表内的术语，其余一律丢弃。"""
    ok = [t for t in terms if t in _CANON_FLAT]
    return list(dict.fromkeys(ok))[:max_terms]