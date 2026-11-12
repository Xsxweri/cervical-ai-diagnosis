import core.env  # noqa: F401
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from utils.mineru_loader import load_mineru_documents
from langchain_core.documents import Document
import os
import torch
import asyncio
import hashlib
import json
import threading
import numpy as np
from typing import List
from concurrent.futures import ThreadPoolExecutor
from collections import OrderedDict
from utils.doc_cleaning import strip_references_pages
from utils.rag_index import atomic_write_json, index_lock, rebuild_reason
from utils.query_rewrite import expand_queries, matched_terms, rule_rewrite
from utils.label_queries import get_classification_query
from utils.retrieval_fusion import (
    BM25, cosine_to_distance, dedupe_knowledge, expand_query, mmr_select, rrf_fuse, tokenize, normalize_query
)

vector_store_executor = ThreadPoolExecutor(max_workers=2)

_DOC_BONUS = {
    "national_guideline": 0.040,   # 国内法定规范
    "standard": 0.040,
    "review": 0.040,               # TBS 是术语唯一定义源，需越过 WHO 体量
    "intl_guideline": 0.020,       # WHO：权威但适用性次于本国规范
    "research": 0.0,
}

# 定义知识库类
class MedicalKnowledgeBase:
    """
    RAG医学诊断知识库, 用于存储和检索医学诊断相关的知识和信息。
    该知识库包含了各种医学文献、指南等内容, 支持基于检索增强生成(RAG)的问答和诊断流程。
    """
    MANIFEST_NAME = "kb_manifest.json"
    CORPUS_VERSION = 4
    def __init__(self, persist_dir="./knowledge_db"):
        self.persist_dir = persist_dir
        _dev = (os.getenv("EMBEDDING_DEVICE") or os.getenv("VISION_DEVICE") or "").strip().lower()
        if _dev in {"cuda", "cpu"}:
            self.device = _dev
        else:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model_name = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")

        self.HYBRID = os.getenv("RAG_HYBRID", "true").strip().lower() in {"1", "true", "yes", "on"}
        self.RRF_K = int(os.getenv("RAG_RRF_K", "60"))
        self.CANDIDATE_POOL = int(os.getenv("RAG_CANDIDATE_POOL", "40"))
        self.MMR = os.getenv("RAG_MMR", "false").strip().lower() in {"1", "true", "yes", "on"}
        self.MMR_LAMBDA = float(os.getenv("RAG_MMR_LAMBDA", "0.7"))
        self.NL_MAX_VARIANTS = int(os.getenv("RAG_NL_MAX_VARIANTS", "3"))
        self.NL_VARIANT_WEIGHT = float(os.getenv("RAG_NL_VARIANT_WEIGHT", "0.5"))
        self.DOC_WEIGHTED = os.getenv("RAG_DOC_WEIGHT", "true").strip().lower() in {"1", "true", "yes", "on"}
        self.MERGE_SAME_PAGE = os.getenv("RAG_MERGE_SAME_PAGE", "true").strip().lower() in {"1", "true", "yes", "on"}

        self.embeddings = None
        self.embedding_error = None
        self.vectorstore = None
        self._count_cache = None
        self._index = None            # 混合检索索引
        self._query_cache = OrderedDict()        # 查询侧缓存
        self.QUERY_CACHE_MAX = int(os.getenv("RAG_QUERY_CACHE", "256"))
        self._load_lock = threading.Lock()      # 进程内串行
        self._index_lock = asyncio.Lock()     # 混合检索索引构建
        self.LOCK_TIMEOUT = float(os.getenv("RAG_INDEX_LOCK_TIMEOUT", "300"))
        self._load_vectorstore()

    def _build_index_sync(self):
        """从 Chroma 拉全量文档 + 向量，构建 BM25 与余弦矩阵。"""
        if self.vectorstore is None:      # 重建窗口内可能已被置 None
            return None
        data = self.vectorstore.get(include=["documents", "metadatas", "embeddings"])
        ids = data.get("ids") or []
        if not ids:
            return None
        docs = data.get("documents") or []
        metas = data.get("metadatas") or []

        embs = data.get("embeddings")
        if embs is None or len(embs) == 0:
            embs = self._ensure_embeddings().embed_documents(docs)
        matrix = np.asarray(embs, dtype="float32")
        # 兜底归一化：点积即余弦
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        matrix = matrix / np.clip(norms, 1e-12, None)

        return {
            "ids": ids,
            "docs": docs,
            "metas": metas,
            "matrix": matrix,
            "bm25": BM25([tokenize(d or "") for d in docs]),
            "count": len(ids),
        }

    async def _ensure_index(self):
        async with self._index_lock:
            if self._index is not None and self._index["count"] == self.chunk_count:
                return self._index
            loop = asyncio.get_running_loop()
            index = await loop.run_in_executor(vector_store_executor, self._build_index_sync)
            self._index = index
            self._query_cache.clear()
            return index
    def _ensure_embeddings(self):
        """惰性加载嵌入模型；失败抛异常，由调用方降级处理。"""
        if self.embeddings is not None:
            return self.embeddings
        if self.embedding_error is not None:
            raise RuntimeError(f"嵌入模型不可用: {self.embedding_error}")
        try:
            self.embeddings = HuggingFaceEmbeddings(
                model_name=self.model_name,
                model_kwargs={'device': self.device},
                encode_kwargs={'normalize_embeddings': True},
            )
        except Exception as e:
            self.embedding_error = e
            raise
        return self.embeddings

    def _manifest_path(self) -> str:
        return os.path.join(self.persist_dir, self.MANIFEST_NAME)

    def _corpus_fingerprint(self, pdf_paths) -> str:
        """语料指纹：文件名 + 大小 + mtime + 切分版本 + 嵌入模型。"""
        items = []
        for p in sorted(pdf_paths):
            try:
                st = os.stat(p)
                items.append(f"{os.path.relpath(p)}:{st.st_size}:{int(st.st_mtime)}")
            except OSError:
                items.append(f"{os.path.relpath(p)}:missing")
        payload = json.dumps({
            "version": self.CORPUS_VERSION,
            "model": self.model_name,
            "files": items,
        }, ensure_ascii=False)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def _read_manifest(self):
        import json
        try:
            with open(self._manifest_path(), encoding="utf-8") as f:
                return json.load(f)
        except (OSError, json.JSONDecodeError):
            return None

    def _reset_vectorstore(self):
        """删除旧 collection，避免旧语料残留稀释检索结果。"""
        if self.vectorstore is not None:
            try:
                self.vectorstore.delete_collection()
            except Exception as e:
                print(f"[RAG] 删除旧 collection 失败（将直接重建）: {e}")
        self.vectorstore = None
        self._count_cache = None
        self._index = None

    def _load_vectorstore(self):
        """加载向量库；失败只降级为"RAG 不可用"，不阻断服务启动。"""
        if not os.path.exists(self.persist_dir):
            print(f"[RAG] 向量库目录不存在: {self.persist_dir}，将初始化为空库")
            return
        try:
            self.vectorstore = Chroma(
                persist_directory=self.persist_dir,
                embedding_function=self._ensure_embeddings(),
            )
        except Exception as e:
            self.embedding_error = e
            self.vectorstore = None
            print(f"[RAG] 加载向量库失败，RAG 降级为不可用: {e}")

    def _actual_count(self) -> int:
        """读取 collection 实际条目数；失败返回 -1。"""
        try:
            return int(self.vectorstore._collection.count())
        except Exception:
            return -1

    @property
    def chunk_count(self) -> int:
        """库内文本块数，返回 -1 表示不可读(与"空库 0 条"区分)"""
        if self.vectorstore is None:
            return -1
        if self._count_cache is not None:
            return self._count_cache
        n = self._actual_count()
        if n >= 0:
            self._count_cache = n
        return n

    @property
    def is_available(self) -> bool:
        """真实可用:向量库已加载且库内确有语料。"""
        return self.vectorstore is not None and self.chunk_count > 0
        
    async def load_documents(self, sources: list[str], force: bool = False) -> int:
        """
        异步加载 PDF / MinerU JSON，构建或更新向量库。
        """
        fingerprint = self._corpus_fingerprint(sources)
        return await asyncio.to_thread(
            self._load_documents_sync, sources, force, fingerprint
        )

    def _load_documents_sync(self, pdf_paths, force: bool, fingerprint: str) -> int:
        # 进程内串行 + 跨进程串行（api 进程与 arq worker 会各自 init_kb）
        lock_path = os.path.join(self.persist_dir, "index.lock")

        with self._load_lock:
            with index_lock(lock_path, self.LOCK_TIMEOUT) as locked:
                actual = self._actual_count() if self.vectorstore is not None else -1
                if not locked:
                    print(f"[RAG] 未取得索引锁，跳过重建（库内 {actual} 条）")
                    return max(actual, 0)

                manifest = self._read_manifest()
                reason = "force=True" if force else rebuild_reason(
                    manifest, fingerprint, actual)
                if reason is None:
                    n = int((manifest or {}).get("chunks") or 0)
                    print(f"[RAG] 知识库已是最新，跳过重建（{n} 个文本块）")
                    return n

                print(f"[RAG] 重建知识库：{reason}")
                self._reset_vectorstore()
                chunks = self._process_documents_sync(pdf_paths)
                self._create_vectorstore_sync(chunks)

                # manifest 最后写：中途崩溃 → 下次 actual≠expected → 自动重建
                atomic_write_json(self._manifest_path(), {
                    "fingerprint": fingerprint, "chunks": len(chunks),
                })
                self._count_cache = None
                self._index = None
                self._query_cache.clear()
                return len(chunks)

    def _process_documents_sync(self, sources: List[str]) -> List:
        """
        同步处理文档。两种来源：

          *.pdf  → PyPDFLoader + 参考文献剥离 + 字符切分
          *.json → MinerU 版面感知解析（已做标题感知分块，图注独立成块）
        """
        documents: List = []
        failed: List[dict] = []
        pdf_paths = [p for p in sources if p.lower().endswith(".pdf")]
        json_paths = [p for p in sources if p.lower().endswith(".json")]
        for path in pdf_paths:
            try:
                loader = PyPDFLoader(path)
                documents.extend(loader.load())
            except Exception as e:
                failed.append({"path": path, "error": str(e)})
                print(f"PDF加载失败: {path} - {e}")

        mineru_docs: List = []
        for path in json_paths:
            try:
                for d in load_mineru_documents(path):
                    mineru_docs.append(
                        Document(page_content=d["text"], metadata=d["metadata"])
                    )
            except Exception as e:
                failed.append({"path": path, "error": str(e)})
                print(f"MinerU JSON 解析失败: {path} - {e}")

        if not documents and not mineru_docs:
            raise ValueError(f"所有文档加载失败: {failed}")

        if documents:
            documents = strip_references_docs(documents)

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", "。", "；", " ", ""]
        )
        chunks = text_splitter.split_documents(documents) + mineru_docs
        return self._apply_doc_meta(chunks, sources)

    def _apply_doc_meta(self, chunks: List, pdf_paths: List[str]) -> List:
        """
        按 corpus_manifest.yaml 给每个 chunk 打 doc_type / lang / year。
        路径可用 RAG_CORPUS_MANIFEST 覆盖；清单缺失时全部降级为 research。
        """
        import yaml
        path = os.getenv("RAG_CORPUS_MANIFEST", "configs/corpus_manifest.yaml")
        try:
            with open(path, encoding="utf-8") as f:
                spec = yaml.safe_load(f) or {}
        except (OSError, yaml.YAMLError) as e:
            print(f"[RAG] 语料清单缺失 {path}({e}), doc_type 加权将不生效")
            return chunks

        meta = {d["file"]: d for d in spec.get("docs") or []}
        hit = 0
        for c in chunks:
            name = os.path.basename(str(c.metadata.get("source", "")))
            info = meta.get(name)
            if info:
                c.metadata["doc_type"] = info.get("doc_type", "research")
                c.metadata["lang"] = info.get("lang", "")
                year = info.get("year")
                if year is not None:
                    c.metadata["year"] = int(year)
                c.metadata["role"] = info.get("role", "")
                hit += 1
            else:
                c.metadata["doc_type"] = "research"
        print(f"[RAG] doc_type 已标注 {hit}/{len(chunks)} 个 chunk")
        return chunks

    def _create_vectorstore_sync(self, chunks):
        """同步创建向量库的私有方法"""
        self._ensure_embeddings()
        self.vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            persist_directory=self.persist_dir
        )
        self._count_cache = None

    async def retrieve(self, query: str, k: int = 5, *, source_filter=None, apply_bias: bool = True):
        """混合检索：向量 + BM25, RRF 融合。

        source_filter: 可选的来源子串列表，如 ["HPV"] 只在该文件名含 HPV 的文献里检索。

        返回项的 score 仍是 **l2 距离**（越小越相关），与改造前语义一致，
        因此 evidence_sufficiency_from / _distance_to_cosine 无需改动。
        """
        store = self.vectorstore
        if store is None or self.chunk_count == 0:
            return []

        q = normalize_query(query)
        if not q:
            return []
        q_bm25 = expand_query(q)
        cache_key = (q, k, tuple(source_filter or ()), apply_bias)
        if (hit := self._query_cache.get(cache_key)) is not None:
            self._query_cache.move_to_end(cache_key)
            return list(hit)

        index = await self._ensure_index()
        if index is None:
            return []

        loop = asyncio.get_running_loop()
        items = await loop.run_in_executor(
            vector_store_executor, self._hybrid_search_sync, index, q, q_bm25, k, source_filter
        )
        items = _apply_doc_bonus(items, self.DOC_WEIGHTED and apply_bias)
        if len(self._query_cache) > 256:
            self._query_cache.clear()
        self._query_cache[cache_key] = items
        self._query_cache.move_to_end(cache_key)
        while len(self._query_cache) > self.QUERY_CACHE_MAX:
            self._query_cache.popitem(last=False)
        return list(items)

    def _hybrid_search_sync(self, index, query, query_bm25, k, source_filter):
        pool = max(self.CANDIDATE_POOL, k * 3)
        qvec = np.asarray(self._ensure_embeddings().embed_query(query), dtype="float32")
        qvec = qvec / max(float(np.linalg.norm(qvec)), 1e-12)

        cos = index["matrix"] @ qvec

        if source_filter:
            pats = [p.lower() for p in source_filter]
            allowed = np.array([
                any(p in str((m or {}).get("source", "")).lower() for p in pats)
                for m in index["metas"]
            ])
        else:
            allowed = np.ones(len(cos), dtype=bool)
        if not allowed.any():
            return []

        vec_order = [int(i) for i in np.argsort(-cos) if allowed[i]][:pool]

        bm25_hits = index["bm25"].search(tokenize(query_bm25), pool * 2)
        bm_order = [i for _, i in bm25_hits if allowed[i]][:pool]
        bm25_score = {i: s for s, i in bm25_hits}

        if self.HYBRID:
            fused = rrf_fuse([vec_order, bm_order], k=self.RRF_K, top_k=pool)
        else:
            fused = [(i, 0.0) for i in vec_order]

        if self.MMR and fused:
            cand = [i for i, _ in fused]
            bonus = dict(fused)
            picked = mmr_select(qvec, index["matrix"][cand], k, self.MMR_LAMBDA)
            fused = [(cand[j], bonus[cand[j]]) for j in picked]

        vec_set, bm_set = set(vec_order), set(bm_order)
        out = []
        for i, rrf in fused[:k]:
            c = float(cos[i])
            meta = index["metas"][i] or {}
            out.append({
                "id": index["ids"][i],
                "content": index["docs"][i],
                "source": meta.get("source", "未知"),
                "page": meta.get("page", 0),
                "doc_type": meta.get("doc_type", ""),
                "block_type": meta.get("block_type", ""),
                "section_path": meta.get("section_path", ""),
                "score": round(cosine_to_distance(c), 4),   # 兼容旧语义
                "cosine": round(max(c, 0.0), 4),
                "bm25": round(bm25_score.get(i, 0.0), 4),
                "rrf": round(rrf, 6),
                "matched_by": [m for m, s in (("vector", vec_set), ("bm25", bm_set))
                               if i in s],
            })
        return out

    async def retrieve_nl(self, query: str, k: int = 5, *, source_filter=None):
        """
        自由文本入口：规则改写 → 多查询 → RRF 融合。
        """
        q = normalize_query(query)
        if not q:
            return []

        extra = rule_rewrite(q)
        variants = expand_queries(q, max_variants=self.NL_MAX_VARIANTS, extra=extra)

        # 规则层零覆盖 → 可能是纯口语，交给 LLM 兜底
        if not extra and not matched_terms(q):
            from Agents.query_rewrite_llm import rewrite_query_llm
            from utils.query_rewrite import _CANON_FLAT

            terms = await rewrite_query_llm(q, "、".join(sorted(_CANON_FLAT)))
            if terms:
                variants.append(f"{q} {' '.join(terms)}")

        return await self.retrieve_multi(variants, k=k, source_filter=source_filter)

    async def retrieve_multi(self, queries: list[str], k: int = 5, *, source_filter=None):
        """多查询检索：每个 query 各跑一遍混合检索，按 doc id 做 RRF 融合。"""
        if not queries:
            return []

        pool_items: dict[str, dict] = {}
        orders: list[list[str]] = []
        for q in queries:
            items = await self.retrieve(q, k=max(k, self.CANDIDATE_POOL), source_filter=source_filter, apply_bias=False)
            if not items:
                continue
            orders.append([it["id"] for it in items if it.get("id")])
            for it in items:
                if it.get("id"):
                    pool_items.setdefault(it["id"], it)

        if not orders:
            return []

        weights = [1.0] + [self.NL_VARIANT_WEIGHT] * (len(orders) - 1)
        fused = rrf_fuse(orders, k=self.RRF_K, top_k=max(k, self.CANDIDATE_POOL), weights=weights)
        out = []
        for doc_id, score in fused:
            item = dict(pool_items[doc_id])
            item["rrf"] = round(score, 6)      # 覆盖单查询的 rrf，语义为"跨查询融合分"
            item["fused_rank"] = round(score, 6)
            out.append(item)
        ranked = _apply_doc_bonus(out, self.DOC_WEIGHTED)
        if self.MERGE_SAME_PAGE:
            ranked = _merge_same_page(ranked)[:k]
        return ranked[:k]

    async def retrieve_by_classification(self, classification: str, k: int = 3):
        """根据分类结果检索对应知识"""
        return await self.retrieve(get_classification_query(classification), k=k)

    async def retrieve_by_morphology(self, morphology_features: dict, k: int = 3):
        """根据形态学特征检索相关知识"""
        queries = []
        if morphology_features.get("nuclear_cytoplasmic_ratio", 0) > 0.5:
            queries.append("核质比增高,细胞异型性诊断意义")
        if morphology_features.get("circularity", 1) < 0.7:
            queries.append("细胞形态不规则病理意义")

        if not queries:
            return []

        # 并发执行多个检索任务
        tasks = [self.retrieve(q, k=2) for q in queries]
        results_list = await asyncio.gather(*tasks)

        # 跨查询合并:按 source:page 去重,保留最相关者,再取 top-k
        merged = [item for sublist in results_list for item in sublist]
        return dedupe_knowledge(merged, sort_by_score=True)[:k]

def _apply_doc_bonus(items: List[dict], weighted: bool) -> List[dict]:
    """
    按 doc_type 给结果加一个小的余弦偏移，然后重排。

    只在「最终输出」施加一次：
    - retrieve()     单查询路径 → 直接输出，施加
    - retrieve_multi 多查询路径 → 对融合结果施加，不对变体施加

    归一化理由: rrf 量级 ~0.01-0.03，bonus 是 0.04，直接相加会让bonus 完全支配排序。
    """
    if not items:
        return items

    has_fused = any(it.get("fused_rank") is not None for it in items)
    base_key = "fused_rank" if has_fused else "cosine"
    values = [float(it.get(base_key) or 0.0) for it in items]
    span = max(values) or 1.0

    for it, v in zip(items, values):
        base = v / span if span else 0.0            # 归一化到 [0,1]
        bonus = _DOC_BONUS.get(it.get("doc_type"), 0.0) if weighted else 0.0
        it["cos_rank"] = round(base + bonus, 6)

    return sorted(items, key=lambda x: -x["cos_rank"])

def _merge_same_page(items: List[dict], max_chars: int = 2400) -> List[dict]:
    """把同一 source+page 的多个 chunk 合并为一条。

    为什么需要：MinerU 按 section 切分，一页内的表格/流程常被拆成多块。
    检索只命中其中一块时，报告只能看到残缺逻辑 ——
    场景完全不同，但词面高度重叠，向量与 BM25 都无法区分。
    合并后 LLM 能看到完整的两条分支，自行按前提选择。

    排序：保留组内最高名次的位置与分数；内容按原顺序拼接（受 max_chars 限制）。
    """
    if not items:
        return items

    groups: dict[tuple, int] = {}          # (source, page) -> 首次出现的下标
    content: dict[tuple, list[str]] = {}
    for it in items:
        key = (str(it.get("source") or ""), it.get("page"))
        if key not in groups:
            groups[key] = len(groups)
            content[key] = []
        content[key].append(it.get("content") or "")

    merged: List[dict] = []
    emitted: set[tuple] = set()
    for it in items:                       # 保持原顺序
        key = (str(it.get("source") or ""), it.get("page"))
        if key in emitted:
            continue
        emitted.add(key)
        body = "\n\n".join(p for p in content[key] if p)
        if len(body) > max_chars:
            body = body[:max_chars] + "…"
        item = dict(it)
        item["content"] = body
        item["merged_chunks"] = len(content[key])
        merged.append(item)
    return merged

def strip_references_docs(docs: List) -> List:
    """按 source 分组剥离参考文献，返回新的 Document 列表。"""
    groups: dict[str, List] = {}
    order: List[str] = []
    for d in docs or []:
        key = str((d.metadata or {}).get("source", ""))
        if key not in groups:
            groups[key] = []
            order.append(key)
        groups[key].append(d)

    out: List = []
    for key in order:
        group = groups[key]
        kept, note = strip_references_pages([d.page_content for d in group])
        if note:
            print(f"[RAG] {os.path.basename(key)} {note}")
        for idx, text in enumerate(kept):
            # kept 由 pages[:i] + [标题前的正文] 构成，下标与 group 对齐
            out.append(Document(page_content=text,
                                metadata=dict(group[idx].metadata or {})))
    return out