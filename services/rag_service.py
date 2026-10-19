from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
import os
import torch
import asyncio
from typing import List, Dict, Any
from concurrent.futures import ThreadPoolExecutor

vector_store_executor = ThreadPoolExecutor(max_workers=2)

# 定义知识库类
class MedicalKnowledgeBase:
    """
    RAG医学诊断知识库，用于存储和检索医学诊断相关的知识和信息。
    该知识库包含了各种医学文献、指南、病例分析等内容，支持基于检索增强生成（RAG）的问答和诊断流程。
    """
    MANIFEST_NAME = "kb_manifest.json"
    CORPUS_VERSION = 1 
    def __init__(self, persist_dir="./knowledge_db"):
        self.persist_dir = persist_dir
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model_name = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")

        self.embeddings = None
        self.embedding_error = None
        self.vectorstore = None
        self._load_vectorstore()

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
        import hashlib, json
        items = []
        for p in sorted(pdf_paths):
            try:
                st = os.stat(p)
                items.append(f"{os.path.basename(p)}:{st.st_size}:{int(st.st_mtime)}")
            except OSError:
                items.append(f"{os.path.basename(p)}:missing")
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

    def _write_manifest(self, fingerprint: str, chunk_count: int):
        import json
        os.makedirs(self.persist_dir, exist_ok=True)
        with open(self._manifest_path(), "w", encoding="utf-8") as f:
            json.dump({"fingerprint": fingerprint, "chunks": chunk_count},
                      f, ensure_ascii=False, indent=2)

    def _reset_vectorstore(self):
        """删除旧 collection，避免旧语料残留稀释检索结果。"""
        if self.vectorstore is not None:
            try:
                self.vectorstore.delete_collection()
            except Exception as e:
                print(f"[RAG] 删除旧 collection 失败（将直接重建）: {e}")
        self.vectorstore = None

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
        
    async def load_documents(self, pdf_paths: list[str], force: bool = False) -> int:
        """
        异步加载 PDF 文档，构建或更新向量库。
        如果 force=True，则强制重建向量库；否则仅在语料指纹不一致时重建。
        返回语料库中的文本块数量。
        """
        fingerprint = self._corpus_fingerprint(pdf_paths)
        manifest = self._read_manifest()

        if (not force and manifest
                and manifest.get("fingerprint") == fingerprint
                and self.vectorstore is not None):
            n = manifest.get("chunks", 0)
            actual = self._actual_count()
            if actual >= 0 and actual != n:
                # manifest 与实际库存不符（历史重复灌入遗留）→ 自动重建
                print(f"[RAG]库内实际 {actual} 条 ≠ manifest {n} 条，自动重建...")
                force = True
            else:
                print(f"[RAG] 知识库已是最新，跳过重建（{n} 个文本块）")
                return n

        if force or (manifest and manifest.get("fingerprint") != fingerprint):
            print("[RAG]重建知识库..." if force else "[RAG]检测到文献变更，重建知识库...")
            await asyncio.get_event_loop().run_in_executor(
                vector_store_executor, self._reset_vectorstore)

        loop = asyncio.get_event_loop()
        chunks = await loop.run_in_executor(None, self._process_documents_sync, pdf_paths)
        if self.vectorstore is None:
            await loop.run_in_executor(vector_store_executor, self._create_vectorstore_sync, chunks)
        else:
            await loop.run_in_executor(vector_store_executor, self.vectorstore.add_documents, chunks)
            await loop.run_in_executor(vector_store_executor, self.vectorstore.persist)

        self._write_manifest(fingerprint, len(chunks))
        return len(chunks)

    def _process_documents_sync(self, pdf_paths: List[str]) -> List:
        """同步处理文档的私有方法"""
        documents = []
        failed = []
        for path in pdf_paths:
            try:
                loader = PyPDFLoader(path)
                documents.extend(loader.load())
            except Exception as e:
                failed.append({"path": path, "error": str(e)})
                print(f"PDF加载失败: {path} - {e}")

        if not documents:
            raise ValueError(f"所有PDF加载失败: {failed}")

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", "。", "；", " ", ""]
        )
        return text_splitter.split_documents(documents)

    def _create_vectorstore_sync(self, chunks):
        """同步创建向量库的私有方法"""
        self.vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            persist_directory=self.persist_dir
        )

    async def retrieve(self, query: str, k: int = 5):
        """异步检索接口，检索相关知识"""
        if not self.vectorstore:
            return []

        loop = asyncio.get_event_loop()

        # 将相似度搜索放到专用线程池执行
        results = await loop.run_in_executor(
            vector_store_executor,
            self.vectorstore.similarity_search_with_score,
            query,
            k
        )

        return [
            {
                "content": doc.page_content,
                "source": doc.metadata.get("source", "未知"),
                "page": doc.metadata.get("page", 0),
                "score": round(score, 4)
            }
            for doc, score in results
        ]

    async def retrieve_by_classification(self, classification: str, k: int = 3):
        """根据分类结果检索对应知识"""
        query_map = {
            # ===正常/异常，粗粒度分类===
            "normal": "正常宫颈鳞状上皮细胞形态学特征 NILM 筛查随访建议",
            "abnormal": "宫颈细胞学异常 TBS分类 ASC-US LSIL HSIL 分流处理原则",

            # ===形态学特征，细粒度亚型（按病理意义分组）===
            # 感染/反应性改变
            "Dyskeratotic": "角化不良细胞 HPV感染相关细胞学特征 LSIL诊断标准",
            "Koilocytotic": "挖空细胞 非典型角化 鉴别HSIL与反应性改变",

            # 化生/良性变异
            "Metaplastic": "鳞状化生细胞 移行区细胞特征 与HSIL鉴别要点",
            "Parabasal": "副基底层细胞 萎缩性阴道炎 低雌激素状态细胞学表现",

            # 成熟度分层
            "Superficial-Intermediate": "表层及中层鳞状上皮细胞 成熟指数评估 激素水平判读"
        }
        query = query_map.get(classification)
        if query is None:
            # 未知分类直接使用原始标签作为查询，避免静默回退到无意义字符串
            print(f"未知分类 '{classification}'，使用原始标签检索")
            query = f"宫颈细胞学 {classification} 形态学特征及临床意义"

        return await self.retrieve(query, k=k)

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

        # 扁平化结果并去重
        results = [item for sublist in results_list for item in sublist]

        seen_sources = set()
        unique_results = []
        for item in results:
            source_key = f"{item['source']}:{item['page']}"
            if source_key not in seen_sources:
                seen_sources.add(source_key)
                unique_results.append(item)

        return unique_results[:k]