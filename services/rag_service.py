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

    def __init__(self, persist_dir="./knowledge_db"):
        self.persist_dir = persist_dir
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.embeddings = HuggingFaceEmbeddings(
            model_name="BAAI/bge-m3",
            model_kwargs={'device': self.device},
            encode_kwargs={'normalize_embeddings': True} 
        )

        self.vectorstore = None
        self._load_vectorstore()

    def _load_vectorstore(self):
        """同步加载向量库（在构造函数中调用）"""
        if os.path.exists(self.persist_dir):
            try:
                self.vectorstore = Chroma(
                    persist_directory=self.persist_dir,
                    embedding_function=self.embeddings
                )
            except Exception as e:
                print(f"加载向量库失败: {e}")
        else:
            print(f"向量库目录不存在: {self.persist_dir}，将初始化为空库")

    async def load_documents(self, pdf_paths: list[str]) -> int:
        """异步加载文档（主要耗时在磁盘IO和文本切分）"""
        loop = asyncio.get_event_loop()

        # 将耗时的文件读取和切分放到线程池
        chunks = await loop.run_in_executor(None, self._process_documents_sync, pdf_paths)

        if self.vectorstore is None:
            # 如果是新建库，需要同步操作
            await loop.run_in_executor(vector_store_executor, self._create_vectorstore_sync, chunks)
        else:
            await loop.run_in_executor(vector_store_executor, self.vectorstore.add_documents, chunks)
            await loop.run_in_executor(vector_store_executor, self.vectorstore.persist)

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
            "im_Dyskeratotic": "角化不良细胞 HPV感染相关细胞学特征 LSIL诊断标准",
            "im_Koilocytotic": "挖空细胞 非典型角化 鉴别HSIL与反应性改变",

            # 化生/良性变异
            "im_Metaplastic": "鳞状化生细胞 移行区细胞特征 与HSIL鉴别要点",
            "im_Parabasal": "副基底层细胞 萎缩性阴道炎 低雌激素状态细胞学表现",

            # 成熟度分层
            "im_Superficial-Intermediate": "表层及中层鳞状上皮细胞 成熟指数评估 激素水平判读"
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