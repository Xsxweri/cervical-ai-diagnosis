"""
医学知识库RAG生命周期管理。
"""

from __future__ import annotations

import asyncio
import glob
import os
from typing import Any, List, Optional

from services.rag_service import MedicalKnowledgeBase
from utils.label_queries import warn_uncovered_labels
from utils.mineru_loader import discover_mineru_json


class UnavailableKnowledgeBase:
    """RAG 不可用时的空实现：检索一律返回空，避免上层调用崩溃。"""
    chunk_count = 0
    is_available = False
    reason = ""

    async def retrieve(self, query: str, k: int = 5) -> List[Any]:
        return []

    async def retrieve_by_classification(self, classification: str, k: int = 3) -> List[Any]:
        return []

    async def retrieve_by_morphology(self, morphology_features: dict, k: int = 3) -> List[Any]:
        return []

    async def retrieve_nl(self, query: str, k: int = 5, **kwargs) -> List[Any]:
        return []
    async def retrieve_multi(self, queries, k: int = 5, **kwargs) -> List[Any]:
        return []

    async def load_documents(self, pdf_paths, force: bool = False) -> int:
        return 0


# 保留旧名，供历史 import 兼容
_UnavailableKnowledgeBase = UnavailableKnowledgeBase

kb: Optional[Any] = None


def get_kb() -> Any:
    if kb is None:
        raise RuntimeError("知识库未初始化：请先调用 init_kb()")
    return kb

def kb_status() -> dict:
    """RAG 真实可用性。未初始化 / 降级 / 空库均判为不可用, 绝不抛异常。"""
    if kb is None:
        return {"available": False, "chunks": 0, "reason": "知识库未初始化"}
    if isinstance(kb, UnavailableKnowledgeBase):
        return {"available": False, "chunks": 0,
                "reason": getattr(kb, "reason", "") or "RAG 已降级"}
    try:
        n = int(kb.chunk_count)
    except Exception as e:
        return {"available": False, "chunks": 0, "reason": f"向量库不可读: {e}"}
    if n < 0:
        return {"available": False, "chunks": 0, "reason": "向量库条目数不可读"}
    if n == 0:
        return {"available": False, "chunks": 0, "reason": "知识库为空(未灌入语料)"}
    return {"available": True, "chunks": n, "reason": ""}

async def init_kb(
    pdf_dir: str = "./Cervical_cytology",
    persist_dir: str = "./knowledge_db",
) -> Any:
    """构建/加载知识库。任何失败都降级为「无 RAG」，绝不阻断服务启动。"""
    global kb

    # 启动期自检
    warn_uncovered_labels()

    try:
        # 嵌入模型加载是重同步 IO，必须放到线程里，否则阻塞事件循环
        kb = await asyncio.to_thread(MedicalKnowledgeBase, persist_dir)
        pdf_files = glob.glob(f"{pdf_dir}/*.pdf")
        pdf_files = sorted(glob.glob(f"{pdf_dir}/*.pdf"))
        json_files = discover_mineru_json(pdf_dir) if os.path.isdir(pdf_dir) else []
        sources = pdf_files + json_files
        if sources:
            print(f"[RAG] 发现 {len(pdf_files)} 篇 PDF + {len(json_files)} 篇 MinerU JSON，"
                  f"正在检查知识库...")
            chunk_count = await kb.load_documents(sources)
            print(f"[RAG] 知识库就绪，共 {chunk_count} 个文本块")
        else:
            print("[RAG] 未找到 PDF / MinerU JSON，RAG 检索将返回空结果")
    except Exception as e:
        print(f"[RAG] 知识库初始化失败，RAG 降级为不可用: {e}")
        kb = UnavailableKnowledgeBase()
        kb.reason = f"知识库初始化失败: {e}"
    return kb


def reset_kb() -> None:
    """测试用：清空单例。"""
    global kb
    kb = None