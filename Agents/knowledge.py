"""
医学知识库RAG生命周期管理。
"""

from __future__ import annotations

import asyncio
import glob
from typing import Any, List, Optional

from services.rag_service import MedicalKnowledgeBase


class UnavailableKnowledgeBase:
    """RAG 不可用时的空实现：检索一律返回空，避免上层调用崩溃。"""

    async def retrieve(self, query: str, k: int = 5) -> List[Any]:
        return []

    async def retrieve_by_classification(self, classification: str, k: int = 3) -> List[Any]:
        return []

    async def retrieve_by_morphology(self, morphology_features: dict, k: int = 3) -> List[Any]:
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


async def init_kb(
    pdf_dir: str = "./Cervical_cytology",
    persist_dir: str = "./knowledge_db",
) -> Any:
    """构建/加载知识库。任何失败都降级为「无 RAG」，绝不阻断服务启动。"""
    global kb
    try:
        # 嵌入模型加载是重同步 IO，必须放到线程里，否则阻塞事件循环
        kb = await asyncio.to_thread(MedicalKnowledgeBase, persist_dir)
        pdf_files = glob.glob(f"{pdf_dir}/*.pdf")
        if pdf_files:
            print(f"[RAG] 发现 {len(pdf_files)} 篇医学文献，正在检查知识库...")
            chunk_count = await kb.load_documents(pdf_files)
            print(f"[RAG] 知识库就绪，共 {chunk_count} 个文本块")
        else:
            print("[RAG] 未找到 PDF 文献，RAG 检索将返回空结果")
    except Exception as e:
        print(f"[RAG] 知识库初始化失败，RAG 降级为不可用: {e}")
        kb = UnavailableKnowledgeBase()
    return kb


def reset_kb() -> None:
    """测试用：清空单例。"""
    global kb
    kb = None