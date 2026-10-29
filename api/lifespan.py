"""
应用生命周期：模型 / RAG / checkpointer 预热与释放
"""

import os
import asyncio
from contextlib import asynccontextmanager

from fastapi import FastAPI

from api.logging_setup import logger
from api.uploads import uploads_cleanup_loop


@asynccontextmanager
async def lifespan(app: FastAPI):
    from Agents.diagnosis_graph import close_checkpoint_store, init_checkpoint_store, init_kb
    from api.runtime import close_arq_pool, diagnosis_mode, init_arq_pool
    from core.db import close_pool, init_pool
    from services.vision_service import get_model_manager

    if not os.getenv("OPENAI_API_KEY"):
        logger.warning("OPENAI_API_KEY 未设置：追问与报告生成将降级为内置模板")

    mode = diagnosis_mode()
    if mode not in {"sync", "queue"}:
        logger.warning("DIAGNOSIS_MODE=%s 非法，按 sync 处理", mode)
        mode = "sync"

    # 数据库连接
    try:
        await init_pool()
    except Exception as e:
        raise RuntimeError(
            f"PostgreSQL 连接失败，服务拒绝启动: {e}\n"
            f"  请确认容器已运行：docker compose up -d\n"
            f"  连接串见 .env 的 DATABASE_URL（当前默认 127.0.0.1:15432）"
        ) from e
    
    # 视觉模型最慢，放线程避免阻塞事件循环
    await asyncio.to_thread(get_model_manager)
    # RAG 失败降级为「无知识库」，不阻断启动
    await init_kb()
    # checkpointer + 编译图
    await init_checkpoint_store()

    if mode == "queue":
        await init_arq_pool()

    cleanup_task = asyncio.create_task(uploads_cleanup_loop())
    try:
        yield
    finally:
        cleanup_task.cancel()
        if mode == "queue":
            await close_arq_pool()
        await close_checkpoint_store()
        await close_pool()