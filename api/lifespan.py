"""
应用生命周期：模型 / RAG / checkpointer 预热与释放
"""

import asyncio
from contextlib import asynccontextmanager

from fastapi import FastAPI

from api.uploads import uploads_cleanup_loop


@asynccontextmanager
async def lifespan(app: FastAPI):
    from Agents.diagnosis_graph import close_checkpoint_store, init_checkpoint_store, init_kb
    from services.vision_service import get_model_manager

    await asyncio.to_thread(get_model_manager)
    await init_kb()
    await init_checkpoint_store()

    cleanup_task = asyncio.create_task(uploads_cleanup_loop())
    try:
        yield
    finally:
        cleanup_task.cancel()
        await close_checkpoint_store()