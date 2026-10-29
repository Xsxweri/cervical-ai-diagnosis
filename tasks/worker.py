"""worker 入口"""

from __future__ import annotations

import asyncio
import logging
import os
import time

import core.env  # noqa: F401  —— 必须先于读 REDIS_URL
from arq import cron
from arq.connections import RedisSettings

from core.redis_client import REDIS_URL, get_redis_client
from tasks.jobs import run_diagnosis

logger = logging.getLogger("tasks.worker")

QUEUE = os.getenv("ARQ_QUEUE_NAME", "arq:diagnosis")
HEARTBEAT_KEY = "worker:heartbeat"
HEARTBEAT_TTL = 120      # 秒；心跳过期即判定 worker 不可用


async def heartbeat(ctx: dict) -> None:
    """写心跳。cron 每 30 秒调一次；TTL 120 秒留两次重试余量。"""
    client = await get_redis_client()
    await client.set(HEARTBEAT_KEY, int(time.time()), ex=HEARTBEAT_TTL)


async def on_startup(ctx: dict) -> None:
    """对齐 api/lifespan.py：DB → 视觉模型 → RAG → checkpointer。

    worker 是独立进程，不能依赖 FastAPI 的 lifespan，必须自行初始化。
    """
    from Agents.checkpoint import init_checkpoint_store
    from Agents.conversation import ConversationManager
    from Agents.knowledge import init_kb
    from core.db import init_pool
    from services.vision_service import get_model_manager

    await init_pool()
    # 模型加载最慢（数十秒），放线程避免阻塞 worker 的事件循环
    await asyncio.to_thread(get_model_manager)
    await init_kb()
    await init_checkpoint_store()

    # 立即写一次心跳，避免启动后到首个 cron tick 之间出现"无 worker"假警报
    await heartbeat(ctx)

    ctx["conv_manager"] = ConversationManager()
    logger.info("worker 就绪：DB / 视觉模型 / RAG / checkpointer 已初始化")


async def on_shutdown(ctx: dict) -> None:
    from Agents.checkpoint import close_checkpoint_store
    from core.db import close_pool

    # 主动清心跳：正常停机应立刻表现为"无 worker"，不必等 120 秒 TTL 过期
    try:
        client = await get_redis_client()
        await client.delete(HEARTBEAT_KEY)
    except Exception:
        logger.warning("清理心跳失败（不影响停机）")

    await close_checkpoint_store()
    await close_pool()
    logger.info("worker 已释放资源")


class WorkerSettings:
    functions = [run_diagnosis]
    redis_settings = RedisSettings.from_dsn(REDIS_URL)
    queue_name = QUEUE

    # GPU 推理：并发 2 会显存翻倍，8GB 卡必 OOM
    max_jobs = int(os.getenv("ARQ_MAX_JOBS", "1"))
    job_timeout = int(os.getenv("ARQ_JOB_TIMEOUT", "600"))   # CV + LLM 留足 10 分钟
    max_tries = 1                                            # 见 tasks/jobs.py 说明
    keep_result = 600

    cron_jobs = [cron(heartbeat, second={0, 30})]

    # 注意：这里只能放"引用"，不能放任何 await / 函数调用
    on_startup = on_startup
    on_shutdown = on_shutdown