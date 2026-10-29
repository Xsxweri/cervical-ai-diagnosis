"""
任务状态存储
存储任务状态，包括任务状态、当前步骤、当前步骤结果
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, Optional

import redis.asyncio as aioredis

from core.redis_client import get_redis_client

logger = logging.getLogger("task_store")

_TTL = int(os.getenv("TASK_TTL_SECONDS", "86400"))

QUEUED = "queued"
RUNNING = "running"
DONE = "done"
FAILED = "failed"

_TERMINAL = {DONE, FAILED}
HEARTBEAT_KEY = "worker:heartbeat"


def _key(session_id: str) -> str:
    return f"task:{session_id}"


async def set_status(session_id: str, status: str, **extra: Any) -> None:
    payload: Dict[str, Any] = {"status": status, **extra}
    client: aioredis.Redis = await get_redis_client()
    await client.set(_key(session_id), json.dumps(payload, ensure_ascii=False), ex=_TTL)


async def get_status(session_id: str) -> Optional[Dict[str, Any]]:
    client: aioredis.Redis = await get_redis_client()
    raw = await client.get(_key(session_id))
    if not raw:
        return None
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        logger.warning("任务状态损坏，按缺失处理: %s", session_id)
        return None


async def is_terminal(session_id: str) -> bool:
    data = await get_status(session_id)
    return bool(data and data.get("status") in _TERMINAL)