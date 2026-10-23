"""
检查点存储管理
"""

from __future__ import annotations

from contextlib import AsyncExitStack
from typing import Any, Optional

from langgraph.checkpoint.redis import AsyncRedisSaver

from core.redis_client import REDIS_URL

_exit_stack: Optional[AsyncExitStack] = None
_checkpointer_cm: Any = None
graph: Any = None


def _get_exit_stack() -> AsyncExitStack:
    global _exit_stack
    if _exit_stack is None:
        _exit_stack = AsyncExitStack()
    return _exit_stack


def _get_checkpointer_cm():
    global _checkpointer_cm
    if _checkpointer_cm is None:
        _checkpointer_cm = AsyncRedisSaver.from_conn_string(REDIS_URL)
    return _checkpointer_cm


async def init_checkpoint_store(builder=None) -> Any:
    """进入上下文打开连接 → 建索引 → 编译图（幂等性由 lifespan 保证只调一次）。"""
    global graph
    if builder is None:
        # 延迟导入：checkpoint 不依赖节点实现，避免循环 import
        from Agents.diagnosis_graph import get_builder

        builder = get_builder()

    stack = _get_exit_stack()
    checkpointer = await stack.enter_async_context(_get_checkpointer_cm())
    await checkpointer.asetup()
    graph = builder.compile(checkpointer=checkpointer)
    return graph


async def close_checkpoint_store() -> None:
    global _exit_stack, _checkpointer_cm, graph
    if _exit_stack is not None:
        await _exit_stack.aclose()
        _exit_stack = None
    _checkpointer_cm = None
    graph = None


def get_graph() -> Any:
    if graph is None:
        raise RuntimeError("图未初始化：请先在 lifespan 中调用 init_checkpoint_store()")
    return graph