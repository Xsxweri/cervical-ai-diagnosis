"""
诊断任务
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from core import task_store

logger = logging.getLogger("tasks.diagnosis")


async def run_diagnosis(
    ctx: Dict[str, Any],
    session_id: str,
    image_path: str,
    owner_id: Optional[str],
) -> Dict[str, Any]:
    """执行一次完整诊断。

    ctx 由 worker 的 on_startup 填充（conv_manager）。
    """
    conv = ctx["conv_manager"]
    await task_store.set_status(session_id, task_store.RUNNING, owner_id=owner_id)

    try:
        state = await conv.start_diagnosis(
            image_path, session_id=session_id, owner_id=owner_id
        )
    except Exception as e:
        logger.exception("会话 [%s] 诊断失败", session_id)
        # 明确记录，前端据此停止轮询并展示原因；不吞异常，让 arq 记为 failed
        await task_store.set_status(session_id, task_store.FAILED, error=str(e))
        raise

    await task_store.set_status(
        session_id,
        task_store.DONE,
        owner_id=owner_id,
        risk_level=state.get("risk_level"),
        needs_clarification=bool(state.get("needs_clarification")),
        is_complete=bool(state.get("is_complete")),
    )
    logger.info("会话 [%s] 诊断任务完成", session_id)
    return {"session_id": session_id, "risk_level": state.get("risk_level")} 