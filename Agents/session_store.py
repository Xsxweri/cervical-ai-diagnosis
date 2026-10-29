"""
会话存储层:序列化、Redis读写、归属校验。
"""

import json
import logging
from typing import Any, Awaitable, Callable, Optional, Union

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    message_to_dict,
)

logger = logging.getLogger(__name__)

SESSION_KEY_PREFIX = "session:"
OWNER_KEY_PREFIX = "session_owner:"
_MESSAGE_KEYS = ("messages", "conversation_history")

RedisFactory = Callable[[], Awaitable[Any]]


# 状态序列化
def _msg_from_dict(d: dict) -> BaseMessage:
    """根据 type 字段还原具体的 Message 类型"""
    msg_type = d.get("type", "")
    content = d.get("data", {}).get("content", "")
    if msg_type == "human":
        return HumanMessage(content=content)
    elif msg_type == "ai":
        return AIMessage(content=content)
    elif msg_type == "system":
        return SystemMessage(content=content)
    return HumanMessage(content=content)


def serialize_state(state: Union[dict, Any]) -> str:
    """将 DiagnosisState 序列化为 JSON 字符串，特殊处理 BaseMessage 列表"""
    data = dict(state)
    data.pop("__interrupt__", None)
    for key in _MESSAGE_KEYS:
        msgs = data.get(key) or []
        data[key] = [message_to_dict(m) for m in msgs if isinstance(m, BaseMessage)]
    return json.dumps(data, ensure_ascii=False, default=str)


def deserialize_state(raw: str) -> dict:
    """从 JSON 字符串反序列化为 Dict"""
    data = json.loads(raw)
    for key in _MESSAGE_KEYS:
        if data.get(key):
            data[key] = [_msg_from_dict(d) for d in data[key]]
    return data


# 会话主数据
async def save_session(
    redis_factory: RedisFactory,
    session_id: str,
    state: Union[dict, Any],
    ttl_seconds: int,
) -> None:
    """序列化状态并写入 Redis，同时设置 TTL"""
    redis_client = await redis_factory()
    await redis_client.setex(
        f"{SESSION_KEY_PREFIX}{session_id}", ttl_seconds, serialize_state(state)
    )


async def load_session(redis_factory: RedisFactory, session_id: str) -> Optional[dict]:
    """从 Redis 读取并反序列化状态；反序列化失败视为会话不存在"""
    redis_client = await redis_factory()
    raw = await redis_client.get(f"{SESSION_KEY_PREFIX}{session_id}")
    if raw is None:
        return None
    try:
        return deserialize_state(raw)
    except Exception as e:
        logger.error(f"反序列化会话 [{session_id}] 失败: {e}")
        return None


async def clear_session(redis_factory: RedisFactory, session_id: str) -> bool:
    """删除会话主数据"""
    redis_client = await redis_factory()
    result = await redis_client.delete(f"{SESSION_KEY_PREFIX}{session_id}")
    return result > 0


# 会话归属
async def save_owner(
    redis_factory: RedisFactory,
    session_id: str,
    owner_id: Optional[str],
    ttl_seconds: int,
) -> None:
    """记录会话归属，TTL 与会话一致；供后续越权校验。"""
    if not owner_id:
        logger.warning(f"会话 [{session_id}] 未提供操作者身份，跳过归属记录")
        return
    redis_client = await redis_factory()
    await redis_client.setex(
        f"{OWNER_KEY_PREFIX}{session_id}", ttl_seconds, str(owner_id)
    )


async def assert_owner(
    redis_factory: RedisFactory,
    session_id: str,
    requester_id: Optional[str],
    fallback: Optional[Callable[[str], Awaitable[Optional[str]]]] = None,
) -> None:
    """
    验证会话归属，会话归属记录在 Redis 中，TTL 与会话一致。
    """
    if not requester_id:
        raise PermissionError("缺少操作者身份，拒绝访问该会话")

    redis_client = await redis_factory()
    owner = await redis_client.get(f"{OWNER_KEY_PREFIX}{session_id}")

    if owner is None and fallback is not None:
        # Redis 已过期（或会话来自冷存储恢复）→ 回查权威来源
        try:
            owner = await fallback(session_id)
        except Exception as e:
            logger.error(f"归属回查失败 [{session_id}]: {e}")
            raise PermissionError("会话归属校验暂时不可用，请稍后重试")

    if owner is None:
        # 无归属记录
        raise PermissionError(
            f"会话 [{session_id}] 无归属记录，拒绝访问（请重新发起诊断）"
        )
    if str(owner) != str(requester_id):
        logger.warning(
            f"越权访问会话 [{session_id}]: owner={owner} requester={requester_id}"
        )
        raise PermissionError("无权访问该会话")