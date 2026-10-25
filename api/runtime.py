"""
运行期单例:限流器、WebSocket 连接池、DI 工厂、时空上下文。
"""

import os
from datetime import datetime
from functools import lru_cache
from zoneinfo import ZoneInfo

from fastapi import Request, WebSocket
from slowapi import Limiter
from slowapi.util import get_remote_address

from Agents.conversation import ConversationManager
from api.config import SERVICE_LOCATION

TRUST_PROXY = os.getenv("TRUST_PROXY", "false").strip().lower() in {"1", "true", "yes", "on"}

def _client_key(request: Request) -> str:
    """限流键：代理后可取真实客户端 IP，否则用 TCP 对端地址。"""
    if TRUST_PROXY:
        xff = request.headers.get("x-forwarded-for", "")
        if xff:
            # 取最左侧（最初发起方）；若代理链不可信，应改为取右起第 N 段
            return xff.split(",")[0].strip()
    return get_remote_address(request)

# 限流器：按客户端 IP 计数
limiter = Limiter(key_func=_client_key)


class ConnectionManager:
    """WebSocket 连接池（防止内存泄漏：断开时显式移除 session）。"""

    def __init__(self):
        self.active_connections: dict[str, WebSocket] = {}

    async def connect(self, session_id: str, websocket: WebSocket, subprotocol: str | None = None):
        await websocket.accept(subprotocol=subprotocol if subprotocol == "bearer" else None)
        self.active_connections[session_id] = websocket

    def disconnect(self, session_id: str):
        if session_id in self.active_connections:
            del self.active_connections[session_id]

    async def send_progress(self, session_id: str, message: dict):
        if session_id in self.active_connections:
            await self.active_connections[session_id].send_json(message)


manager = ConnectionManager()


def build_time_location() -> str:
    """当前时空上下文，注入追问 Prompt；时区可由 SERVICE_TIMEZONE 覆盖。

    冒号必须用全角，与 `Agents/followup_prompt.py` 的默认值保持一致。
    """
    tz = ZoneInfo(os.getenv("SERVICE_TIMEZONE", "Asia/Shanghai"))
    now = datetime.now(tz)
    weekday = "星期" + "一二三四五六日"[now.weekday()]
    return (
        f"<time_location>\n"
        f"当前时间：{now.strftime('%Y-%m-%d')} {weekday}\n"
        f"当前地点：{SERVICE_LOCATION}\n"
        f"</time_location>"
    )


@lru_cache()
def get_conversation_manager() -> ConversationManager:
    """依赖注入用单例；lru_cache 保证整个进程只建一个实例。"""
    return ConversationManager()