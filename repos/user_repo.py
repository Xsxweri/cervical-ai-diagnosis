"""
账号数据访问层
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from core.db import acquire

# 连续失败达此次数即锁定；15 分钟后自动解锁
MAX_FAILED_ATTEMPTS = 5
LOCK_MINUTES = 15

_PUBLIC_COLUMNS = "id, username, name, department, hospital, role, is_active, created_at, last_login_at"
_ADMIN_COLUMNS = _PUBLIC_COLUMNS + ", failed_attempts, locked_until"

async def get_by_username(username: str) -> Optional[Dict[str, Any]]:
    """登录用：返回含 password_hash 的完整记录。"""
    async with acquire() as conn:
        row = await conn.fetchrow("SELECT * FROM users WHERE username = $1", username)
    return dict(row) if row else None


async def get_by_id(user_id: int) -> Optional[Dict[str, Any]]:
    async with acquire() as conn:
        row = await conn.fetchrow(
            f"SELECT {_PUBLIC_COLUMNS} FROM users WHERE id = $1", user_id
        )
    return dict(row) if row else None


async def list_users() -> List[Dict[str, Any]]:
    async with acquire() as conn:
        rows = await conn.fetch(f"SELECT {_PUBLIC_COLUMNS} FROM users ORDER BY id")
    return [dict(r) for r in rows]


async def create_user(
    username: str,
    password_hash: str,
    name: Optional[str] = None,
    department: Optional[str] = None,
    hospital: Optional[str] = None,
    role: str = "doctor",
) -> int:
    """新建账号，返回 id。用户名重复时抛 UniqueViolationError（由路由翻译成 409）。"""
    async with acquire() as conn:
        return await conn.fetchval(
            """
            INSERT INTO users (username, password_hash, name, department, hospital, role)
            VALUES ($1,$2,$3,$4,$5,$6) RETURNING id
            """,
            username, password_hash, name, department, hospital, role,
        )


async def update_user(user_id: int, **fields) -> bool:
    """局部更新。白名单校验字段名，避免任意列被改。"""
    allowed = {"name", "department", "hospital", "role", "is_active", "password_hash"}
    updates = {k: v for k, v in fields.items() if k in allowed}
    if not updates:
        return False

    sets = ", ".join(f"{k} = ${i + 2}" for i, k in enumerate(updates))
    async with acquire() as conn:
        result = await conn.execute(
            f"UPDATE users SET {sets} WHERE id = $1",
            user_id, *updates.values(),
        )
    return result.endswith(" 1")


async def record_login_success(user_id: int) -> None:
    async with acquire() as conn:
        await conn.execute(
            "UPDATE users SET failed_attempts = 0, locked_until = NULL, "
            "last_login_at = now() WHERE id = $1",
            user_id,
        )


async def record_login_failure(user_id: int) -> None:
    """失败 +1；达阈值则写 locked_until。"""
    async with acquire() as conn:
        await conn.execute(
            """
            UPDATE users SET
                failed_attempts = failed_attempts + 1,
                locked_until = CASE
                    WHEN failed_attempts + 1 >= $2
                    THEN now() + make_interval(mins => $3)
                    ELSE locked_until
                END
            WHERE id = $1
            """,
            user_id, MAX_FAILED_ATTEMPTS, LOCK_MINUTES,
        )


def is_locked(user: Dict[str, Any]) -> bool:
    """锁定判定放进 repo，路由层不关心时间比较细节。"""
    locked_until = user.get("locked_until")
    if not locked_until:
        return False
    if locked_until.tzinfo is None:              # 保险：DB 返回 naive 时按 UTC 解释
        locked_until = locked_until.replace(tzinfo=timezone.utc)
    return datetime.now(timezone.utc) < locked_until

async def list_users_for_admin() -> List[Dict[str, Any]]:
    async with acquire() as conn:
        rows = await conn.fetch(f"SELECT {_ADMIN_COLUMNS} FROM users ORDER BY id")
    return [dict(r) for r in rows]


async def count_active_sysadmins() -> int:
    """「最后一个管理员」保护的数据来源。"""
    async with acquire() as conn:
        return await conn.fetchval(
            "SELECT count(*) FROM users WHERE role = 'admin' AND is_active"
        )


async def get_password_hash(user_id: int) -> Optional[str]:
    """仅用于自助改密时校验原密码，不要在其他地方外传。"""
    async with acquire() as conn:
        return await conn.fetchval(
            "SELECT password_hash FROM users WHERE id = $1", user_id
        )


async def set_password(user_id: int, password_hash: str) -> bool:
    async with acquire() as conn:
        result = await conn.execute(
            "UPDATE users SET password_hash = $2 WHERE id = $1", user_id, password_hash
        )
    return result.endswith(" 1")


async def unlock_user(user_id: int) -> bool:
    """清零失败计数并解除锁定（管理员代为解锁）。"""
    async with acquire() as conn:
        result = await conn.execute(
            "UPDATE users SET failed_attempts = 0, locked_until = NULL WHERE id = $1",
            user_id,
        )
    return result.endswith(" 1")