"""
PostgreSQL 连接池。

设计要点
--------
1. 池规模：min_size=2（保证冷启动后首个请求不需建连），max_size 由 PG_POOL_MAX 控制。
2. command_timeout=10：单条命令超时，避免慢查询长期占用连接。
3. timeout=10：建连超时，容器刚启动时 PG 可能未就绪。
4. max_inactive_connection_lifetime=300：空闲连接 5 分钟后回收，
   避免 NAT/防火墙静默断连后继续复用死连接（这是 asyncpg 的等价机制，
   它没有 keepalive 参数 —— 协议由 asyncpg 自行实现，不走 libpq）。
5. max_queries=50000：单连接执行 5 万次查询后重建，防止长期运行的内存缓慢累积。
6. init=_init_connection：注册 jsonb 编解码，业务层可直接读写 dict。
7. 不在导入期建连接：池在 lifespan 中创建/释放，避免 pytest 收集阶段依赖外部服务。

"""

from __future__ import annotations

import json
import logging
import os
from contextlib import asynccontextmanager
from typing import AsyncIterator, Optional

import asyncpg

import core.env  # noqa: F401  确保 DATABASE_URL 从 .env 读取

logger = logging.getLogger("db")

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://cervix:cervix_dev_pwd@127.0.0.1:15432/cervix",
)

_pool: Optional[asyncpg.Pool] = None


async def _init_connection(conn: asyncpg.Connection) -> None:
    """注册 jsonb 编解码：写入时自动 json.dumps，读出时自动 json.loads。"""
    await conn.set_type_codec(
        "jsonb",
        encoder=json.dumps,
        decoder=json.loads,
        schema="pg_catalog",
    )


async def init_pool() -> asyncpg.Pool:
    global _pool
    if _pool is not None:
        return _pool

    _pool = await asyncpg.create_pool(
        dsn=DATABASE_URL,
        min_size=2,
        max_size=int(os.getenv("PG_POOL_MAX", "10")),
        init=_init_connection,
        command_timeout=10,
        timeout=10,
        max_inactive_connection_lifetime=300,
        max_queries=50000,

    )
    logger.info("PostgreSQL 连接池已建立 (max=%s)", os.getenv("PG_POOL_MAX", "10"))
    return _pool


async def close_pool() -> None:
    global _pool
    if _pool is not None:
        await _pool.close()
        _pool = None
        logger.info("PostgreSQL 连接池已关闭")


def get_pool() -> asyncpg.Pool:
    if _pool is None:
        raise RuntimeError("数据库未初始化：请先在 lifespan 中调用 init_pool()")
    return _pool


@asynccontextmanager
async def acquire() -> AsyncIterator[asyncpg.Connection]:
    """从池中借一个连接，退出时自动归还。"""
    async with get_pool().acquire() as conn:
        yield conn


async def ping() -> bool:
    """健康检查用；不抛异常。"""
    try:
        async with acquire() as conn:
            await conn.fetchval("SELECT 1")
        return True
    except Exception:
        return False