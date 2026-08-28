import os
import redis
import redis.asyncio as aioredis
from dotenv import load_dotenv

load_dotenv()

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")

#创建 Redis 连接池
redis_pool = aioredis.ConnectionPool.from_url(
    REDIS_URL,
    encodings="utf-8",
    decode_responses=True,
    max_connections=20
)

_sync_pool = redis.ConnectionPool.from_url(
    REDIS_URL,
    encoding="utf-8",
    decode_responses=True,
    max_connections=20
)

#获取 redis 客户端实例
async def get_redis_client():
    return aioredis.Redis(connection_pool=redis_pool)

def get_sync_redis_client():
    return redis.Redis(connection_pool=_sync_pool)
