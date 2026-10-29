"""
运行时磁盘/内存治理：清理空闲 checkpoint 与过期热力图。

checkpoint 由 langgraph-checkpoint-redis 创建，**不带 TTL**,
不清理会持续累积。这里按「最后访问时间」判定空闲。

用法：
    python scripts/cleanup_runtime.py            # 预览
    python scripts/cleanup_runtime.py --apply
"""
from __future__ import annotations

import argparse
import asyncio
import os
import time
from pathlib import Path

import core.env  # noqa: F401  确保 .env 已加载

import redis.asyncio as aioredis
from redis.exceptions import RedisError

# 项目根仅用于定位 outputs/gradcam，与 import 无关
_ROOT = Path(__file__).resolve().parent.parent

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:16379")
IDLE_H = int(os.getenv("CHECKPOINT_IDLE_HOURS", "48"))
GRADCAM_DAYS = int(os.getenv("GRADCAM_KEEP_DAYS", "7"))
GRADCAM_DIR = _ROOT / "outputs" / "gradcam"

# langgraph-checkpoint-redis 的 key 家族。
# 实测确认方式：
#   redis-cli -p 16379 --scan --pattern 'checkpoint*' | Select-Object -First 5
CHECKPOINT_PREFIXES = (
    "checkpoint:",
    "checkpoint_blob:",
    "checkpoint_write:",
    "checkpoint_ns:",
)


def _thread_id_of(key: str) -> str | None:
    """从 checkpoint key 里取 thread_id（即 session_id）。

    格式：<prefix>:<thread_id>:<...>
    thread_id 为 UUID，不含冒号，因此按 ':' 切分取第 2 段是安全的。
    """
    parts = key.split(":")
    return parts[1] if len(parts) >= 2 and parts[1] else None


async def cleanup_checkpoints(apply: bool) -> None:
    r = aioredis.from_url(REDIS_URL, decode_responses=True)
    try:
        threshold = IDLE_H * 3600
        scanned = 0
        candidates = 0
        deleted = 0

        for pfx in CHECKPOINT_PREFIXES:
            async for key in r.scan_iter(f"{pfx}*", count=500):
                scanned += 1
                tid = _thread_id_of(key)
                if not tid:
                    continue

                # 护栏 1：会话仍存活 → 绝不删
                if await r.exists(f"session:{tid}"):
                    continue

                # 护栏 2：空闲时长足够
                try:
                    idle = await r.object("idletime", key)
                except RedisError:
                    idle = None
                if idle is None or idle < threshold:
                    continue

                candidates += 1
                if apply:
                    await r.delete(key)
                    deleted += 1

        print(
            f"[checkpoint] 扫描 {scanned} 个 key，"
            f"会话已过期且空闲 >{IDLE_H}h 的 {candidates} 个"
            f"{f'（已删 {deleted}）' if apply else '（预览）'}"
        )
    finally:
        await r.aclose()


def cleanup_gradcam(apply: bool) -> None:
    if not GRADCAM_DIR.is_dir():
        print(f"[gradcam] 目录不存在，跳过: {GRADCAM_DIR}")
        return

    cutoff = time.time() - GRADCAM_DAYS * 86400
    n = 0
    for f in GRADCAM_DIR.glob("*.png"):
        if f.stat().st_mtime < cutoff:
            n += 1
            if apply:
                f.unlink(missing_ok=True)
    print(
        f"[gradcam] 超过 {GRADCAM_DAYS} 天的文件 {n} 个"
        f"{'（已删除）' if apply else '（预览）'}"
    )


async def main(apply: bool) -> None:
    await cleanup_checkpoints(apply)
    cleanup_gradcam(apply)
    if not apply:
        print("\n[dry-run] 未做任何修改。加 --apply 执行")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--apply", action="store_true", help="真正执行（默认预览）")
    asyncio.run(main(p.parse_args().apply))
