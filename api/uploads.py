"""
uploads目录的TTL清理:防止上传文件只增不减占满磁盘
"""

import asyncio
import os
import time

from api.config import UPLOAD_CLEANUP_INTERVAL, UPLOAD_DIR, UPLOAD_MAX_AGE_SECONDS
from api.logging_setup import logger


def cleanup_uploads() -> None:
    """删除 uploads/ 中超过 UPLOAD_MAX_AGE_SECONDS 的文件。"""
    if not os.path.isdir(UPLOAD_DIR):
        return
    now = time.time()
    for name in os.listdir(UPLOAD_DIR):
        path = os.path.join(UPLOAD_DIR, name)
        try:
            if os.path.isfile(path) and now - os.path.getmtime(path) > UPLOAD_MAX_AGE_SECONDS:
                os.remove(path)
        except OSError as e:
            logger.warning(f"清理上传文件失败 {path}: {e}")


async def uploads_cleanup_loop() -> None:
    """周期性清理任务，随应用生命周期启动/取消。"""
    while True:
        try:
            await asyncio.sleep(UPLOAD_CLEANUP_INTERVAL)
            await asyncio.to_thread(cleanup_uploads)
        except asyncio.CancelledError:
            break
        except Exception as e:
            logger.warning(f"uploads 清理任务异常: {e}")