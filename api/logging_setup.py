"""
日志装配:root handler + 独立审计日志文件。

设计要点：
- `LOG_DIR` 的创建放在本模块中完成，避免重复创建。
- `setup_logging()` 幂等，重复调用不会重复挂 handler。
- `audit_logger.propagate = False`，避免审计日志同时写进 root 的文件。
"""

import logging
import os
from logging.handlers import RotatingFileHandler

from api.config import LOG_DIR

logger = logging.getLogger("agent_api")
audit_logger = logging.getLogger("audit")

_LOGGING_READY = {"done": False}


def setup_logging() -> None:
    if _LOGGING_READY["done"]:
        return

    os.makedirs(LOG_DIR, exist_ok=True)

    fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s")

    root = logging.getLogger()
    root.setLevel(logging.INFO)

    console = logging.StreamHandler()
    console.setFormatter(fmt)
    root.addHandler(console)

    app_file = RotatingFileHandler(
        os.path.join(LOG_DIR, "agent_api.log"),
        maxBytes=10 * 1024 * 1024,
        backupCount=5,
        encoding="utf-8",
    )
    app_file.setFormatter(fmt)
    root.addHandler(app_file)

    logging.getLogger("watchfiles").setLevel(logging.WARNING)

    audit_fmt = logging.Formatter(
        "%(asctime)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S%z",
    )
    audit_handler = RotatingFileHandler(
        os.path.join(LOG_DIR, "audit.log"),
        maxBytes=10 * 1024 * 1024,
        backupCount=30,
        encoding="utf-8",
    )
    audit_handler.setFormatter(audit_fmt)
    audit_logger.addHandler(audit_handler)
    audit_logger.setLevel(logging.INFO)
    audit_logger.propagate = False

    _LOGGING_READY["done"] = True


# 初始化日志
setup_logging()