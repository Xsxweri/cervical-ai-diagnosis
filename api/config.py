"""
配置文件
"""

import os
import core.env  # noqa: F401


SERVICE_LOCATION = os.getenv("SERVICE_LOCATION", "未知地点")

# --- 日志目录（建目录的动作留给 logging_setup，避免 import 期副作用）---
LOG_DIR = os.path.abspath("./logs")

# --- 上传与清理 ---
UPLOAD_DIR = os.path.abspath("./uploads")
UPLOAD_MAX_AGE_SECONDS = int(os.getenv("UPLOAD_MAX_AGE_SECONDS", str(6 * 3600)))
UPLOAD_CLEANUP_INTERVAL = int(os.getenv("UPLOAD_CLEANUP_INTERVAL", "1800"))

# --- 批量 / 压缩资源上限 ---
MAX_BATCH_FILES = 10
MAX_ZIP_FILES = 20
MAX_SINGLE_FILE = 30 * 1024 * 1024          # 单文件 30MB
MAX_ZIP_FILE = 50 * 1024 * 1024             # zip 50MB
MAX_ZIP_UNCOMPRESSED = 200 * 1024 * 1024    # 解压后 200MB

# --- 文件类型白名单 ---
ALLOWED_EXTS: tuple[str, ...] = (".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp")

# 文件头魔数：键必须带前导点（tests/test_unit.py 有此回归断言）
MAGIC_BYTES: dict[str, list[bytes]] = {
    ".jpg": [b"\xff\xd8\xff"],
    ".jpeg": [b"\xff\xd8\xff"],
    ".png": [b"\x89PNG\r\n\x1a\n"],
    ".tif": [b"II*\x00", b"MM\x00*"],
    ".tiff": [b"II*\x00", b"MM\x00*"],
    ".bmp": [b"BM"],
}

# 允许的 MIME 类型
ALLOWED_MIME: dict[str, set[str]] = {
    ".jpg":  {"image/jpeg"},
    ".jpeg": {"image/jpeg", "image/jpg"},
    ".png":  {"image/png"},
    ".tif":  {"image/tiff"},
    ".tiff": {"image/tiff", "image/x-tiff"},
    ".bmp":  {"image/bmp", "image/x-bmp", "image/x-ms-bmp"},
}

GENERIC_MIMES: set[str] = {
    "",
    "application/octet-stream",
    "binary/octet-stream",
}