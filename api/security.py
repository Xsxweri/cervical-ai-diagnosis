"""鉴权辅助、错误脱敏、上传文件内容校验。"""

import io
import logging
import os
import uuid

from fastapi import HTTPException
from PIL import Image

from api.config import ALLOWED_MIME, ALLOWED_EXTS, MAGIC_BYTES

logger = logging.getLogger("agent_api")


def _requester_id(token_payload: dict) -> str:
    """从 JWT payload 提取稳定的操作者标识(优先uid,回退sub)"""
    return str(token_payload.get("uid") or token_payload.get("sub") or "")


def _internal_error(e: Exception) -> HTTPException:
    """内部错误：服务端留全量堆栈 + 关联编号，客户端只拿到编号。

    避免借 500 响应泄露内部路径、SQL、堆栈等实现细节。
    """
    err_id = uuid.uuid4().hex[:12]
    logger.exception(f"[{err_id}] 内部错误: {e}")
    return HTTPException(
        status_code=500, detail=f"服务内部错误，请稍后重试（错误编号 {err_id}）"
    )


def validate_file_content(content: bytes, filename: str, content_type: str | None = None):
    """按扩展名校验 MIME 白名单 + 文件头魔数 + 图片可解码性。"""
    ext = os.path.splitext(filename)[1].lower()

    allowed = ALLOWED_MIME.get(ext)
    if allowed and content_type:
        allowed_set = allowed if isinstance(allowed, (set, frozenset, tuple, list)) else {allowed}
        if content_type not in allowed_set:
            raise HTTPException(status_code=400, detail="Content-Type 与文件类型不符")

    sig = MAGIC_BYTES.get(ext)
    if sig and not any(content.startswith(s) for s in sig):
        raise HTTPException(status_code=400, detail=f"文件内容与扩展名不匹配: {filename}")

    if ext in ALLOWED_EXTS:
        try:
            with Image.open(io.BytesIO(content)) as img:
                img.verify()
        except Exception:
            raise HTTPException(status_code=400, detail=f"文件不是有效图片: {filename}")