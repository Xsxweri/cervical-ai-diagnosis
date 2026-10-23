"""
医疗合规审计日志：记录 IP、方法、路径、状态码、延迟
"""

import os
import time
import uuid

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from api.logging_setup import audit_logger, logger


def cors_headers_for(request: Request) -> dict:
    """为越过 CORSMiddleware 的响应（如 500）按白名单手动补 CORS 头。"""
    origin = request.headers.get("origin")
    if not origin:
        return {}
    allowed = [o.strip() for o in os.getenv("CORS_ORIGINS", "").split(",") if o.strip()]
    if origin not in allowed:
        return {}
    return {"Access-Control-Allow-Origin": origin, "Vary": "Origin"}


async def audit_log_middleware(request: Request, call_next):
    """记录 IP、操作者、方法、路径、状态码、延迟。"""
    start = time.perf_counter()
    client_ip = request.client.host if request.client else "unknown"

    def _who() -> str:
        """操作者身份；未认证请求记为 anonymous。"""
        return getattr(request.state, "requester_id", "") or "anonymous"

    try:
        response = await call_next(request)
    except Exception as e:
        err_id = uuid.uuid4().hex[:12]
        latency = (time.perf_counter() - start) * 1000
        logger.exception(f"[{err_id}] 未捕获异常: {e}")
        audit_logger.error(
            f"operator={_who()} ip={client_ip} {request.method} {request.url.path} "
            f"500 {latency:.1f}ms err={err_id}"
        )
        return JSONResponse(
            status_code=500,
            content={"detail": f"服务内部错误，请稍后重试（错误编号 {err_id}）"},
            headers=cors_headers_for(request),
        )

    latency = (time.perf_counter() - start) * 1000
    audit_logger.info(
        f"operator={_who()} ip={client_ip} {request.method} {request.url.path} "
        f"{response.status_code} {latency:.1f}ms"
    )
    return response


def register_http_middleware(app: FastAPI) -> None:
    """按固定顺序注册中间件，勿调换。"""
    # 1) 审计（外层为 CORS，故先注册）
    app.middleware("http")(audit_log_middleware)

    # 2) CORS 白名单
    app.add_middleware(
        CORSMiddleware,
        allow_origins=os.getenv("CORS_ORIGINS", "http://localhost:3000").split(","),
        allow_methods=["*"],
        allow_headers=["*"],
    )