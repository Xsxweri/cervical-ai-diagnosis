import uuid

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded

# --- 门面：tests/test_unit.py 从 agent_api 导入这些名字 ---
from api.config import (  # noqa: F401
    ALLOWED_EXTS,
    ALLOWED_MIME,
    LOG_DIR,
    MAGIC_BYTES,
    MAX_BATCH_FILES,
    MAX_SINGLE_FILE,
    MAX_ZIP_FILES,
    MAX_ZIP_FILE,
    MAX_ZIP_UNCOMPRESSED,
    SERVICE_LOCATION,
    UPLOAD_CLEANUP_INTERVAL,
    UPLOAD_DIR,
    UPLOAD_MAX_AGE_SECONDS,
)
from api.lifespan import lifespan
from api.logging_setup import audit_logger, logger  # noqa: F401
from api.middleware import cors_headers_for, register_http_middleware
from api.routes import accounts_router, agent_router, auth_router, vision_router
from api.runtime import limiter
from api.schemas import _agent_payload  # noqa: F401
from api.security import _requester_id, validate_file_content  # noqa: F401
from api.runtime import build_time_location  # noqa: F401

# ==== app 装配 ====
app = FastAPI(title="AI Agent宫颈病理诊断系统", version="2.0", lifespan=lifespan)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    err_id = uuid.uuid4().hex[:12]
    logger.exception(f"[{err_id}] 未捕获异常: {exc}")
    return JSONResponse(
        status_code=500,
        content={"detail": f"服务内部错误，请稍后重试（错误编号 {err_id}）"},
        headers=cors_headers_for(request),
    )


# 顺序敏感：先审计、后 CORS（见 api/middleware.py 说明）
register_http_middleware(app)

# ==== 路由注册 ====
app.include_router(auth_router)
app.include_router(accounts_router)
app.include_router(vision_router)
app.include_router(agent_router)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "agent_api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        reload_excludes=["logs/*", "uploads/*", "knowledge_db/*", "outputs/*", "runs/*", "*.log"],
    )