"""认证路由：医生登录。"""

from fastapi import APIRouter, HTTPException, Request
from werkzeug.security import check_password_hash, generate_password_hash

import repos.user_repo as user_repo
from api.logging_setup import audit_logger, logger
from api.runtime import limiter
from api.schemas import LoginRequest
from core.auth import create_token

router = APIRouter(tags=["auth"])
_DUMMY_HASH = generate_password_hash("timing-equalizer", method="scrypt")

@router.post("/api/auth/login", summary="医生登录")
@limiter.limit("5/minute")
async def login(request: Request, req: LoginRequest):
    user = None
    try:
        user = await user_repo.get_by_username(req.username)
    except Exception as e:
        # 数据库不可用
        logger.exception(f"账号库不可用: {e}")
        raise HTTPException(status_code=503, detail="账号服务暂不可用，请稍后重试")

    if user is None:
        check_password_hash(_DUMMY_HASH, req.password)
        raise HTTPException(status_code=401, detail="用户名或密码错误")

    # ---- 以下走 PG ----
    # 统一文案，不区分「用户不存在」与「密码错误」，避免账号枚举
    if not user["is_active"]:
        raise HTTPException(status_code=401, detail="用户名或密码错误")

    if user_repo.is_locked(user):
        raise HTTPException(status_code=423, detail="账号已临时锁定，请稍后重试")

    if not check_password_hash(user["password_hash"], req.password):
        await user_repo.record_login_failure(user["id"])
        logger.warning(f"登录失败 user={req.username} id={user['id']}")
        raise HTTPException(status_code=401, detail="用户名或密码错误")

    await user_repo.record_login_success(user["id"])
    token = create_token(
        {"sub": user["username"], "uid": user["id"], "role": user["role"]}
    )
    audit_logger.info(
        f"operator={user['id']} ip={request.client.host if request.client else 'unknown'} "
        f"LOGIN ok(来源=pg) user={user['username']}"
    )
    return {
        "token": token,
        "doctor": {
            "name": user.get("name"),
            "department": user.get("department"),
            "hospital": user.get("hospital"),
        },
    }