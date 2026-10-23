"""认证路由：医生登录。"""

import json

from fastapi import APIRouter, HTTPException, Request
from werkzeug.security import check_password_hash

from api.config import ACCOUNTS_PATH
from api.runtime import limiter
from api.schemas import LoginRequest
from core.auth import create_token

router = APIRouter(tags=["auth"])


@router.post("/api/auth/login", summary="医生登录")
@limiter.limit("5/minute")
async def login(request: Request, req: LoginRequest):
    try:
        with open(ACCOUNTS_PATH, "r", encoding="utf-8") as f:
            accounts = json.load(f)
    except (OSError, json.JSONDecodeError):
        raise HTTPException(status_code=500, detail="账号数据读取失败")

    doctor = next((d for d in accounts if d.get("username") == req.username), None)
    if not doctor or not check_password_hash(doctor.get("password_hash", ""), req.password):
        raise HTTPException(status_code=401, detail="用户名或密码错误")

    token = create_token({"sub": doctor["username"], "uid": doctor.get("id")})
    return {
        "token": token,
        "doctor": {
            "name": doctor.get("name"),
            "department": doctor.get("department"),
            "hospital": doctor.get("hospital"),
        },
    }