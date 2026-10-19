import os
from dotenv import load_dotenv
from fastapi import Depends, HTTPException, status, Request
from datetime import datetime, timedelta, timezone
from typing import Optional
from jose import JWTError, jwt
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

load_dotenv()

SECRET_KEY =os.getenv("JWT_SECRET_KEY")
if not SECRET_KEY or SECRET_KEY == "please-set-JWT_SECRET_KEY-in-env":
    # 如果没有设置 SECRET_KEY 或者使用了默认弱密钥，则抛出异常
    raise RuntimeError("JWT_SECRET_KEY 未设置或为弱默认值，请在 .env 配置强随机密钥后再启动")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_HOURS = 1

security = HTTPBearer()

def create_token(data:dict, expires_delta:Optional[timedelta] = None)-> str:
    """生成JWT Token"""
    to_encode = data.copy()
    expire = datetime.now(timezone.utc) + (expires_delta or timedelta(hours=ACCESS_TOKEN_EXPIRE_HOURS))
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

def decode_token(token: str) -> dict:
    """校验并解码 JWT（供 WebSocket 等非 Depends 场景使用），失败抛 ValueError"""
    try:
        return jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
    except JWTError:
        raise ValueError("无效的认证凭证")

def verify_token(request: Request, credentials: HTTPAuthorizationCredentials = Depends(security))-> dict:
    """验证JWT Token"""
    try:
        payload = jwt.decode(credentials.credentials, SECRET_KEY, algorithms=[ALGORITHM])
    except JWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="无效的认证凭证",
            headers={"WWW-Authenticate": "Bearer"},
        )
    request.state.requester_id = str(payload.get("uid") or payload.get("sub") or "")
    return payload
