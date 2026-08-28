import os
from fastapi import Depends,HTTPException,status
from datetime import datetime, timedelta
from typing import Optional
from jose import JWTError,jwt
from fastapi.security import HTTPBearer,HTTPAuthorizationCredentials

SECRET_KEY =os.getenv("JWT_SECRET_KEY", "please-set-JWT_SECRET_KEY-in-env")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_HOURS = 1

security = HTTPBearer()

def create_token(data:dict, expires_delta:Optional[timedelta] = None)-> str:
    """生成JWT Token"""
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(hours=ACCESS_TOKEN_EXPIRE_HOURS))
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security))-> dict:
    """验证JWT Token"""
    try:
        payload = jwt.decode(credentials.credentials, SECRET_KEY, algorithms=[ALGORITHM])
        return payload
    except JWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="无效的认证凭证",
            headers={"WWW-Authenticate": "Bearer"},
        )
