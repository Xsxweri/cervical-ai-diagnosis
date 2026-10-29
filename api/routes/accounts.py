"""
账号管理路由
"""

from asyncpg.exceptions import UniqueViolationError
from fastapi import APIRouter, Depends, HTTPException, Request
from werkzeug.security import check_password_hash, generate_password_hash

import repos.user_repo as user_repo
from api.logging_setup import audit_logger
from api.runtime import limiter
from api.schemas import (
    PasswordResetRequest,
    SelfPasswordChangeRequest,
    UserCreate,
    UserPublic,
    UserUpdate,
)
from api.security import require_sysadmin
from core.auth import verify_token

router = APIRouter(prefix="/api/accounts", tags=["accounts"])


def _hash(pw: str) -> str:
    # scrypt 与历史账号密码格式一致
    return generate_password_hash(pw, method="scrypt")


def _audit(request: Request, operator_id: int, action: str, target: str) -> None:
    ip = request.client.host if request.client else "unknown"
    audit_logger.info(f"operator={operator_id} ip={ip} ACCOUNT {action} {target}")


# ============ 自助（任意已登录账号） ============

@router.get("/me", response_model=UserPublic, summary="当前账号信息")
async def get_me(payload: dict = Depends(verify_token)):
    uid = payload.get("uid")
    if uid is None:
        raise HTTPException(status_code=403, detail="缺少操作者标识")
    user = await user_repo.get_by_id(int(uid))
    if not user:
        raise HTTPException(status_code=404, detail="账号不存在")
    return user


@router.post("/me/password", status_code=204, summary="修改自己的密码")
@limiter.limit("10/hour")
async def change_own_password(
    request: Request,
    req: SelfPasswordChangeRequest,
    payload: dict = Depends(verify_token),
):
    uid = payload.get("uid")
    if uid is None:
        raise HTTPException(status_code=403, detail="缺少操作者标识")
    uid = int(uid)

    current = await user_repo.get_password_hash(uid)
    if not current:
        raise HTTPException(status_code=404, detail="账号不存在")

    # 必须验旧密码：即便 token 被盗，也无法直接改密码把本人锁死
    if not check_password_hash(current, req.old_password):
        raise HTTPException(status_code=401, detail="原密码错误")
    if req.old_password == req.new_password:
        raise HTTPException(status_code=400, detail="新密码不能与原密码相同")

    await user_repo.set_password(uid, _hash(req.new_password))
    _audit(request, uid, "change_own_password", f"id={uid}")
    return None


# ============ 管理（仅 admin） ============

@router.get("", response_model=list[UserPublic], summary="账号列表")
async def list_accounts(operator: dict = Depends(require_sysadmin)):
    return await user_repo.list_users_for_admin()


@router.post("", response_model=UserPublic, status_code=201, summary="新建账号")
@limiter.limit("20/hour")
async def create_account(
    request: Request,
    req: UserCreate,
    operator: dict = Depends(require_sysadmin),
):
    try:
        new_id = await user_repo.create_user(
            username=req.username,
            password_hash=_hash(req.password),
            name=req.name,
            department=req.department,
            hospital=req.hospital,
            role=req.role,
        )
    except UniqueViolationError:
        raise HTTPException(status_code=409, detail=f"用户名已存在: {req.username}")

    _audit(request, operator["id"], "create", f"id={new_id} user={req.username} role={req.role}")
    user = await user_repo.get_by_id(new_id)
    if not user:
        raise HTTPException(status_code=500, detail="账号已创建但读取失败")
    return user


@router.patch("/{user_id}", response_model=UserPublic, summary="修改账号")
async def update_account(
    request: Request,
    user_id: int,
    req: UserUpdate,
    operator: dict = Depends(require_sysadmin),
):
    # exclude_none：PATCH 里显式传 null 不应把 role/is_active 写成 NULL
    changes = req.model_dump(exclude_unset=True, exclude_none=True)
    if not changes:
        raise HTTPException(status_code=400, detail="没有需要修改的字段")

    target = await user_repo.get_by_id(user_id)
    if not target:
        raise HTTPException(status_code=404, detail="账号不存在")

    # --- 防自锁 ---
    if user_id == operator["id"] and changes.get("is_active") is False:
        raise HTTPException(status_code=400, detail="不能停用自己的账号")
    if user_id == operator["id"] and changes.get("role", "sysadmin") != "sysadmin":
        raise HTTPException(status_code=400, detail="不能撤销自己的管理员权限")

    # --- 防「最后一个管理员」消失 ---
    next_role = changes.get("role", target["role"])
    next_active = changes.get("is_active", target["is_active"])
    losing_sysadmin = (
        target["role"] == "sysadmin"
        and target["is_active"]
        and (next_role != "sysadmin" or next_active is False)
    )
    if losing_sysadmin and await user_repo.count_active_sysadmins() <= 1:
        raise HTTPException(status_code=409, detail="系统至少需要保留一个启用的管理员")

    if not await user_repo.update_user(user_id, **changes):
        raise HTTPException(status_code=400, detail="更新未生效")

    _audit(request, operator["id"], "update", f"id={user_id} fields={sorted(changes)}")
    return await user_repo.get_by_id(user_id)


@router.post("/{user_id}/password", status_code=204, summary="重置密码（管理员代办）")
@limiter.limit("20/hour")
async def reset_password(
    request: Request,
    user_id: int,
    req: PasswordResetRequest,
    operator: dict = Depends(require_sysadmin),
):
    if not await user_repo.get_by_id(user_id):
        raise HTTPException(status_code=404, detail="账号不存在")
    await user_repo.set_password(user_id, _hash(req.password))
    _audit(request, operator["id"], "reset_password", f"id={user_id}")
    return None


@router.post("/{user_id}/unlock", status_code=204, summary="解除锁定（管理员代办）")
async def unlock_account(
    request: Request,
    user_id: int,
    operator: dict = Depends(require_sysadmin),
):
    if not await user_repo.get_by_id(user_id):
        raise HTTPException(status_code=404, detail="账号不存在")
    await user_repo.unlock_user(user_id)
    _audit(request, operator["id"], "unlock", f"id={user_id}")
    return None