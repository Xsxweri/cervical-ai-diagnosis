"""账号管理路由测试。

不依赖 PostgreSQL：repo 层用 monkeypatch 替换，测的是路由逻辑本身
（权限门禁、防自锁、最后管理员保护、响应不泄露哈希）。
"""

import pytest
from fastapi.testclient import TestClient
from werkzeug.security import generate_password_hash

import repos.user_repo as user_repo
from agent_api import app
from core.auth import create_token

# 不用 with → 不跑 lifespan → 无需 PG / Redis / GPU 模型
client = TestClient(app)


# ---------------- 假数据 ----------------

ADMIN = {"id": 1, "username": "admin", "name": "管理员", "department": "病理科",
         "hospital": "测试医院", "role": "sysadmin", "is_active": True,
         "created_at": None, "last_login_at": None,
         "failed_attempts": 0, "locked_until": None}

DOCTOR = {"id": 2, "username": "doc", "name": "医生", "department": "妇科",
          "hospital": "测试医院", "role": "doctor", "is_active": True,
          "created_at": None, "last_login_at": None,
          "failed_attempts": 0, "locked_until": None}

@pytest.fixture
def as_admin(monkeypatch):
    """把当前用户固定为 ADMIN。"""
    async def _get_by_id(uid):
        return ADMIN if int(uid) == 1 else None
    monkeypatch.setattr(user_repo, "get_by_id", _get_by_id)
    return {"Authorization": f"Bearer {create_token({'sub': 'admin', 'uid': 1, 'role': 'admin'})}"}


@pytest.fixture
def as_doctor(monkeypatch):
    async def _get_by_id(uid):
        return DOCTOR if int(uid) == 2 else None
    monkeypatch.setattr(user_repo, "get_by_id", _get_by_id)
    return {"Authorization": f"Bearer {create_token({'sub': 'doc', 'uid': 2, 'role': 'doctor'})}"}


# ---------------- 鉴权 ----------------

def test_no_token_rejected():
    assert client.get("/api/accounts").status_code in (401, 403)


def test_doctor_forbidden(as_doctor):
    assert client.get("/api/accounts", headers=as_doctor).status_code == 403


def test_inactive_account_forbidden(monkeypatch):
    """JWT 有效但账号已被停用 → 必须 403（证明门禁回查 DB 而非信 claim）。"""
    dead = {**ADMIN, "is_active": False}
    async def _get_by_id(uid):
        return dead
    monkeypatch.setattr(user_repo, "get_by_id", _get_by_id)
    h = {"Authorization": f"Bearer {create_token({'sub': 'admin', 'uid': 1, 'role': 'admin'})}"}
    assert client.get("/api/accounts", headers=h).status_code == 403


# ---------------- 列表 ----------------

def test_list_never_exposes_password_hash(monkeypatch, as_admin):
    rows = [{**ADMIN, "password_hash": "scrypt:LEAKED"}, {**DOCTOR, "password_hash": "scrypt:LEAKED"}]
    async def _list():
        return rows
    monkeypatch.setattr(user_repo, "list_users_for_admin", _list)

    r = client.get("/api/accounts", headers=as_admin)
    assert r.status_code == 200
    assert len(r.json()) == 2
    assert "password_hash" not in r.text
    assert "LEAKED" not in r.text


# ---------------- 防自锁 / 最后管理员 ----------------

def test_cannot_deactivate_self(as_admin):
    r = client.patch("/api/accounts/1", headers=as_admin, json={"is_active": False})
    assert r.status_code == 400
    assert "自己" in r.json()["detail"]


def test_cannot_demote_self(as_admin):
    r = client.patch("/api/accounts/1", headers=as_admin, json={"role": "doctor"})
    assert r.status_code == 400


def test_cannot_remove_last_sysadmin(monkeypatch, as_admin):
    other = {**ADMIN, "id": 9, "username": "admin2"}
    async def _get_by_id(uid):
        return ADMIN if int(uid) == 1 else other if int(uid) == 9 else None
    async def _count():
        return 1
    monkeypatch.setattr(user_repo, "get_by_id", _get_by_id)
    monkeypatch.setattr(user_repo, "count_active_sysadmins", _count)

    r = client.patch("/api/accounts/9", headers=as_admin, json={"is_active": False})
    assert r.status_code == 409

def test_business_admin_cannot_manage_accounts(monkeypatch):
    """role='admin' 是业务角色，不得进入账号管理"""
    biz = {**ADMIN, "role": "admin"}
    async def _by_id(uid):
        return biz
    monkeypatch.setattr(user_repo, "get_by_id", _by_id)
    h = {"Authorization": f"Bearer {create_token({'sub': 'a', 'uid': 1, 'role': 'admin'})}"}

    assert client.get("/api/accounts", headers=h).status_code == 403

def test_last_admin_transfer_allowed(monkeypatch, as_admin):
    """ 最后一个管理员被停用 → 允许被停用 """
    other = {**ADMIN, "id": 9}
    async def _get_by_id(uid):
        return ADMIN if int(uid) == 1 else other
    async def _count():
        return 2
    async def _ok(*a, **kw):
        return True
    monkeypatch.setattr(user_repo, "get_by_id", _get_by_id)
    monkeypatch.setattr(user_repo, "count_active_sysadmins", _count)
    monkeypatch.setattr(user_repo, "update_user", _ok)

    r = client.patch("/api/accounts/9", headers=as_admin, json={"is_active": False})
    assert r.status_code == 200


def test_create_duplicate_username_409(monkeypatch, as_admin):
    from asyncpg.exceptions import UniqueViolationError
    async def _create(**kw):
        raise UniqueViolationError("duplicate key")
    monkeypatch.setattr(user_repo, "create_user", _create)

    r = client.post("/api/accounts", headers=as_admin,
                    json={"username": "admin", "password": "Abcdefg123"})
    assert r.status_code == 409


def test_create_rejects_short_password(as_admin):
    r = client.post("/api/accounts", headers=as_admin,
                    json={"username": "newdoc", "password": "123"})
    assert r.status_code == 422


def test_create_rejects_bad_role(as_admin):
    r = client.post("/api/accounts", headers=as_admin,
                    json={"username": "newdoc", "password": "Abcdefg123", "role": "root"})
    assert r.status_code == 422


# ---------------- 自助改密 ----------------

def test_change_own_password_wrong_old(monkeypatch):
    async def _by_id(uid):
        return DOCTOR
    async def _hash(uid):
        return generate_password_hash("correct-old", method="scrypt")
    monkeypatch.setattr(user_repo, "get_by_id", _by_id)
    monkeypatch.setattr(user_repo, "get_password_hash", _hash)
    h = {"Authorization": f"Bearer {create_token({'sub': 'doc', 'uid': 2, 'role': 'doctor'})}"}

    r = client.post("/api/accounts/me/password", headers=h,
                    json={"old_password": "wrong", "new_password": "Abcdefg123"})
    assert r.status_code == 401


def test_login_db_down_returns_503(monkeypatch):
    """P0 核心行为：DB 故障必须 503，不能被误报为密码错误。"""
    async def _boom(username):
        raise RuntimeError("db down")
    monkeypatch.setattr(user_repo, "get_by_username", _boom)

    r = client.post("/api/auth/login", json={"username": "dr_zhang", "password": "x"})
    assert r.status_code == 503
    assert "用户名或密码错误" not in r.text

def test_me_never_exposes_password_hash(monkeypatch):
    async def _by_id(uid):
        return {**DOCTOR, "password_hash": "scrypt:LEAKED"}
    monkeypatch.setattr(user_repo, "get_by_id", _by_id)
    h = {"Authorization": f"Bearer {create_token({'sub': 'doc', 'uid': 2, 'role': 'doctor'})}"}

    r = client.get("/api/accounts/me", headers=h)
    assert r.status_code == 200
    assert r.json()["username"] == "doc"
    assert "LEAKED" not in r.text


def test_change_own_password_ok(monkeypatch):
    seen = {}
    async def _by_id(uid): return DOCTOR
    async def _hash(uid): return generate_password_hash("correct-old", method="scrypt")
    async def _set(uid, h):
        seen["uid"] = uid
        return True
    monkeypatch.setattr(user_repo, "get_by_id", _by_id)
    monkeypatch.setattr(user_repo, "get_password_hash", _hash)
    monkeypatch.setattr(user_repo, "set_password", _set)
    h = {"Authorization": f"Bearer {create_token({'sub': 'doc', 'uid': 2, 'role': 'doctor'})}"}

    r = client.post("/api/accounts/me/password", headers=h,
                    json={"old_password": "correct-old", "new_password": "Abcdefg123"})
    assert r.status_code == 204
    assert seen["uid"] == 2          # 证明改的是自己，不是别人


def test_change_own_password_rejects_same(monkeypatch):
    async def _by_id(uid): return DOCTOR
    async def _hash(uid): return generate_password_hash("same-pass", method="scrypt")
    monkeypatch.setattr(user_repo, "get_by_id", _by_id)
    monkeypatch.setattr(user_repo, "get_password_hash", _hash)
    h = {"Authorization": f"Bearer {create_token({'sub': 'doc', 'uid': 2, 'role': 'doctor'})}"}

    r = client.post("/api/accounts/me/password", headers=h,
                    json={"old_password": "same-pass", "new_password": "same-pass"})
    assert r.status_code == 400


# ---------------- 管理：成功路径 / 404 / 空 body ----------------

def _with_doctor_target(monkeypatch):
    """as_admin 的 get_by_id 对 uid!=1 返回 None，成功路径需要目标存在。"""
    async def _by_id(uid):
        return ADMIN if int(uid) == 1 else {**DOCTOR, "id": 2}
    monkeypatch.setattr(user_repo, "get_by_id", _by_id)


def test_reset_password_ok(monkeypatch, as_admin):
    _with_doctor_target(monkeypatch)
    async def _set(uid, h): return True
    monkeypatch.setattr(user_repo, "set_password", _set)

    r = client.post("/api/accounts/2/password", headers=as_admin,
                    json={"password": "Abcdefg123"})
    assert r.status_code == 204


def test_unlock_ok(monkeypatch, as_admin):
    _with_doctor_target(monkeypatch)
    async def _unlock(uid): return True
    monkeypatch.setattr(user_repo, "unlock_user", _unlock)

    assert client.post("/api/accounts/2/unlock", headers=as_admin).status_code == 204

def test_patch_empty_body_400(as_admin):   # 管理：空 body 400
    assert client.patch("/api/accounts/2", headers=as_admin, json={}).status_code == 400


def test_patch_unknown_user_404(as_admin):
    # as_admin 的 get_by_id 对 uid!=1 返回 None
    r = client.patch("/api/accounts/999", headers=as_admin, json={"name": "x"})
    assert r.status_code == 404