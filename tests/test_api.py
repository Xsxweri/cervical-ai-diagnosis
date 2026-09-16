from fastapi.testclient import TestClient
from agent_api import app

# 不用 with 上下文 → 不运行 lifespan → 无需 GPU 模型 / Redis Stack
client = TestClient(app)


def test_health_ok():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_login_wrong_password_401():
    r = client.post("/api/auth/login", json={"username": "dr_zhang", "password": "wrong-pass"})
    assert r.status_code == 401


def test_login_unknown_user_401():
    r = client.post("/api/auth/login", json={"username": "no_such_user", "password": "x"})
    assert r.status_code == 401


def test_analyze_requires_auth():
    assert client.post("/analyze").status_code in (401, 403)


def test_agent_upload_requires_auth():
    assert client.post("/api/agent/upload").status_code in (401, 403)