"""task_store 状态机测试(fake Redis)"""

import pytest
from core import task_store


class _FakeRedis:
    def __init__(self):
        self.store = {}
    async def set(self, k, v, ex=None):
        self.store[k] = v
    async def get(self, k):
        return self.store.get(k)


@pytest.fixture
def fake(monkeypatch):
    f = _FakeRedis()
    async def _client():
        return f
    monkeypatch.setattr(task_store, "get_redis_client", _client)
    return f


async def test_roundtrip(fake):
    await task_store.set_status("s1", task_store.QUEUED, owner_id="1")
    data = await task_store.get_status("s1")
    assert data["status"] == "queued"
    assert data["owner_id"] == "1"


async def test_done_keeps_owner(fake):
    """回归：jobs.py 每个状态都必须带 owner_id，否则归属校验失效。"""
    await task_store.set_status("s2", task_store.QUEUED, owner_id="7")
    await task_store.set_status("s2", task_store.DONE, owner_id="7")
    assert (await task_store.get_status("s2"))["owner_id"] == "7"


async def test_failed_carries_error(fake):
    await task_store.set_status("s3", task_store.FAILED, error="boom")
    assert (await task_store.get_status("s3"))["error"] == "boom"


async def test_missing_returns_none(fake):
    assert await task_store.get_status("nope") is None


async def test_corrupt_payload_returns_none(fake):
    fake.store["task:bad"] = "{not json"
    assert await task_store.get_status("bad") is None


async def test_is_terminal(fake):
    await task_store.set_status("s4", task_store.RUNNING)
    assert not await task_store.is_terminal("s4")
    await task_store.set_status("s4", task_store.DONE)
    assert await task_store.is_terminal("s4")