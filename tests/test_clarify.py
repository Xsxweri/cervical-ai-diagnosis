import os
import sys
import pytest
import httpx, asyncio

BASE = os.getenv("TEST_BASE_URL", "http://localhost:8000")
IMG = os.getenv("TEST_IMAGE", r"data\sipakmed\im_Koilocytotic\im_Koilocytotic\001.bmp")
USER = os.getenv("TEST_DOCTOR_USER", "")
PWD = os.getenv("TEST_DOCTOR_PWD", "")

if not USER or not PWD:
    pytest.skip(
        "未设置 TEST_DOCTOR_USER / TEST_DOCTOR_PWD，跳过澄清链路集成测试",
        allow_module_level=True,
    )

async def main():
    async with httpx.AsyncClient(timeout=180.0) as c:
        lr = await c.post(f"{BASE}/api/auth/login", json={"username": USER, "password": PWD})
        print("[login]  HTTP", lr.status_code)
        if lr.status_code != 200:
            print("[login]  body", lr.text[:300]); return
        h = {"Authorization": f"Bearer {lr.json()['token']}"}

        with open(IMG, "rb") as f:
            ur = await c.post(f"{BASE}/api/agent/upload", headers=h,
                              files={"file": ("cell.bmp", f, "image/bmp")})
        print("[upload] HTTP", ur.status_code)
        print("[upload] body", ur.text[:600])
        if ur.status_code != 200:
            return
        up = ur.json()
        print("[upload] status   =", up["status"])
        print("[upload] questions=", up.get("questions"))
        assert up["status"] == "awaiting_clarification", "未触发澄清：确认 UNCERTAINTY_HIGH=-1.0 且已重启"
        sid = up["session_id"]

        cr = await c.post(f"{BASE}/api/agent/clarify/{sid}", headers=h,
                          json={"user_input": "患者35岁，HPV16阳性，去年TCT为ASC-US，本次复查。"})
        print("[clarify] HTTP", cr.status_code)
        print("[clarify] body", cr.text[:600])
        if cr.status_code != 200:
            return
        cl = cr.json()
        print("[clarify] status =", cl["status"])
        print("[clarify] report =", (cl.get("report") or "")[:80], "...")
        assert cl["status"] == "complete", "resume 后未完成"
        print("\n✅ 澄清 / 中断 / Redis 恢复 全路径通过")

if __name__ == "__main__":
    asyncio.run(main())