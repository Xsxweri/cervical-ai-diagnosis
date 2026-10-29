"""把 appWeb/doctor_accounts.json 导入PostgreSQL

保留原 Werkzeug scrypt 哈希（格式完全兼容），不重置任何密码
幂等：已存在的 username 跳过 ON CONFLICT DO NOTHING

用法：
    python scripts/migrate_accounts.py            # 预览
    python scripts/migrate_accounts.py --apply    # 执行
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

# 项目已通过 pip install -e . 安装，项目根由 .pth 提供，无需 sys.path 自举
from core.db import acquire, close_pool, init_pool

# 项目根仅用于定位数据文件（appWeb/doctor_accounts.json），与 import 无关
_ROOT = Path(__file__).resolve().parent.parent

SRC = _ROOT / "appWeb" / "doctor_accounts.json"

async def main(apply: bool) -> int:
    if not SRC.is_file():
        print(f"源文件不存在: {SRC}", file=sys.stderr)
        return 1

    accounts = json.loads(SRC.read_text(encoding="utf-8"))
    print(f"源文件共 {len(accounts)} 个账号")

    await init_pool()
    try:
        async with acquire() as conn:
            existing = {
                r["username"]
                for r in await conn.fetch("SELECT username FROM users")
            }

        to_insert = [a for a in accounts if a.get("username") not in existing]
        skipped = len(accounts) - len(to_insert)

        for a in accounts:
            mark = "跳过" if a.get("username") in existing else "导入"
            print(f"  [{mark}] {a.get('username'):20s} {a.get('name') or ''}")

        if not apply:
            print(f"\n预览：将导入 {len(to_insert)} 个，跳过 {skipped} 个")
            print("加 --apply 真正执行")
            return 0

        if not to_insert:
            print("\n无新账号需要导入")
            return 0

        async with acquire() as conn:
            async with conn.transaction():     # 要么全成，要么全不成
                for a in to_insert:
                    await conn.execute(
                        """
                        INSERT INTO users
                            (username, password_hash, name, department, hospital, role)
                        VALUES ($1,$2,$3,$4,$5,$6)
                        ON CONFLICT (username) DO NOTHING
                        """,
                        a["username"],
                        a["password_hash"],
                        a.get("name"),
                        a.get("department"),
                        a.get("hospital"),
                        # 默认 doctor, 系统管理员必须由运维显式授予
                        "doctor",
                    )
        print(f"\n已导入 {len(to_insert)} 个账号，跳过 {skipped} 个")
        return 0
    finally:
        await close_pool()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--apply", action="store_true", help="真正执行（默认预览）")
    sys.exit(asyncio.run(main(p.parse_args().apply)))
