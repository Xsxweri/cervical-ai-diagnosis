"""端到端验证：报告落库 + Redis 过期后仍可读 + 越权拦截。

需要: PG 已运行、Redis 已运行、迁移脚本已跑。
用法: python -m pytest tests/_smoke_persist.py 或 python tests/_smoke_persist.py
"""

import asyncio
import repos.report_repo as report_repo
from Agents.conversation import ConversationManager
from core.db import acquire, close_pool, init_pool
from core.redis_client import get_redis_client

SID = "__persist_sid__"
OWNER = "doctor-persist-test"
OTHER = "doctor-other"

FAKE_STATE = {
    "risk_level": "high",
    "risk_scope": "case",
    "diagnosis_report": "## 关键发现\n\n测试报告",
    "recommendations": ["3 个月后复查"],
    "references": ["ASCCP 指南 (第 12 页)"],
    "retrieved_knowledge": [],
    "gravcam_path": None,
    "gravcam": None,
    "is_complete": True,
    "clarification_rounds": 0,
    "cv_result": {
        "total_cells": 8,
        "abnormal_cells": 5,
        "highest_risk": "severe_dysplastic",
        "max_uncertainty": 0.35,
        "label_space": "herlev",
        "review_required": True,
        "conflict_count": 1,
        "grading_conflict_count": 1,
        "dual_conflict_count": 0,
        "cells": [{"bbox": [0, 0, 50, 50], "final_class": "severe_dysplastic",
                   "final_confidence": 0.9}],
    },
}


async def main() -> None:
    await init_pool()
    cm = ConversationManager()

    try:
        # 1) 写库
        await report_repo.persist(SID, FAKE_STATE, OWNER)
        print("[1] 已写入 PG")

        # 2) Redis 里放一份（模拟正常流程）
        await cm._save_session(SID, FAKE_STATE)
        await cm._save_owner(SID, OWNER)
        print("[2] 已写入 Redis")

        # 3) 正常读（走 Redis）
        s = await cm.get_session_state(SID, requester_id=OWNER)
        assert s and s["risk_level"] == "high", "Redis 读取失败"
        print("[3] Redis 读取 OK")

        # 4) 抹掉 Redis（模拟 TTL 过期）
        redis = await get_redis_client()
        await redis.delete(f"session:{SID}", f"session_owner:{SID}")
        print("[4] 已删除 Redis key（模拟 TTL 过期）")

        # 5) 再读：应走 PG，且归属校验用 PG 的 operator_id
        s2 = await cm.get_session_state(SID, requester_id=OWNER)
        assert s2 is not None, "Redis 过期后无法从 PG 恢复"
        assert s2["risk_level"] == "high"
        assert s2["cv_result"]["cells"][0]["final_class"] == "severe_dysplastic"
        print("[5] PG 恢复 OK，归属校验通过")

        # 6) 越权：他人应被拒
        try:
            await cm.get_session_state(SID, requester_id=OTHER)
            raise AssertionError("越权未被拦截！")
        except PermissionError as e:
            print(f"[6] 越权拦截 OK: {e}")

        # 7) 回填验证：第 5 步应已把 state 写回 Redis
        redis = await get_redis_client()
        assert await redis.exists(f"session:{SID}"), "未回填 Redis"
        print("[7] Redis 回填 OK")

        print("\n全部通过")
    finally:
        # 清理
        try:
            redis = await get_redis_client()
            await redis.delete(f"session:{SID}", f"session_owner:{SID}")
        except Exception:
            pass
        try:
            async with acquire() as conn:
                await conn.execute(
                    "DELETE FROM diagnosis_reports WHERE session_id = $1", SID
                )
        except Exception:
            pass
        await close_pool()


if __name__ == "__main__":
    asyncio.run(main())