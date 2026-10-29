"""
报告归档：先导出，再删除。默认只做 dry-run。

用法：
    python docker/scripts/pg_archive.py                # 预览将归档多少条
    python docker/scripts/pg_archive.py --apply        # 真正执行
"""
from __future__ import annotations

import argparse
import asyncio
import os
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import core.env  # noqa: F401  加载 .env，否则 DATABASE_URL 读不到

import asyncpg

# 项目根仅用于定位 outputs/archive，与 import 无关
_ROOT = Path(__file__).resolve().parent.parent.parent

# 与 docker-compose.yml 里的 container_name / POSTGRES_* 对齐
PG_CONTAINER = os.getenv("PG_CONTAINER", "cervix-pg")
PG_USER = os.getenv("POSTGRES_USER", "cervix")
PG_DB = os.getenv("POSTGRES_DB", "cervix")

DATABASE_URL = os.getenv("DATABASE_URL", "")
MONTHS = int(os.getenv("REPORT_RETENTION_MONTHS", "24"))
OUT = _ROOT / "outputs" / "archive"


def _full_snapshot(dest: Path) -> None:
    """容器内 pg_dump 整库快照。失败即抛，不留半截文件。"""
    dest.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "docker", "exec", PG_CONTAINER,
        "pg_dump", "-U", PG_USER, "-d", PG_DB, "-Fc",
    ]
    with dest.open("wb") as f:
         proc = subprocess.run(cmd, stdout=f, stderr=subprocess.PIPE, check=False)
    if proc.returncode != 0:
        dest.unlink(missing_ok=True)
        raise RuntimeError(
            f"pg_dump 失败（容器 {PG_CONTAINER}）: "
            f"{proc.stderr.decode(errors='replace').strip()}"
        )


async def main(apply: bool) -> int:
    if not DATABASE_URL:
        print("DATABASE_URL 未设置，请检查 .env", file=sys.stderr)
        return 1

    cutoff = datetime.now(timezone.utc) - timedelta(days=30 * MONTHS)
    conn = await asyncpg.connect(DATABASE_URL)
    try:
        pending = await conn.fetchval(
            "SELECT count(*) FROM diagnosis_reports WHERE created_at < $1", cutoff
        )
        total = await conn.fetchval("SELECT count(*) FROM diagnosis_reports")
        print(f"库内报告总数       : {total}")
        print(f"早于 {cutoff:%Y-%m-%d}  : {pending} 条（保留期 {MONTHS} 个月）")

        if pending == 0:
            print("无需归档")
            return 0
        if not apply:
            print("\n[dry-run] 未做任何修改。加 --apply 执行（会先做整库快照）")
            return 0

        # 1) 快照
        snap = OUT / f"snapshot_before_archive_{datetime.now():%Y%m%dT%H%M%S}.dump"
        print(f"\n正在做整库快照 → {snap}")
        _full_snapshot(snap)
        size_mb = snap.stat().st_size / (1024 * 1024)
        print(f"快照完成: {size_mb:.1f} MB")
        if size_mb < 0.01:
            raise RuntimeError(f"快照文件过小（{size_mb:.3f} MB），中止删除")

        # 2) 删除（在事务内，且校验影响行数）
        async with conn.transaction():
            status = await conn.execute(
                "DELETE FROM diagnosis_reports WHERE created_at < $1", cutoff
            )
        deleted = int(status.split()[-1]) if status else 0
        print(f"已删除 {deleted} 条")

        if deleted != pending:
            print(
                f"⚠️ 删除数({deleted}) 与预检数({pending}) 不一致 —— "
                f"期间可能有新数据或条件漂移，请人工核查。快照保留在 {snap}",
                file=sys.stderr,
            )
            return 2

        print(f"归档完成。如需回滚：docker exec -i {PG_CONTAINER} "
              f"pg_restore -U {PG_USER} -d {PG_DB} < {snap.name}")
        return 0
    finally:
        await conn.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--apply", action="store_true", help="真正执行（默认预览）")
    sys.exit(asyncio.run(main(p.parse_args().apply)))
