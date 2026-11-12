"""
知识库状态判定、跨进程锁、manifest 原子读写。
"""

from __future__ import annotations

import contextlib
import json
import os
import sys
import time


def rebuild_reason(manifest: dict | None, fingerprint: str, actual: int) -> str | None:
    """判断是否需要重建索引。返回原因字符串: None 表示可跳过。

    actual: collection 实际条目数；-1 表示不可读（如向量库尚未加载）。

    判定顺序（每条对应一个真实缺陷）：
      1. manifest 缺失但库里有数据 → 状态不可信（历史重复灌入 / 中途崩溃）→ 重建
      2. manifest 缺失且库为空 → 首次索引
      3. 指纹变化 → 语料或嵌入模型变了
      4. manifest.chunks ≠ 实际条数 → 上次写入不完整 → 重建
    """
    if manifest is None:
        if actual > 0:
            return f"manifest 缺失但库内已有 {actual} 条，状态不可信"
        return "首次索引（无 manifest 且库为空）"
    if manifest.get("fingerprint") != fingerprint:
        return "语料指纹变化"
    expected = int(manifest.get("chunks") or 0)
    if actual >= 0 and actual != expected:
        return f"库内 {actual} 条 ≠ manifest {expected} 条"
    return None


def atomic_write_json(path: str, payload: dict) -> None:
    """tmp + fsync + os.replace。避免进程在写 manifest 时被杀导致文件半截。"""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = f"{path}.tmp.{os.getpid()}"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def _lock_fh(fh) -> None:
    if os.name == "nt":                       # Windows
        import msvcrt
        fh.seek(0)
        msvcrt.locking(fh.fileno(), msvcrt.LK_NBLCK, 1)
    else:                                     # POSIX
        import fcntl
        fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)


def _unlock_fh(fh) -> None:
    if os.name == "nt":
        import msvcrt
        fh.seek(0)
        msvcrt.locking(fh.fileno(), msvcrt.LK_UNLCK, 1)
    else:
        import fcntl
        fcntl.flock(fh.fileno(), fcntl.LOCK_UN)


@contextlib.contextmanager
def index_lock(path: str, timeout: float = 300.0, poll: float = 0.5):
    """跨进程互斥锁（api 进程 与 arq worker 会各自 init_kb）。

    yield 出是否真正拿到锁。超时后**放弃等待并继续**：
    宁可重复索引一次，也不要让服务启动卡死 —— 与 RAG「永不阻断启动」的约定一致。
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fh = open(path, "a+b")
    if os.fstat(fh.fileno()).st_size == 0:
        fh.write(b"\0")
        fh.flush()
    acquired = False
    deadline = time.monotonic() + timeout
    try:
        while True:
            try:
                _lock_fh(fh)
                acquired = True
                break
            except OSError:
                if time.monotonic() >= deadline:
                    print(f"[RAG] 等待索引锁超时（{timeout}s），继续执行", file=sys.stderr)
                    break
                time.sleep(poll)
        yield acquired
    finally:
        if acquired:
            try:
                _unlock_fh(fh)
            except OSError:
                pass
        fh.close()