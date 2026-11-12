"""
RAG 检索质量评测。

用法：
    python scripts/eval_retrieval.py                    # 全量, retrieve_nl 路径
    python scripts/eval_retrieval.py --mode plain       # 只测 retrieve(无改写)
    python scripts/eval_retrieval.py --k 5 --json out.json
    python scripts/eval_retrieval.py --baseline outputs/rag_eval_prev.json

"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from datetime import datetime
from pathlib import Path
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_ROOT = Path(__file__).resolve().parent.parent
CASES = str(_ROOT / "tests/rag_eval/queries.yaml")


def _basename(src: str) -> str:
    return str(src or "").replace("\\", "/").split("/")[-1]


def is_relevant(item: dict, case: dict) -> bool:
    """相关判定：命中 expect_terms 之一，且文件匹配（若指定）。"""
    files = case.get("expect_files_any") or []
    must = case.get("expect_all_terms") or []
    terms = case.get("expect_terms") or []

    if files:
        name = _basename(item.get("source", ""))
        if not any(f.lower() in name.lower() for f in files):
            return False

    if not must and not terms:
        return True                      # 无限定 → 有结果即算相关

    body = (item.get("content") or "").lower()
    if must and not all(t.lower() in body for t in must):
        return False
    if terms:
        return any(t.lower() in body for t in terms)
    return True


async def run_case(kb, case, k, mode):
    label = case.get("label")
    if label:
        return await kb.retrieve_by_classification(label, k=k)
    q = case["query"]
    if mode == "nl":
        return await kb.retrieve_nl(q, k=k)
    return await kb.retrieve(q, k=k)


def score_case(case, items, k, pos_mean=None):
    cosines = [float(i.get("cosine") or 0.0) for i in items]
    tags = {
        "known_limitation": bool(case.get("known_limitation")),
        "known_gap": bool(case.get("known_gap")),
    }
    if case.get("type") == "negative":
        worst = max(cosines) if cosines else 0.0
        limit = float(case["max_cos"]) if case.get("max_cos") else (
            pos_mean if pos_mean is not None else 0.6)
        ok = worst < limit
        return {"ok": ok, "cos_max": round(worst, 4), "n": len(items),
                "rr": 0.0, "hit": 0, "limit": round(limit, 4), **tags}

    hits = [i for i, it in enumerate(items, 1) if is_relevant(it, case)]
    rr = 1.0 / hits[0] if hits else 0.0
    ok = bool(hits)
    if case.get("min_cos"):
        ok = ok and (cosines[0] if cosines else 0.0) >= float(case["min_cos"])
    return {"ok": ok, "cos_max": round(cosines[0], 4) if cosines else 0.0,
            "n": len(items), "rr": round(rr, 4), "hit": len(hits), **tags}


def corpus_health(kb) -> list[dict]:
    """每个文件贡献多少 chunk。0 条通常意味着扫描版 PDF 抽不出文本（静默缺失）。"""
    if kb.vectorstore is None:
        return []
    data = kb.vectorstore.get(include=["metadatas"])
    from collections import Counter
    c = Counter(_basename((m or {}).get("source", "")) for m in data.get("metadatas") or [])
    return [{"file": f, "chunks": n} for f, n in c.most_common()]


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--mode", choices=["nl", "plain"], default="nl")
    ap.add_argument("--json", default=None)
    ap.add_argument("--baseline", default=None)
    args = ap.parse_args()

    from services.rag_service import MedicalKnowledgeBase
    from utils.label_queries import CLASSIFICATION_QUERIES

    kb = MedicalKnowledgeBase()
    if kb.vectorstore is None or kb.chunk_count <= 0:
        print("向量库不可用或为空，先运行服务初始化知识库")
        return 1

    print(f"库内 {kb.chunk_count} 条 chunk\n")
    print("—— 语料健康检查（chunk 数 / 文件）——")
    health = corpus_health(kb)
    for row in health:
        flag = "  ← 可疑：0 或极少 chunk" if row["chunks"] < 3 else ""
        print(f"  {row['chunks']:>4}  {row['file']}{flag}")
    print()

    with open(CASES, encoding="utf-8") as f:
        spec = yaml.safe_load(f)

    cases = spec["cases"]

    pos_cases = [c for c in cases if c.get("type") != "negative"]
    cache: dict[str, list] = {}
    pos_cos: list[float] = []
    skipped: list[str] = []
    for case in pos_cases:
        items = await run_case(kb, case, args.k, args.mode)
        cache[case["id"]] = items
        if case.get("known_gap"):
            continue
        if items and is_relevant(items[0], case):
            pos_cos.append(float(items[0].get("cosine") or 0.0))
        else:
            skipped.append(case["id"])
    pos_mean = sum(pos_cos) / len(pos_cos) if pos_cos else 0.0
    print(f"正样本 top-1 余弦均值 = {pos_mean:.4f} (n={len(pos_cos)}，负样本上限)")
    if skipped:
        print(f"  未计入基准（top-1 不相关）：{skipped}")
    print()

    results = []
    for case in cases:
        items = cache.get(case["id"])
        if items is None:
            items = await run_case(kb, case, args.k, args.mode)
        r = score_case(case, items, args.k, pos_mean=pos_mean)
        r.update({"id": case["id"], "type": case["type"], "query": case["query"]})
        results.append(r)
        mark = "OK  " if r["ok"] else "FAIL"
        extra = f" (限 {r['limit']:.3f})" if r["type"] == "negative" else ""
        if r.get("known_gap"):
            extra += " [gap]"
        print(f"  {mark} {case['id']:<5} cos={r['cos_max']:.3f} n={r['n']} "
              f"rr={r['rr']:.2f}{extra}  {case['query'][:40]}")

    pos = [r for r in results if r["type"] != "negative" and not r.get("known_gap")]
    neg = [r for r in results if r["type"] == "negative" and not r.get("known_limitation")]
    gaps = [r for r in results if r.get("known_gap")]
    summary = {
        "chunks": kb.chunk_count,
        "cases": len(results),
        "pos_mean_cos": round(pos_mean, 4),
        "scored_cases": len(pos) + len(neg),
        "pass_rate": round(
            sum(r["ok"] for r in pos + neg) / (len(pos) + len(neg)), 4
        ) if (pos or neg) else 0.0,
        "recall": round(sum(r["ok"] for r in pos) / len(pos), 4) if pos else 0.0,
        "mrr": round(sum(r["rr"] for r in pos) / len(pos), 4) if pos else 0.0,
        "mean_cos": round(sum(r["cos_max"] for r in pos) / len(pos), 4) if pos else 0.0,
        "neg_pass": round(sum(r["ok"] for r in neg) / len(neg), 4) if neg else 1.0,
        "neg_known_limitation": sum(
            1 for r in results
            if r["type"] == "negative" and r.get("known_limitation")),
        "corpus_gaps": len(gaps),
        "corpus_gap_ids": [r["id"] for r in gaps],
        "corpus_gap_hit_rate": round(
            sum(r["ok"] for r in gaps) / len(gaps), 4) if gaps else 0.0,
        "empty_rate": round(sum(1 for r in results if r["n"] == 0) / len(results), 4),
    }
    print("\n—— 汇总 ——")
    for kk, vv in summary.items():
        print(f"  {kk:<12} {vv}")

    if args.baseline and os.path.exists(args.baseline):
        with open(args.baseline, encoding="utf-8") as f:
            base = json.load(f).get("summary", {})
        print("\n—— 对比基线 ——")
        for kk in ("recall", "mrr", "mean_cos", "neg_pass"):
            if kk in base:
                delta = summary[kk] - base[kk]
                print(f"  {kk:<12} {base[kk]} → {summary[kk]}  ({delta:+.4f})")

    if args.json:
        os.makedirs(os.path.dirname(args.json) or ".", exist_ok=True)
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump({"ts": datetime.now().isoformat(), "summary": summary,
                       "health": health, "results": results},
                      f, ensure_ascii=False, indent=2)
        print(f"\n已写入 {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))