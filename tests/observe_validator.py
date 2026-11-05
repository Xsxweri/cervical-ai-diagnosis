"""批量观察 validator 的 blocker 率。

用途：判断校验规则是否过严 —— 而不是等真实门诊病例。
用法：
    python tests/observe_validator.py --limit 30 --space sipakmed

输出：
    outputs/results/validator_observation.json   逐例明细
    控制台   按细胞数分桶的 blocker 率
"""

from __future__ import annotations

import argparse
import asyncio
import glob
import json
import os
import random
from collections import Counter

# 项目已通过 pip install -e . 安装，无需 sys.path 自举
import utils.uncertainty as U
from Agents.llm import get_llm                      # noqa: F401 预热配置
from Agents.nodes.report import generate_report_agent
from Agents.nodes.uncertainty import uncertainty_assessment_agent
from services.vision_service import analyze_image_from_path
from Agents.nodes.retrieval import (
    deep_retrieval_agent,
    post_retrieval_assessment_agent,
)
from utils.trace import Trace


_RISK_RANK = {"low": 0, "medium": 1, "high": 2, "emergency": 3}


def _rank(level) -> int:
    """风险等级 → 序列。"""
    return _RISK_RANK.get(str(level or "").lower(), -1)

async def run_one(path: str, label_space: str) -> dict:
    """单例：CV → 构造最小 state → 生成报告 → 校验。"""
    tr = Trace()
    with tr.node("cv_analysis"):
        cv = await analyze_image_from_path(path)
    

    state = {
        "image_path": path,
        "cv_result": cv,
        "max_uncertainty": cv.get("max_uncertainty", 0.0),
        "model_uncertainty": cv.get("max_uncertainty", 0.0),   # 无检索路径，两者相同
        "evidence_sufficiency": None,                            # 暂无
        "risk_level": cv.get("risk_level"),
        "risk_scope": cv.get("risk_scope"),
        "clarification_rounds": 0,
        "retrieval_done": False,
        "conversation_history": [],
        "_trace": tr,
    }

    decision = uncertainty_assessment_agent(state)
    state.update(decision)

    if state.get("action") == "deep_retrieve":
        state.update(await deep_retrieval_agent(state))
        state.update(await post_retrieval_assessment_agent(state))

    base = {
        "file": os.path.relpath(path, f"data/{label_space}"),   # 用相对路径，避免同名文件混淆
        "total_cells": cv.get("total_cells", 0),
        "risk_level": state.get("risk_level"),
        "risk_scope": state.get("risk_scope"),
        "conflict_count": cv.get("conflict_count", 0),
        "action": state.get("action"),
        "cv_risk_level": cv.get("risk_level"),
        "max_uncertainty": round(float(cv.get("max_uncertainty") or 0.0), 4),
        "abnormal_cells": cv.get("abnormal_cells", 0),
    }

    if state.get("action") != "direct_report":
        return {
            **base,
            "skipped": True,
            "fallback_triggered": False,
            "fallback_reason": "none",
            "n_blocker": 0,
            "n_warning": 0,
            "kinds": [],
            "blocker_kinds": [],
            "warning_kinds": [],
            "blocker_details": [],
            "report_snippet": "",
            "final_risk": state.get("risk_level"),
            "declared_risk": None,
            "diverged": False,
            "trace": tr.as_dict(),
        }

    out = await generate_report_agent(state)

    issues = out.get("validator_issues") or []
    n_blocker = sum(1 for i in issues if i["severity"] == "blocker")
    fallback = out.get("diagnosis_report", "").startswith("AI 辅助报告生成失败")
    return {
        **base,
        "skipped": False,
        "fallback_triggered": fallback,
        "fallback_reason": (
            "validator" if (fallback and n_blocker) else
            "llm_error" if fallback else "none"
        ),
        "n_blocker": n_blocker,
        "n_warning": sum(
            1 for i in issues
            if i["severity"] == "warning" and i["kind"] != "undisclosed_insufficiency"
        ),
        "kinds": sorted({i["kind"] for i in issues}),
        "blocker_kinds": sorted({i["kind"] for i in issues if i["severity"] == "blocker"}),
        "warning_kinds": sorted({i["kind"] for i in issues if i["severity"] == "warning"}),
        "blocker_details": [i["detail"] for i in issues if i["severity"] == "blocker"],
        "report_snippet": (
            (out.get("draft_report") or out.get("diagnosis_report") or "")[:600]
        ),
        "final_risk": out.get("final_risk") or state.get("risk_level"),
        "declared_risk": (out.get("dual_track") or {}).get("declared_risk"),
        "diverged": bool((out.get("dual_track") or {}).get("diverged")),
        "trace": tr.as_dict(),
    }


async def main(limit: int, space: str, seed: int):
    # 分层抽样：按细胞数分桶，确保 <4 与 >=4 都有样本
    from Agents.knowledge import init_kb
    await init_kb()

    exts = ("*.bmp", "*.png", "*.jpg")
    files: list[str] = []
    for e in exts:
        files += glob.glob(f"data/{space}/**/{e}", recursive=True)
    if not files:
        raise SystemExit(f"未找到 {space} 的图片")

    random.seed(seed)
    random.shuffle(files)
    files = files[:limit]
    print(f"抽样 {len(files)} 张 → 输出 outputs/results/validator_observation.json\n")

    rows = []
    errors: list[tuple[str, str]] = []
    for i, p in enumerate(files, 1):
        try:
            r = await run_one(p, space)
        except Exception as e:
            msg = f"{type(e).__name__}: {e}"
            errors.append((os.path.relpath(p, f"data/{space}"), msg))
            print(f"[{i}/{len(files)}] ✗ {os.path.basename(p)}: {msg}")
            continue
        rows.append(r)
        flag = "FALLBACK" if r["fallback_triggered"] else "ok"

        div_mark = ""
        if r.get("diverged"):
            div_mark = f" | 分歧 {r['risk_level']}→{r.get('declared_risk')}"
        print(f"[{i}/{len(files)}] cells={r['total_cells']:<3} "
              f"risk={str(r['risk_level']):<8} blocker={r['n_blocker']} "
              f"warn={r['n_warning']} {flag:<8} "
              f"{','.join(r['blocker_kinds']) or '-'}"
              f"{' | W:' + ','.join(r['warning_kinds']) if r['warning_kinds'] else ''}"
              f"{div_mark}")

    # ---------------- 汇总 ----------------
    def bucket(n: int) -> str:
        if n == 0:
            return "unreadable(0)"
        return "cell级(1-3)" if n < U.CASE_RISK_MIN_CELLS else "case级(>=4)"

    print("\n=== 汇总 ===")
    for b in ("cell级(1-3)", "case级(>=4)", "unreadable(0)"):
        sub_all = [r for r in rows if bucket(r["total_cells"]) == b]
        if not sub_all:
            continue
        sub = [r for r in sub_all if not r["skipped"]]
        skipped = len(sub_all) - len(sub)
        print(f"{b}: n={len(sub_all)}  出报告={len(sub)}  跳过={skipped}")

        if not sub:
            continue
        fb = sum(1 for r in sub if r["fallback_triggered"])
        print(f"  fallback: {fb}/{len(sub)} ({fb/len(sub):.1%})")
        print(f"  code风险 : {dict(Counter(r['risk_level'] for r in sub))}")
        print(f"  final风险: {dict(Counter(r.get('final_risk') for r in sub))}")
        for label, field in (("blocker", "blocker_kinds"), ("warning", "warning_kinds")):
            counts: dict[str, int] = {}
            for r in sub:
                for k in r.get(field) or []:
                    counts[k] = counts.get(k, 0) + 1
            for k, c in sorted(counts.items(), key=lambda kv: -kv[1]):
                print(f"    [{label:<7}] {k:<24} {c}/{len(sub)}")

    # 跳过的原因分布（对应 action）
    skipped_actions = Counter(r["action"] for r in rows if r["skipped"])
    if skipped_actions:
        print(f"\n跳过（非 direct_report）: {dict(skipped_actions)}")

    # 分歧率统计
    reported = [r for r in rows if not r["skipped"] and r.get("declared_risk")]
    if reported:
        div = [r for r in reported if r.get("diverged")]
        rate = len(div) / len(reported)
        print(f"\n=== 双轨分歧 ===")
        print(f"可比较样本: {len(reported)}/{len(rows)}（其余报告未声明等级）")
        print(f"分歧: {len(div)}/{len(reported)} ({rate:.1%})")
        if div:
            for r in div:
                print(f"    {r['file'][:46]:<46} code={str(r['risk_level']):<8} "
                    f"LLM={str(r.get('declared_risk')):<8} → final={r.get('final_risk')}")
        # 分歧方向统计：LLM 上调还是下调
        up = sum(1 for r in div
                if _rank(r.get("declared_risk")) > _rank(r["risk_level"]))
        down = len(div) - up
        print(f"    方向: LLM 上调 {up} / 下调 {down}")

    out_path = "outputs/results/validator_observation.json"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"rows": rows, "errors": errors}, f, ensure_ascii=False, indent=2)
    print(f"\n明细已保存: {out_path}")

    if errors:
        print(f"\n {len(errors)}/{len(files)} 例异常被跳过 —— "
              f"汇总分母已扣除，结果存在偏差")
        for name, msg in errors[:5]:
            print(f"    {name}: {msg}")
        if len(errors) > 5:
            print(f"    …另有 {len(errors) - 5} 例")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=30)
    ap.add_argument("--space", default="sipakmed", choices=["sipakmed", "herlev"])
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    asyncio.run(main(args.limit, args.space, args.seed))