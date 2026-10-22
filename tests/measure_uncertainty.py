import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch, torch._dynamo
from dotenv import load_dotenv
load_dotenv()
import glob, numpy as np, asyncio
from services.vision_service import analyze_image_from_path

CLASSES = ["im_Dyskeratotic", "im_Koilocytotic", "im_Metaplastic",
           "im_Parabasal", "im_Superficial-Intermediate"]
PER_CLASS = 12
THRESHOLDS = (0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80)


def collect_files():
    files = []
    for c in CLASSES:
        files += glob.glob(f"data/sipakmed/{c}/**/*.bmp", recursive=True)[:PER_CLASS]
    return files


def report(name, arr):
    a = np.array(arr, dtype=float)
    if a.size == 0:
        print(f"\n[{name}] 空数组"); return
    print(f"\n[{name}] n={a.size} min={a.min():.4f} mean={a.mean():.4f} max={a.max():.4f}")
    for q in (50, 75, 90, 95, 99):
        print(f"  P{q} = {np.percentile(a, q):.4f}")
    for th in THRESHOLDS:
        print(f"  超过 {th:>4}: {(a > th).mean() * 100:5.1f}%")


async def main():
    files = collect_files()
    print(f"[glob] 命中 {len(files)} 张 (跨 {len(CLASSES)} 类)")
    if not files:
        print("[!] 没命中, 确认从项目根运行"); return
    results = [await analyze_image_from_path(f) for f in files]

    n_cells = [len(r.get("cells") or []) for r in results]
    print(f"[每图细胞数] mean={np.mean(n_cells):.2f} max={max(n_cells)} "
          f"(>1 说明检测框重复，需按第3项处理)")

    def cells_of(pred):
        return [c for r in results for c in (r.get("cells") or []) if pred(c)]

    per_cell = [c["uncertainty"] for c in cells_of(lambda c: True)]
    bin_cells = [c["uncertainty"] for c in cells_of(lambda c: c.get("model_used") == "binary")]
    multi_cells = [c["uncertainty"] for c in cells_of(lambda c: c.get("model_used") == "multiclass")]

    agg = results[0].get("uncertainty_aggregation", "unknown") if results else "unknown"
    per_img = [r["max_uncertainty"] for r in results if r.get("max_uncertainty") is not None]

    report("每细胞 uncertainty(全部)", per_cell)
    report("每细胞 uncertainty(二分类路径)", bin_cells)
    report("每细胞 uncertainty(多分类路径)", multi_cells)
    report(f"每图 max_uncertainty(路由实际比较),聚合={agg})", per_img)

    if per_img:
        a = np.array(per_img, dtype=float)
        print(f"\n[P90 vs 旧 max 的差异] P90={np.percentile(a, 90):.4f} "
              f"P95={np.percentile(a, 95):.4f} mean={a.mean():.4f}")


asyncio.run(main())