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
THRESHOLDS = (0.02, 0.03, 0.05, 0.08, 0.10, 0.15)


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
    per_cell = [c.get("uncertainty", 0.0) for r in results if r.get("cells") for c in r["cells"]]
    per_img = [r["max_uncertainty"] for r in results if r.get("max_uncertainty") is not None]
    print(f"[结果] 有细胞 {len(per_img)}/{len(results)} 张")
    report("每细胞 uncertainty", per_cell)
    report("每图 max_uncertainty(路由实际比较)", per_img)


asyncio.run(main())