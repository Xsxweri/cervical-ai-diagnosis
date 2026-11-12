"""清单 vs 实际文件 一致性检查。重建索引前跑，避免 doc_type 静默降级。"""
import glob, os, sys
import yaml
from utils.mineru_loader import discover_mineru_json, source_name_from
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
ROOT = str(_ROOT / "Cervical_cytology")

pdfs = {os.path.basename(p) for p in glob.glob(f"{ROOT}/*.pdf")}
jsons = {source_name_from(f) for f in discover_mineru_json(ROOT)}

with open(_ROOT / "configs" / "corpus_manifest.yaml", encoding="utf-8") as f:
    spec = yaml.safe_load(f) or {}
listed = {d["file"] for d in spec.get("docs") or []}

print(f"PDF 文件      : {len(pdfs)}")
print(f"MinerU JSON   : {len(jsons)}")
print(f"清单条目      : {len(listed)}\n")

actual = pdfs | jsons
missing_in_manifest = sorted(actual - listed)
stale_in_manifest = sorted(listed - actual)

ok = True
if missing_in_manifest:
    ok = False
    print("❌ 未登记到清单（会被判 research / 无加成）：")
    for x in missing_in_manifest:
        print(f"   - {x}")

if stale_in_manifest:
    ok = False
    print("❌ 清单里有但文件不存在（疑似改名/删除后未同步）：")
    for x in stale_in_manifest:
        print(f"   - {x}")

if ok:
    print("✅ 清单与实际文件完全一致")
else:
    sys.exit(1)
