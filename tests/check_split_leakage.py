"""核对分类器数据集的病例分组划分（可独立于训练运行）。

用法:
    python tests/check_split_leakage.py configs/sipakmed_classifier.yaml sipakmed
    python tests/check_split_leakage.py configs/herlev_classifier.yaml herlev
"""
import os
import sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.preprocess import CombinedCervicalCancerDataset, load_config
from utils.data_loader import case_key, group_split_indices

cfg_path = sys.argv[1] if len(sys.argv) > 1 else "configs/sipakmed_classifier.yaml"
label_space = sys.argv[2] if len(sys.argv) > 2 else "sipakmed"

cfg = load_config(cfg_path)
ds = CombinedCervicalCancerDataset(
    cfg["data"]["datasets"], None, cfg, "multiclass", label_space=label_space)

total = len(ds.images)
tr, va, te = group_split_indices(ds, cfg["data"]["train_ratio"], cfg["data"]["val_ratio"])

def cases(idxs):
    return {case_key(ds.images[i][0]) for i in idxs}

c_tr, c_va, c_te = cases(tr), cases(va), cases(te)
all_cases = c_tr | c_va | c_te

print(f"\n[病例] 总 {len(all_cases)} | train {len(c_tr)} val {len(c_va)} test {len(c_te)}")
print(f"[图像] 总 {total} | train {len(tr)} val {len(va)} test {len(te)}")

leak = (c_tr & c_va) | (c_tr & c_te) | (c_va & c_te)
assert not leak, f"泄漏 {len(leak)} 个病例: {sorted(leak)[:5]}"
print(f"[泄漏] 0 ")

for name, idxs, ratio in (("train", tr, cfg["data"]["train_ratio"]),
                          ("val", va, cfg["data"]["val_ratio"]),
                          ("test", te, cfg["data"]["test_ratio"])):
    got = len(idxs) / total
    print(f"[比例] {name}: 图像实际 {got:.3f} / 配置 {ratio:.3f}")

print(f"\n{'类别':<28}{'train':>7}{'val':>7}{'test':>7}{'合计':>7}")
bad = False
for cls_idx, cls_name in enumerate(ds.target_names):
    cols = []
    for idxs in (tr, va, te):
        cols.append(sum(1 for i in idxs if ds.images[i][1] == cls_idx))
    tot = sum(cols)
    mark = ""
    if min(cols) == 0:
        mark = "有空集"
        bad = True
    print(f"{cls_name:<28}{cols[0]:>7}{cols[1]:>7}{cols[2]:>7}{tot:>7}{mark}")

print("\n" + ("存在某类在某个 split 为空，评估会失真" if bad else "各类在三个 split 均有样本"))