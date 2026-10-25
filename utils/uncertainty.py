"""
不确定性度量与分级阈值的单一事实来源。
阈值在此定义；不确定性改用**归一化熵**（见normalized_entropy）。
"""

import math
import os

import numpy as np

# ---------------------------------------------------------------------------
# 阈值（作用在归一化熵 ∈ [0,1] 上）
#   1.0 = 完全不确定（各类别等概率），0.0 = 完全确定
# ---------------------------------------------------------------------------
UNCERTAINTY_HIGH = float(os.getenv("UNCERTAINTY_HIGH", "0.60"))
UNCERTAINTY_MEDIUM = float(os.getenv("UNCERTAINTY_MEDIUM", "0.30"))

METRIC_NAME = "normalized_entropy"
UNCERTAINTY_PERCENTILE = int(os.getenv("UNCERTAINTY_PERCENTILE", "90"))
CASE_RISK_MIN_CELLS = int(os.getenv("CASE_RISK_MIN_CELLS", "4"))
MAX_CLARIFICATION_ROUNDS = int(os.getenv("MAX_CLARIFICATION_ROUNDS", "2"))


def risk_scope_for(total_cells: int) -> str:
    """'case'（占比有意义）或 'cell'（细胞级提示）。"""
    return "case" if total_cells >= CASE_RISK_MIN_CELLS else "cell"
def normalized_entropy(mean_probs) -> float:
    """归一化预测熵 H(p)/ln(K) ∈ [0,1]。

    与类别数 K 无关，因此 2 类（二分类路径）与 5 类（多分类路径）的结果可直接比较，
    这正是原先 max(std) 做不到的。

    Args:
        mean_probs: MC Dropout 的平均预测概率（长度 = K）
    """
    p = np.asarray(mean_probs, dtype=np.float64).reshape(-1)
    if p.size <= 1:
        return 0.0
    p = np.clip(p, 1e-12, None)
    total = p.sum()
    if total <= 0:
        return 0.0
    p = p / total
    return float(-np.sum(p * np.log(p)) / math.log(p.size))


def classify_uncertainty(value: float) -> str:
    """把归一化不确定性映射为 "high" / "medium" / "low"。"""
    if value > UNCERTAINTY_HIGH:
        return "high"
    if value > UNCERTAINTY_MEDIUM:
        return "medium"
    return "low"

def risk_level_from_ratio(abnormal_ratio: float) -> str:
    """按异常细胞占比给出粗粒度风险等级（与 Agent 状态机共用同一判据）。"""
    if abnormal_ratio <= 0:
        return "low"
    if abnormal_ratio < 0.3:
        return "medium"
    if abnormal_ratio < 0.6:
        return "high"
    return "emergency"