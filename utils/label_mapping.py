"""
标签 / 类别索引的单一事实来源 (Single Source of Truth)。
"""

from typing import Dict, List, Optional

# ---------------------------------------------------------------------------
# 二分类 (binary)
# 由本模块 canonical 固定，训练/推理统一引用
# ---------------------------------------------------------------------------
BINARY_CLASSES: List[str] = ["abnormal", "normal"]
BINARY_ABNORMAL_IDX: int = BINARY_CLASSES.index("abnormal")  # -> 0
BINARY_NORMAL_IDX: int = BINARY_CLASSES.index("normal")      # -> 1

# 原始类别名 -> 统一二分类类别
BINARY_CLASS_MAPPING: Dict[str, str] = {
    # Herlev 7 类
    "normal_columnar": "normal",
    "normal_intermediate": "normal",
    "normal_superficial": "normal",
    "light_dysplastic": "abnormal",
    "moderate_dysplastic": "abnormal",
    "severe_dysplastic": "abnormal",
    "carcinoma_in_situ": "abnormal",
    # SIPaKMeD 5 类
    "Superficial-Intermediate": "normal",
    "Metaplastic": "normal",
    "Parabasal": "normal",
    "Dyskeratotic": "abnormal",
    "Koilocytotic": "abnormal",
}


def binary_index(name: str) -> int:
    """返回二分类类别名对应的索引（''abnormal'' -> 0, ''normal'' -> 1）。"""
    return BINARY_CLASSES.index(name)


def is_abnormal_binary(idx: int) -> bool:
    """给定二分类 argmax 索引，判断是否为异常。"""
    return idx == BINARY_ABNORMAL_IDX



# ---------------------------------------------------------------------------
# SIPaKMeD 5 类（检测/多分类共用同一顺序）
# ---------------------------------------------------------------------------
SIPAKMED_CLASSES: List[str] = [
    "Dyskeratotic",
    "Koilocytotic",
    "Metaplastic",
    "Parabasal",
    "Superficial-Intermediate",
]
SIPAKMED_CLASS_TO_IDX: Dict[str, int] = {c: i for i, c in enumerate(SIPAKMED_CLASSES)}

# SIPaKMeD 原始目录名 -> 类别名
SIPAKMED_DIR_MAPPING: Dict[str, str] = {
    "im_Dyskeratotic": "Dyskeratotic",
    "im_Koilocytotic": "Koilocytotic",
    "im_Metaplastic": "Metaplastic",
    "im_Parabasal": "Parabasal",
    "im_Superficial-Intermediate": "Superficial-Intermediate",
}

# 医学意义上的三分类（SIPaKMeD 官方定义）
#   normal  : 正常鳞状细胞
#   benign  : 良性反应性改变（化生），非肿瘤性，不得与 Koilocytotic/Dyskeratotic 并列表述为"异常分型"
#   abnormal: 异常但非恶性（LSIL 范畴）
SIPAKMED_ABNORMAL_CLASSES = {"Dyskeratotic", "Koilocytotic"}
SIPAKMED_BENIGN_CLASSES = {"Metaplastic"}
SIPAKMED_NORMAL_CLASSES = {"Parabasal", "Superficial-Intermediate"}
# 评估脚本用的"非肿瘤性"合并集合（保持检测指标口径不变）
SIPAKMED_NON_NEOPLASTIC = SIPAKMED_NORMAL_CLASSES | SIPAKMED_BENIGN_CLASSES


# ---------------------------------------------------------------------------
# Herlev 7类病理分级
# 顺序：正常3类 -> 病变4类（由轻到重）
# ---------------------------------------------------------------------------
HERLEV_CLASSES: List[str] = [
    "normal_superficial",
    "normal_intermediate",
    "normal_columnar",
    "light_dysplastic",
    "moderate_dysplastic",
    "severe_dysplastic",
    "carcinoma_in_situ",
]
HERLEV_CLASS_TO_IDX: Dict[str, int] = {c: i for i, c in enumerate(HERLEV_CLASSES)}
# Herlev 正常 / 异常集合
HERLEV_NORMAL_CLASSES = {"normal_superficial", "normal_intermediate", "normal_columnar"}
HERLEV_ABNORMAL_CLASSES = {
    "light_dysplastic", "moderate_dysplastic",
    "severe_dysplastic", "carcinoma_in_situ",
}

# ---------------------------------------------------------------------------
# 严重度分层（跨标签空间对齐用）
#   nil    : 无上皮内病变
#   benign : 良性反应性改变
#   low    : LSIL / CIN1
#   high   : HSIL / CIN2+ / CIS
# 注意：SIPaKMeD 只能覆盖到 low，无法表达 high——这不是缺陷，是标签空间的能力边界。
# ---------------------------------------------------------------------------
SEVERITY_TIER: Dict[str, str] = {
    # SIPaKMeD
    "Superficial-Intermediate": "nil",
    "Parabasal": "nil",
    "Metaplastic": "benign",
    "Koilocytotic": "low",
    "Dyskeratotic": "low",
    # Herlev
    "normal_superficial": "nil",
    "normal_intermediate": "nil",
    "normal_columnar": "nil",
    "light_dysplastic": "low",
    "moderate_dysplastic": "high",
    "severe_dysplastic": "high",
    "carcinoma_in_situ": "high",
}
TIER_RANK: Dict[str, int] = {"nil": 0, "benign": 1, "low": 2, "high": 3}

# 高严重度标签 → 病例级风险下限
# 目的：防止"20 个细胞里只有 1 个 CIS → 占比 5% → low"这类风险低估
HERLEV_RISK_FLOOR: Dict[str, str] = {
    "carcinoma_in_situ": "high",
    "severe_dysplastic": "high",
    "moderate_dysplastic": "medium",
    "light_dysplastic": "medium",
}

_RISK_ORDER: List[str] = ["low", "medium", "high", "emergency"]


def abnormal_classes_for(label_space: str) -> set:
    """返回该标签空间下的异常类集合；未知空间直接报错，不静默回退。"""
    if label_space == "herlev":
        return set(HERLEV_ABNORMAL_CLASSES)
    if label_space == "sipakmed":
        return set(SIPAKMED_ABNORMAL_CLASSES)
    raise ValueError(f"未知标签空间 '{label_space}'，可选: sipakmed / herlev")


def risk_floor_for(cls: str, label_space: str) -> str | None:
    """该类别对病例级风险的下限；无下限返回 None。"""
    if label_space != "herlev":
        return None
    return HERLEV_RISK_FLOOR.get(cls)


def severity_tier_for(cls: str) -> str | None:
    return SEVERITY_TIER.get(cls)


def max_risk(*levels: str) -> str:
    """在所有已知等级中取最高。"""
    known = [lv for lv in levels if lv in _RISK_ORDER]
    return max(known, key=_RISK_ORDER.index) if known else "low"

def tier_rank_of(cls: str) -> Optional[int]:
    """类别 → 严重度序号；未知类别返回 None。"""
    tier = SEVERITY_TIER.get(cls)
    return TIER_RANK.get(tier) if tier else None


def dual_conflict_reason(primary_cls: str, aux_cls: str) -> Optional[str]:
    """dual 模式的一致性校验（单向：只把「主空间更轻」判为冲突）。

    返回 None 表示一致；否则返回冲突原因。

    为什么单向：SIPaKMeD 只能表达到 low（LSIL），无法表达 high（CIN2+）。
    Herlev 判 high 而 SIPaKMeD 判 low 属标签空间能力差异，不是矛盾；
    反之 Herlev 判 nil/benign 而 SIPaKMeD 判 low，说明主空间低估，必须上报。
    """
    pr = tier_rank_of(primary_cls)
    ar = tier_rank_of(aux_cls)
    if pr is None or ar is None:
        return None
    if ar >= TIER_RANK["low"] and pr < ar:
        return (
            f"主空间判 {primary_cls}({SEVERITY_TIER[primary_cls]})"
            f"低于辅助空间 {aux_cls}({SEVERITY_TIER[aux_cls]})，存在病变漏判风险"
        )
    return None

# ---------------------------------------------------------------------------
# 多分类标签空间注册表
# 通过 label_space 名称显式选择标签空间，避免"标签空间随数据集组合漂移"。
# 默认 "sipakmed"（5 类），与现有 5 类多分类权重兼容。
# ---------------------------------------------------------------------------
MULTICLASS_SPACES: Dict[str, List[str]] = {
    "sipakmed": list(SIPAKMED_CLASSES),
    "herlev": list(HERLEV_CLASSES),
}
DEFAULT_LABEL_SPACE = "sipakmed"

def get_multiclass_classes(label_space: str = DEFAULT_LABEL_SPACE) -> List[str]:
    """返回指定多分类标签空间的类别列表；未知空间直接报错，不静默回退。"""
    if label_space not in MULTICLASS_SPACES:
        raise ValueError(
            f"未知的多分类标签空间 '{label_space}'，可选: {sorted(MULTICLASS_SPACES)}"
        )
    return list(MULTICLASS_SPACES[label_space])

# 以SIPaKMeD的5类为主标签空间；Herlev的7类作为病理分级补充
MULTICLASS_CLASSES: List[str] = list(SIPAKMED_CLASSES)
MULTICLASS_CLASS_TO_IDX: Dict[str, int] = {c: i for i, c in enumerate(MULTICLASS_CLASSES)}

# 检测器可选的结构类别（胞核）。默认不启用，作为独立类别追加在细胞类别之后。
NUCLEUS_CLASS_NAME = "nucleus"
