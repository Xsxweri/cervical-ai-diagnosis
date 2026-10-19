"""
标签 / 类别索引的单一事实来源 (Single Source of Truth)。
"""

from typing import Dict, List

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

# 医学意义上的异常 / 正常集合（与 BINARY_CLASS_MAPPING 保持一致）
SIPAKMED_ABNORMAL_CLASSES = {"Dyskeratotic", "Koilocytotic"}
SIPAKMED_NORMAL_CLASSES = {"Metaplastic", "Parabasal", "Superficial-Intermediate"}

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
