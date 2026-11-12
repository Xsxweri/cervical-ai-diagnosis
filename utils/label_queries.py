"""
标签 → 检索查询映射。

与 utils/label_mapping.py 的分工：
    label_mapping  : 标签空间定义（顺序即模型输出维度，不可乱序）
    label_queries  : 每个标签对应的检索 query 文本

设计原则：query 面向「管理决策」，不面向「形态学描述」

当前语料（WHO 筛查指南、中国筛查指南、TBS 综述、研究论文）覆盖
筛查策略、异常后分流、治疗指征，但**不含细胞形态学判据**
（核质比、挖空、核拥挤、成熟指数等）。

若 query 问形态（如「挖空细胞 非典型角化」），检索只能靠词面重叠
命中「LSIL/HSIL 分类枚举页」，语义错配，且 evidence_sufficiency
会因此虚高（cos 够 0.50 但内容无关）。
"""

from __future__ import annotations

# 每类标签对应一段检索 query。新增标签空间/类别时必须同步，
# 否则该标签会静默回退到通用模糊查询（检索质量下降但无告警）。
CLASSIFICATION_QUERIES: dict[str, str] = {
    # ===正常/异常，粗粒度分类===
    # 粗粒度标签本就用于分流，保留管理导向
    "normal": "正常鳞状上皮细胞形态 NILM 筛查间隔 随访建议",
    "abnormal": "宫颈细胞学异常 TBS分类 ASC-US LSIL HSIL 分流处理原则",

    # ===形态学亚型（SIPaKMeD）===
    # 双段式：前半形态学（命中 TBS 分册图注），后半分级锚定（避免命中分类枚举页）
    "Dyskeratotic": "角化不良细胞 胞质角化 核深染 形态学特征 LSIL",
    "Koilocytotic": "挖空细胞 核周空晕 HPV感染 形态学特征 LSIL",
    "Metaplastic": "鳞状化生细胞 化生 移行区 形态学特征 良性反应性改变",
    "Parabasal": "副基底层细胞 小细胞 核质比 形态学特征 萎缩 判读",
    "Superficial-Intermediate": "表层中层鳞状细胞 成熟指数 雌激素影响 形态学特征",

    # ===Herlev 7 级===
    "normal_superficial":  "表层鳞状细胞 多边形 核小 形态学特征 NILM",
    "normal_intermediate": "中层鳞状细胞 舟状 核居中 形态学特征 NILM",
    "normal_columnar": "颈管柱状细胞 腺上皮细胞 形态学特征 标本质量",
    "light_dysplastic": "轻度异型增生 CIN1 LSIL 细胞学特征与随访间隔",
    "moderate_dysplastic": "中度异型增生 CIN2 HSIL 阴道镜指征与处理原则",
    "severe_dysplastic": "重度异型增生 CIN3 HSIL 锥切与随访管理",
    "carcinoma_in_situ": "重度异型增生 CIN3 HSIL 原位癌 形态学特征",
}


def get_classification_query(classification: str) -> str:
    """取标签对应的检索 query。

    未知标签回退为使用原始标签本身 —— 不回退到空串或无意义常量，
    这样新标签至少还能检索到相关内容，同时保留一条告警线索。
    """
    query = CLASSIFICATION_QUERIES.get(classification)
    if query is not None:
        return query
    print(f"[RAG] 未知分类 '{classification}'，使用原始标签检索")
    return f"宫颈细胞学 {classification} 形态学特征及临床意义"


def missing_classification_queries() -> list[str]:
    """标签空间里未被 CLASSIFICATION_QUERIES 覆盖的标签。"""
    from utils.label_mapping import MULTICLASS_SPACES

    labels = {c for cs in MULTICLASS_SPACES.values() for c in cs}
    return sorted(labels - set(CLASSIFICATION_QUERIES))


def warn_uncovered_labels() -> list[str]:
    """启动期自检：未覆盖标签只告警，不抛异常。

    未覆盖的后果是检索质量降级，不影响诊断流程，
    不应阻断服务启动（与「RAG 永不阻断启动」的约定一致）。
    """
    try:
        missing = missing_classification_queries()
    except Exception as e:
        print(f"[RAG] 标签映射自检失败（已忽略）: {e}")
        return []
    if missing:
        print(
            f"[RAG] 警告：CLASSIFICATION_QUERIES 未覆盖标签 {missing}，"
            f"这些标签将回退到通用模糊查询。"
            f"请同步 utils/label_queries.py 的 CLASSIFICATION_QUERIES"
        )
    return missing