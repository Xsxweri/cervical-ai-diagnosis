from Agents.nodes.retrieval import evidence_sufficiency_from

base = {"content": "x", "source": "a.pdf", "page": 1, "cosine": 0.65}
cases = {
    "3 篇指南":     [dict(base, doc_type="national_guideline") for _ in range(3)],
    "3 篇原始研究": [dict(base, doc_type="research") for _ in range(3)],
    "1 指南+2 研究": [dict(base, doc_type="national_guideline"),
                     dict(base, doc_type="research"),
                     dict(base, doc_type="research")],
    "未标注×3":     [dict(base) for _ in range(3)],
}
for name, items in cases.items():
    s, n, mc = evidence_sufficiency_from(items)
    print(f"{name:14} sufficiency={s:.2f}  n={n}")
