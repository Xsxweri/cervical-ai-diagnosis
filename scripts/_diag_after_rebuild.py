import asyncio
from services.rag_service import MedicalKnowledgeBase


def bn(s):
    return str(s or "").replace("\\", "/").split("/")[-1]


async def main():
    kb = MedicalKnowledgeBase()
    d = kb.vectorstore.get(include=["documents", "metadatas"])
    docs, metas = d["documents"], d["metadatas"]

    # ① 全库搜关键片段
    print("=" * 78)
    print("① 全库命中「25～64」/「常规人群」的 chunk")
    print("=" * 78)
    hits = 0
    for t, m in zip(docs, metas):
        if any(k in (t or "") for k in ["25～64", "25~64", "25-64", "常规人群"]):
            hits += 1
            print(f"  [{m.get('doc_type')}/{m.get('block_type') or '-'}] "
                  f"{bn(m.get('source'))[:36]} p{m.get('page')}")
            print(f"     {t[:240].replace(chr(10), ' ')!r}\n")
    print(f"  → 命中 {hits} 条\n")

    # ② R08 检索
    print("=" * 78)
    print("② R08 检索结果")
    print("=" * 78)
    q = "25–29 岁女性宫颈癌筛查首选细胞学还是 HPV 检测"
    for i, x in enumerate(await kb.retrieve_nl(q, k=5), 1):
        print(f"  {i}. cos={x['cosine']:.4f} rank={x.get('cos_rank', 0):.4f} "
              f"[{x.get('doc_type')}/{x.get('block_type') or '-'}] {bn(x['source'])[:32]}")
        print(f"     {x['content'][:200].replace(chr(10), ' ')!r}")

    # ③ 页眉噪声是否消除
    print("\n" + "=" * 78)
    print("③ 中文指南的页眉噪声检查")
    print("=" * 78)
    for name in ["中国子宫颈癌筛查指南（一）_李明珠", "子宫颈癌筛查规范（2025年版）"]:
        idx = [i for i, m in enumerate(metas or [])
               if name in str((m or {}).get("source", ""))]
        noisy = sum(1 for i in idx
                    if (docs[i] or "").lstrip()[:3] in ("●", "·")
                    or "第 24 卷" in (docs[i] or "")[:80]
                    or "Chin J Clin" in (docs[i] or "")[:120])
        print(f"  {name}: {len(idx)} chunks，页眉噪声 {noisy} 条")


asyncio.run(main())
