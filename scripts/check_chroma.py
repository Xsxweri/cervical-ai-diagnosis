"""确认向量库的距离度量与 score 方向。一次性诊断脚本，可随时删除。"""
import asyncio
from services.rag_service import MedicalKnowledgeBase


async def main():
    kb = MedicalKnowledgeBase()
    if kb.vectorstore is None:
        print("向量库不可用，跳过")
        return

    meta = kb.vectorstore._collection.metadata
    print(f"collection metadata: {meta}")
    print(f"hnsw:space = {(meta or {}).get('hnsw:space', '未设置 → 默认 l2')}")

    # 取 3 个已知相关的查询，看 score 分布与排序
    for q in ["HSIL 诊断标准", "挖空细胞 HPV 感染", "CIN2 阴道镜指征"]:
        items = await kb.retrieve(q, k=5)
        print(f"\n=== {q} ===")
        for i, it in enumerate(items):
            print(f"  rank={i+1} score={it['score']:.4f}  {it['source']} p{it['page']}")

asyncio.run(main())