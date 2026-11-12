import asyncio
from services.rag_service import MedicalKnowledgeBase

async def main():
    kb = MedicalKnowledgeBase()
    # 完全无关的查询：应当取得很大（低余弦）
    for q in ['Python 编程教程', '股票投资策略', '今天天气怎么样']:
        items = await kb.retrieve(q, k=5)
        print(f'\n=== {q} ===')
        for it in items:
            cos = 1 - (it['score'] ** 2) / 2
            print(f'  d={it["score"]:.4f}  cos={cos:.4f}  {it["source"].split(chr(92))[-1]}')

asyncio.run(main())