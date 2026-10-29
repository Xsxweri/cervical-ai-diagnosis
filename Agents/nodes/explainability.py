"""节点9:可解释性报告(形态学 + 不确定性 + 可选 RAG/Grad-CAM)"""

import asyncio
from typing import Any, Dict

from langchain_core.prompts import ChatPromptTemplate

from Agents.llm import get_llm
from Agents.state import DiagnosisState
from Prompts import raw
from services.explainability_service import compute_case_gradcam, grad_cam_enabled


async def explainable_report_agent(state: DiagnosisState) -> Dict[str, Any]:
    """
    生成带可解释性的综合报告
    结合形态学分析 + 不确定性 + (可选)RAG医学依据
    无论是否检索到知识都会生成解释，避免低不确定性病例缺失可解释性
    """
    cv_results = state.get("cv_result") or {}
    knowledge = state.get("retrieved_knowledge") or []

    # 构建可解释性上下文
    explain_context = f'''
    诊断结果: {state.get("diagnosis_report", "")}

    可解释性信息:
    1. 模型关注区域: AI主要关注细胞的核区域和细胞质边界
    2. 关键形态特征:
    '''

    for i, cell in enumerate((cv_results.get("cells") or [])[:3], 1):
        morph = cell.get("morphology") or {}
        explain_context += f'''
    细胞{i}:
    - 核质比: {(morph.get("nuclear_cytoplasmic_ratio") or 0.0):.3f}
    - 圆形度: {(morph.get("circularity") or 0.0):.3f}
    - 解读: {morph.get("interpretation", "")}
    '''

    explain_context += f'''
    3. 不确定性分析: {cv_results.get("uncertainty_metric", "normalized_entropy")}={(state.get("max_uncertainty") or 0.0):.3f}
    (图像级取 {cv_results.get("uncertainty_aggregation", "p90")} 聚合，范围 0~1, 越大越不确定)
    '''

    # 仅当检索到知识时追加医学依据
    if knowledge:
        explain_context += '\n  相关医学依据:\n'
        for doc in knowledge[:3]:
            explain_context += f'    - {(doc.get("content", "") or "")[:150]}...\n'

    # Grad-CAM 两种触发方式：
    #   ENABLE_GRADCAM=true      报告生成时同步产出，并把注意力描述注入 LLM（成本高）
    #   ENABLE_GRADCAM=false     不参与报告，改由 POST /gradcam 按需生成
    gradcam_path = None
    if grad_cam_enabled():
        gradcam = await asyncio.to_thread(
            compute_case_gradcam, state.get("image_path"), cv_results
        )
        if gradcam:
            gradcam_path = gradcam.get("overlay_path")
            explain_context += f'''
       4. 模型注意力(Grad-CAM): {gradcam.get("attention_analysis", "")}
       '''

    explain_prompt = ChatPromptTemplate.from_messages([
        ("system", raw("explainability_system")),
        ("human", "{context}\n\n请生成可解释性报告。")
    ])

    # 从检索结果构建参考文献（按 source:page 去重，过滤未知来源）
    references = []
    seen = set()
    for doc in knowledge:
        source = doc.get("source", "")
        page = doc.get("page", 0)
        key = f"{source}:{page}"
        if key in seen or not source or source == "未知":
            continue
        seen.add(key)
        references.append(f"{source} (第{page}页)")

    try:
        llm = get_llm()
        chain = explain_prompt | llm
        response = await chain.ainvoke({"context": explain_context})
        full_report = (state.get("diagnosis_report") or "") + "\n\n## 判读逻辑解释\n\n" + response.content
    except Exception as e:
        print(f"LLM生成可解释性报告失败: {e}")
        full_report = state.get("diagnosis_report") or ""

    return {
        "diagnosis_report": full_report,
        "references": references,
        "gradcam_path": gradcam_path,
        "current_step": "explainable_report",
        "is_complete": True
    }