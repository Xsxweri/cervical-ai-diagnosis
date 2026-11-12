"""
LLM 辅助查询改写。仅在规则层覆盖率为 0 时调用。

安全边界（三条硬约束）：
  1. 只做术语归一，不产出诊断结论；
  2. 输出必须命中 utils.query_rewrite 的词表，否则丢弃；
  3. 任何异常/超时 → 返回空，调用方退化到纯规则结果。
"""

from __future__ import annotations

import asyncio
import logging
import os
from typing import List

from pydantic import BaseModel, Field

from Agents.llm import get_llm
from utils.query_rewrite import validate_terms

logger = logging.getLogger("query_rewrite")

TIMEOUT = float(os.getenv("RAG_REWRITE_TIMEOUT", "8"))
ENABLED = os.getenv("RAG_LLM_REWRITE", "false").strip().lower() in {"1", "true", "yes", "on"}

_PROMPT = """你是宫颈细胞学术语归一化工具。

## 任务
把医生口语化的描述，映射为标准 TBS 术语。

## 铁律
- 只能输出下列词表中**已经存在**的术语，逐字复制，不得改写、不得创造新词：
{lexicon}
- 只做术语映射，不解释、不推论、不下诊断结论
- 输入里没提到的病变，不得输出
- 若无法映射，返回空数组

## 输出格式（只返回 JSON，不要 Markdown 代码块）
{{"terms": ["术语1", "术语2"]}}

## 输入
{query}
"""


class RewriteOutput(BaseModel):
    terms: List[str] = Field(default_factory=list, description="命中的规范术语")


async def rewrite_query_llm(query: str, lexicon: str) -> list[str]:
    """返回通过白名单校验的术语；失败返回 []。"""
    if not ENABLED or not (query or "").strip():
        return []
    try:
        llm = get_llm().with_structured_output(RewriteOutput)
        out = await asyncio.wait_for(
            llm.ainvoke(_PROMPT.format(lexicon=lexicon, query=query)), timeout=TIMEOUT
        )
        terms = validate_terms(list(out.terms or []))
        if terms != list(out.terms or []):
            logger.warning("LLM 改写含非法术语，已丢弃：%s → %s", out.terms, terms)
        return terms
    except Exception as e:
        logger.info("LLM 改写失败，退化到规则层：%s", e)
        return []