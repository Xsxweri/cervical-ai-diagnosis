"""
LLM 工厂与结构化输出模型。
"""

from __future__ import annotations

import os
from typing import List

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

load_dotenv()


class ClarificationResponse(BaseModel):
    """澄清追问的结构化输出（替代 json.loads）"""

    questions: List[str] = Field(description="2-3个追问问题")
    reasoning: str = Field(default="", description="追问理由")

_NO_THINKING = {"enable_thinking": False}

def _base_kwargs() -> dict:
    return {
        "model": os.getenv("LLM_MODEL_ID"),
        "api_key": os.getenv("OPENAI_API_KEY"),
        "base_url": os.getenv("LLM_BASE_URL"),
        # 快速失败优于长时间等待；调用方（节点）都有 fallback
        "timeout": 60.0,
        "max_retries": 1,
    }

def get_llm() -> ChatOpenAI:
    """通用工厂：追问 / 澄清 / 可解释性"""
    return ChatOpenAI(
        **_base_kwargs(),
        temperature=0.3,
        max_tokens=1500,
        extra_body=_NO_THINKING,
    )

def get_report_llm() -> ChatOpenAI:
    """
    报告生成专用：更严格的输出上限。

    报告是固定 5 节模板，实测正常长度 700-900 字符(约 500 tokens)。
    给 1200 是留 2x 余量；超出说明模型跑题了。
    """
    return ChatOpenAI(
        **_base_kwargs(),
        temperature=0.3,
        max_tokens=1200,
        extra_body=_NO_THINKING,
    )