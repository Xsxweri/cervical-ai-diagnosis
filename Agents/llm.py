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


def get_llm() -> ChatOpenAI:
    """每次调用新建客户端，避免跨协程共享连接状态。"""
    return ChatOpenAI(
        model=os.getenv("LLM_MODEL_ID"),
        temperature=0.3,
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("LLM_BASE_URL"),
        timeout=120.0,
        max_retries=3,
    )