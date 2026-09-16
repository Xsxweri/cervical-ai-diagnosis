import io
import os
import pytest
from fastapi import HTTPException
from PIL import Image
from langchain_core.messages import HumanMessage

from agent_api import validate_file_content, build_time_location, MAGIC_BYTES, ALLOWED_MIME
from Agents.diagnosis_graph import decision_router, uncertainty_assessment_agent
from Agents.conversation import _serialize_state, _deserialize_state
from utils.preprocess import CombinedCervicalCancerDataset
os.environ["TORCH_DYNAMO_DISABLE"] = "1"

def _png_bytes() -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (8, 8), "white").save(buf, format="PNG")
    return buf.getvalue()


# --- 文件魔数 / MIME 校验 ---
def test_valid_png_passes():
    validate_file_content(_png_bytes(), "a.png", "image/png")  # 不抛异常即通过


def test_text_renamed_png_rejected():
    with pytest.raises(HTTPException):
        validate_file_content(b"hello world", "a.png", "image/png")


def test_magic_bytes_keys_have_leading_dot():
    # 回归防护：键必须带前导点，否则与 splitext 的 ext 永远匹配不上（历史 bug）
    assert all(k.startswith(".") for k in MAGIC_BYTES)
    assert all(k.startswith(".") for k in ALLOWED_MIME)


# --- 时空上下文 ---
def test_build_time_location():
    s = build_time_location()
    assert "<time_location>" in s and "当前时间" in s and "当前地点" in s


# --- 决策路由 ---
def test_decision_router():
    assert decision_router({}) == "direct_report"
    assert decision_router({"action": "clarify"}) == "clarify"


# --- 不确定性分级 ---
def test_high_uncertainty_goes_clarify():
    out = uncertainty_assessment_agent({"cv_result": {"max_uncertainty": 0.5}})
    assert out["action"] == "clarify" and out["risk_level"] == "pending"


def test_low_uncertainty_goes_direct_report():
    out = uncertainty_assessment_agent(
        {"cv_result": {"max_uncertainty": 0.05, "abnormal_cells": 0, "total_cells": 10}}
    )
    assert out["action"] == "direct_report" and out["risk_level"] == "low"


# --- 二分类映射修正回归（Metaplastic/Parabasal 应为 normal）---
def test_binary_mapping_metaplastic_parabasal_normal():
    ds = CombinedCervicalCancerDataset([], transform=None, config={"data": {}}, classification_mode="binary")
    assert ds.class_mapping["Metaplastic"] == "normal"
    assert ds.class_mapping["Parabasal"] == "normal"


# --- 会话序列化往返 ---
def test_serialize_roundtrip():
    state = {"risk_level": "low", "conversation_history": [HumanMessage(content="hi")]}
    back = _deserialize_state(_serialize_state(state))
    assert back["risk_level"] == "low"
    assert back["conversation_history"][0].content == "hi"