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
    out = uncertainty_assessment_agent({"cv_result": {"max_uncertainty": 0.9}})
    assert out["action"] == "clarify" and out["risk_level"] == "pending"


def test_low_uncertainty_goes_direct_report():
    out = uncertainty_assessment_agent(
        {"cv_result": {"max_uncertainty": 0.02, "abnormal_cells": 0, "total_cells": 10}}
    )
    assert out["action"] == "direct_report" and out["risk_level"] == "low"

def test_medium_uncertainty_goes_deep_retrieve():
    out = uncertainty_assessment_agent({"cv_result": {"max_uncertainty": 0.5}})
    assert out["action"] == "deep_retrieve" and out["risk_level"] == "medium"

# --- 二分类映射修正回归（Metaplastic/Parabasal 应为 normal）---
def test_binary_mapping_metaplastic_parabasal_normal():
    ds = CombinedCervicalCancerDataset([], transform=None, config={"data": {}}, classification_mode="binary")
    assert ds.class_mapping["Metaplastic"] == "normal"
    assert ds.class_mapping["Parabasal"] == "normal"


# --- 致命回归：二分类索引顺序必须锁死 (0=abnormal, 1=normal) ---
# 历史 bug: 训练端 ['abnormal','normal']，推理端 ['normal','abnormal'] 且用 idx==1 判异常，
# 导致整条业务链把正常报成异常。以下用例确保索引顺序永不再漂移。
def test_binary_class_index_order_is_canonical():
    from utils.label_mapping import (
        BINARY_CLASSES, BINARY_ABNORMAL_IDX, BINARY_NORMAL_IDX, is_abnormal_binary,
    )
    assert BINARY_CLASSES == ["abnormal", "normal"]
    assert BINARY_ABNORMAL_IDX == 0
    assert BINARY_NORMAL_IDX == 1
    assert is_abnormal_binary(0) is True
    assert is_abnormal_binary(1) is False


def test_training_dataset_binary_index_matches_canonical():
    from utils.label_mapping import BINARY_CLASSES
    ds = CombinedCervicalCancerDataset([], transform=None, config={"data": {}}, classification_mode="binary")
    assert ds.target_names == BINARY_CLASSES
    assert ds.class_to_idx == {"abnormal": 0, "normal": 1}
    # 每一个原始类名都映射到 canonical 索引，不允许出现未映射/错位
    for raw, mapped in ds.class_mapping.items():
        assert ds.class_to_idx[mapped] == BINARY_CLASSES.index(mapped)


def test_inference_binary_order_matches_training():
    """推理端的 BINARY_CLASSES 必须与训练端一致（否则标签反向）。"""
    from utils.label_mapping import BINARY_CLASSES
    vs = pytest.importorskip("services.vision_service")
    assert list(vs.BINARY_CLASSES) == list(BINARY_CLASSES)
    assert vs.BINARY_CLASSES[0] == "abnormal"

def test_multiclass_index_order_is_canonical():
    from utils.label_mapping import MULTICLASS_CLASSES, MULTICLASS_CLASS_TO_IDX
    assert MULTICLASS_CLASSES == [
        "Dyskeratotic", "Koilocytotic", "Metaplastic",
        "Parabasal", "Superficial-Intermediate",
    ]
    assert MULTICLASS_CLASS_TO_IDX["Superficial-Intermediate"] == 4


def test_training_multiclass_space_matches_canonical():
    """数据集多分类标签空间必须等于 canonical 5 类（不允许随数据集组合变成 12 类）。"""
    from utils.label_mapping import MULTICLASS_CLASSES
    ds = CombinedCervicalCancerDataset([], transform=None, config={"data": {}}, classification_mode="multiclass")
    assert ds.target_names == MULTICLASS_CLASSES
    assert list(ds.class_to_idx.values()) == [0, 1, 2, 3, 4]


def test_inference_multiclass_matches_canonical():
    from utils.label_mapping import MULTICLASS_CLASSES
    vs = pytest.importorskip("services.vision_service")
    assert list(vs.MULTICLASS_CLASSES) == list(MULTICLASS_CLASSES)

# --- Herlev 病理分级标签空间 ---
def test_herlev_classes_canonical():
    from utils.label_mapping import HERLEV_CLASSES, HERLEV_CLASS_TO_IDX
    assert len(HERLEV_CLASSES) == 7
    assert HERLEV_CLASSES[0] == "normal_superficial"
    assert HERLEV_CLASS_TO_IDX["carcinoma_in_situ"] == 6


def test_get_multiclass_classes_unknown_space_raises():
    from utils.label_mapping import get_multiclass_classes
    with pytest.raises(ValueError):
        get_multiclass_classes("not_a_space")


def test_multiclass_label_space_selection():
    ds_sipa = CombinedCervicalCancerDataset(
        [], transform=None, config={"data": {}}, classification_mode="multiclass")
    ds_herlev = CombinedCervicalCancerDataset(
        [], transform=None, config={"data": {}}, classification_mode="multiclass",
        label_space="herlev")
    assert len(ds_sipa.target_names) == 5
    assert len(ds_herlev.target_names) == 7
    assert ds_herlev.class_to_idx["carcinoma_in_situ"] == 6

def test_multiclass_sidecar_matches_canonical():
    """多分类标签空间必须与 canonical 5 类一致（不允许随数据集组合变成 12 类）。"""
    import json, os
    from utils.label_mapping import get_multiclass_classes
    path = "outputs/models/best_resnet50_multiclass_sipakmed_model.classes.json"
    if not os.path.isfile(path):
        pytest.skip("权重尚未训练")
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    assert data["label_space"] == "sipakmed"
    assert data["classes"] == get_multiclass_classes("sipakmed")

def test_cbam_spatial_attention_is_active():
    """CBAM 激活时，SpatialAttention 必须被调用（否则退化为 SE，无法捕捉空间信息）。"""
    import torch
    from models.cer_cancer_model import CervicalCancerModel

    model = CervicalCancerModel(num_classes=5, pretrained=False,
                                model_type='resnet50',
                                use_attention=True, attention_type='cbam').eval()
    called = {"sa": 0}
    model.attention.sa.register_forward_hook(lambda *a: called.__setitem__("sa", called["sa"] + 1))

    with torch.no_grad():
        out = model(torch.randn(2, 3, 224, 224))
    assert out.shape == (2, 5)
    assert called["sa"] == 1, "SpatialAttention 未被调用 → CBAM 退化为 SE"


def test_cbam_adds_no_duplicate_state_dict_keys():
    """CBAM 添加的 state_dict 键不能重复"""
    from models.cer_cancer_model import CervicalCancerModel

    m = CervicalCancerModel(num_classes=5, pretrained=False, model_type='resnet50',
                            use_attention=True, attention_type='cbam')
    keys = list(m.state_dict().keys())
    assert len(keys) == len(set(keys)), "state_dict 存在重复键"
    assert not any(k.startswith('feature_extractor.') for k in keys)


def test_se_path_output_shape_unchanged():
    """SE 模块的输出形状必须与输入相同。"""
    import torch
    from models.cer_cancer_model import CervicalCancerModel

    m = CervicalCancerModel(num_classes=2, pretrained=False, model_type='resnet50',
                            use_attention=True, attention_type='se').eval()
    with torch.no_grad():
        f = m.forward_features(torch.randn(1, 3, 224, 224))
    assert f.dim() == 2 and f.shape[0] == 1

# --- 会话序列化往返 ---
def test_serialize_roundtrip():
    state = {"risk_level": "low", "conversation_history": [HumanMessage(content="hi")]}
    back = _deserialize_state(_serialize_state(state))
    assert back["risk_level"] == "low"
    assert back["conversation_history"][0].content == "hi"

# --- 会话归属校验 ---
@pytest.mark.asyncio
async def test_session_owner_mismatch_is_denied(monkeypatch):
    from Agents.conversation import ConversationManager, OWNER_KEY_PREFIX

    cm = ConversationManager()

    class _FakeRedis:
        def __init__(self, store): self.store = store
        async def get(self, k): return self.store.get(k)
        async def setex(self, k, ttl, v): self.store[k] = str(v)

    store = {f"{OWNER_KEY_PREFIX}sid-1": "doctor-A"}
    fake = _FakeRedis(store)

    async def _get_client(): return fake
    monkeypatch.setattr("Agents.conversation.get_redis_client", _get_client)

    # 本人可访问（此处仅校验归属这一步）
    await cm._assert_owner("sid-1", "doctor-A")

    # 他人被拒
    with pytest.raises(PermissionError):
        await cm._assert_owner("sid-1", "doctor-B")

    # 无归属记录的历史会话：fail-closed
    with pytest.raises(PermissionError):
        await cm._assert_owner("sid-legacy", "doctor-A")

    # 缺少身份
    with pytest.raises(PermissionError):
        await cm._assert_owner("sid-1", None)

# --- 不确定性度量：跨类别数可比 ---
def test_normalized_entropy_is_comparable_across_k():
    from utils.uncertainty import normalized_entropy

    # 完全确定 → 0
    assert normalized_entropy([1.0, 0.0]) == pytest.approx(0.0, abs=1e-9)
    assert normalized_entropy([1.0, 0, 0, 0, 0]) == pytest.approx(0.0, abs=1e-9)

    # 完全不确定 → 1（2 类与 5 类都归一化到同一上界，这正是旧 max(std) 做不到的）
    assert normalized_entropy([0.5, 0.5]) == pytest.approx(1.0, abs=1e-9)
    assert normalized_entropy([0.2] * 5) == pytest.approx(1.0, abs=1e-9)

    # 同样是"top1 概率 0.9"，H 对 K 单调，但都落在 (0,1) 内且量级接近
    e2 = normalized_entropy([0.9, 0.1])
    e5 = normalized_entropy([0.9, 0.025, 0.025, 0.025, 0.025])
    assert 0 < e2 < 1 and 0 < e5 < 1


def test_uncertainty_thresholds_single_source():
    import utils.uncertainty as u
    from Agents import diagnosis_graph as dg

    # 两处必须指向同一对阈值，禁止再次分叉
    assert dg.HIGH_UNCERTAINTY == u.UNCERTAINTY_HIGH
    assert dg.MEDIUM_UNCERTAINTY == u.UNCERTAINTY_MEDIUM
    assert u.UNCERTAINTY_HIGH > u.UNCERTAINTY_MEDIUM

    # --- MC Dropout 线程安全 ---
def test_only_head_has_dropout():
    """mc_dropout_predict 只重采样分类头的前提：模型里除 self.dropout 外无其它 Dropout。"""
    import torch
    from models.cer_cancer_model import CervicalCancerModel

    model = CervicalCancerModel(num_classes=5, pretrained=False)
    names = [n for n, m in model.named_modules()
             if isinstance(m, torch.nn.modules.dropout._DropoutNd)]
    assert names == ["dropout"], f"发现意外 Dropout 层: {names}（需改回全网络 MC 采样）"


def test_mc_dropout_does_not_mutate_model_state():
    """回归:MC Dropout 不得改写共享模型的 training 状态（曾导致并发竞态）。"""
    import torch
    from models.cer_cancer_model import CervicalCancerModel
    vs = pytest.importorskip("services.vision_service")

    model = CervicalCancerModel(num_classes=2, pretrained=False).eval()
    x = torch.randn(1, 3, 224, 224)
    vs.mc_dropout_predict(model, x, n_samples=3)
    assert model.training is False, "MC Dropout 修改了模型 training 状态"

def test_dedup_removes_nested_nucleus_box():
    """回归:胞核小框不应被计为独立细胞。"""
    from services.vision_service import dedup_boxes
    cell = [10, 10, 210, 210]     # 整细胞
    nuc = [90, 90, 130, 130]      # 胞核（完全落在细胞内）
    kept = dedup_boxes([(cell, 0.9), (nuc, 0.8)])
    assert kept == [(cell, 0.9)]


def test_dedup_removes_duplicate_same_cell_box():
    """回归:两个几乎相同的整图框只保留一个。"""
    from services.vision_service import dedup_boxes
    a = [10, 10, 200, 200]
    b = [12, 12, 202, 202]
    kept = dedup_boxes([(a, 0.9), (b, 0.85)])
    assert len(kept) == 1


def test_dedup_keeps_distinct_cells():
    """相邻但不重叠的细胞必须都保留。"""
    from services.vision_service import dedup_boxes
    left = [10, 10, 100, 100]
    right = [200, 200, 300, 300]
    kept = dedup_boxes([(left, 0.9), (right, 0.88)])
    assert len(kept) == 2


def test_mc_dropout_head_is_stochastic():
    """dropout 确实在采样：std 不应恒为 0。"""
    import numpy as np
    import torch
    from models.cer_cancer_model import CervicalCancerModel
    vs = pytest.importorskip("services.vision_service")

    model = CervicalCancerModel(num_classes=2, pretrained=False).eval()
    x = torch.randn(1, 3, 224, 224)
    _, std = vs.mc_dropout_predict(model, x, n_samples=5)
    assert np.max(std) > 0

    # --- 风险分级单一来源 ---
def test_risk_level_from_ratio_boundaries():
    from utils.uncertainty import risk_level_from_ratio
    assert risk_level_from_ratio(0.0) == "low"
    assert risk_level_from_ratio(0.1) == "medium"
    assert risk_level_from_ratio(0.29) == "medium"
    assert risk_level_from_ratio(0.3) == "high"
    assert risk_level_from_ratio(0.59) == "high"
    assert risk_level_from_ratio(0.6) == "emergency"
    assert risk_level_from_ratio(1.0) == "emergency"


def test_agent_risk_uses_shared_judgement():
    """Agent 状态机的 risk_level 必须与共享判据一致（防止再次内联分叉）。"""
    from utils.uncertainty import risk_level_from_ratio
    # 低不确定性 + 无异常 → low；低不确定性 + 全异常 → emergency
    for abnormal, total, expect in ((0, 10, "low"), (3, 10, "medium"),
                                    (5, 10, "high"), (9, 10, "emergency")):
        out = uncertainty_assessment_agent(
            {"cv_result": {"max_uncertainty": 0.01,
                           "abnormal_cells": abnormal, "total_cells": total}}
        )
        assert out["risk_level"] == expect == risk_level_from_ratio(abnormal / total)

# --- 不可判读拦截（0 细胞不得判低风险）---
def test_zero_cells_is_not_low_risk():
    """回归：未检出细胞时必须不可判读，而非 low（曾用 max(total,1) 兜底导致假阴性）。"""
    out = uncertainty_assessment_agent({
        "cv_result": {"total_cells": 0, "abnormal_cells": 0, "max_uncertainty": 0.01}
    })
    assert out["risk_level"] == "pending"
    assert out["action"] == "clarify"
    assert out["risk_level"] != "low"
    assert out.get("error_message"), "应给出不可判读原因"


# --- 风险口径 ---
def test_risk_scope_is_cell_for_single_cell():
    out = uncertainty_assessment_agent({
        "cv_result": {"total_cells": 1, "abnormal_cells": 1, "max_uncertainty": 0.01}
    })
    assert out["risk_scope"] == "cell"
    assert out["risk_level"] == "emergency"     # 单细胞仍给强提示，但口径是 cell


def test_risk_scope_is_case_for_many_cells():
    out = uncertainty_assessment_agent({
        "cv_result": {"total_cells": 10, "abnormal_cells": 5, "max_uncertainty": 0.01}
    })
    assert out["risk_scope"] == "case"
    assert out["risk_level"] == "high"


def test_risk_scope_threshold_makes_all_levels_reachable():
    """CASE_RISK_MIN_CELLS 必须让 low/medium/high/emergency 在数学上可达。"""
    from utils.uncertainty import CASE_RISK_MIN_CELLS, risk_level_from_ratio
    assert CASE_RISK_MIN_CELLS == 4
    levels = {risk_level_from_ratio(k / CASE_RISK_MIN_CELLS)
              for k in range(CASE_RISK_MIN_CELLS + 1)}
    assert {"low", "medium", "high", "emergency"} <= levels