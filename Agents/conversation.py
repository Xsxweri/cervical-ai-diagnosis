"""
会话编排层：会话生命周期 + CV/Agent 图/追问流程协调。

分工：
- 序列化与 Redis 读写 -> Agents.session_store
- 追问 Prompt 拼装     -> Agents.followup_prompt
"""

import logging
import uuid
from datetime import timedelta
from typing import Optional

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.types import Command

from Agents.checkpoint import get_graph
from Agents.followup_prompt import build_followup_prompt, build_followup_system
from Agents.llm import get_llm
from Agents.session_store import (
    OWNER_KEY_PREFIX,  # noqa: F401  (re-export, tests 需要)
    SESSION_KEY_PREFIX,  # noqa: F401  (re-export)
    assert_owner,
    clear_session as _store_clear_session,
    deserialize_state,
    load_session,
    save_owner,
    save_session,
    serialize_state,
)
from Agents.state import DiagnosisState
from Prompts.clinical_content import FOLLOWUP_FALLBACK
from core.redis_client import get_redis_client
from services.vision_service import analyze_image_from_path  # 导入CV服务

logger = logging.getLogger(__name__)

# 兼容旧引用：tests/test_unit.py 直接 import 这两个私有名
_serialize_state = serialize_state
_deserialize_state = deserialize_state


class ConversationManager:
    """
    宫颈癌AI诊断多轮对话管理器

    功能：
    1. 管理会话生命周期（创建、恢复、清除）
    2. 协调 Vision Service (CV) 和 Diagnosis Graph (Agent)
    3. 处理澄清追问后的状态恢复
    4. 提供诊断完成后的追问聊天
    """

    def __init__(self):
        self.session_ttl = timedelta(days=1)  # 会话过期时间

    @property
    def _ttl_seconds(self) -> int:
        return int(self.session_ttl.total_seconds())

    # 存储代理
    async def _save_session(self, session_id: str, state):
        """序列化状态并写入 Redis，同时设置 TTL"""
        await save_session(get_redis_client, session_id, state, self._ttl_seconds)

    async def _load_session(self, session_id: str) -> Optional[dict]:
        """从Redis读取并反序列化状态"""
        return await load_session(get_redis_client, session_id)

    async def _save_owner(self, session_id: str, owner_id: Optional[str]):
        """记录会话归属，TTL 与会话一致；供后续越权校验。"""
        await save_owner(get_redis_client, session_id, owner_id, self._ttl_seconds)

    async def _assert_owner(self, session_id: str, requester_id: Optional[str]):
        """校验会话归属，失败抛 PermissionError（fail-closed）。"""
        await assert_owner(get_redis_client, session_id, requester_id)

    @staticmethod
    def _apply_interrupt(state: dict) -> dict:
        """从图返回值中提取 interrupt 载荷，转换为前端可消费的字段"""
        interrupted = state.pop("__interrupt__", None)
        if interrupted:
            first = interrupted[0] if isinstance(interrupted, (list, tuple)) else interrupted
            payload = getattr(first, "value", None) or {}
            state["clarification_questions"] = payload.get("questions", [])
            state["needs_clarification"] = True
            state["is_complete"] = False
        return state

    # 主流程
    async def start_diagnosis(
        self,
        image_path: str,
        session_id: Optional[str] = None,
        owner_id: Optional[str] = None,
    ) -> dict:
        """启动新诊断流程：执行 CV 分析 -> 初始化状态 -> 触发 Agent 图"""
        # 1. 生成会话ID
        if not session_id:
            session_id = str(uuid.uuid4())

        # 2. 调用视觉服务进行细胞分析 (YOLO + 分类器双模型推理)
        logger.info(f"会话 [{session_id}] 开始CV图像分析: {image_path}")
        try:
            cv_results = await analyze_image_from_path(image_path)
        except Exception as e:
            logger.error(f"CV图像分析失败: {e}")
            raise ValueError(f"图像分析服务异常: {e}")

        # 3. 构建初始状态
        initial_state: DiagnosisState = {
            "image_path": image_path,
            "cv_result": cv_results,
            "max_uncertainty": cv_results.get("max_uncertainty", 0.0),
            "needs_clarification": cv_results.get("needs_clarification", False),
            "conversation_history": [HumanMessage(content="用户上传了宫颈细胞图片，开始AI辅助诊断。")],
            "current_step": "uncertainty_assessment",
            "clarification_rounds": 0,
            "is_complete": False,
        }

        # 4. 运行 Agent 图（必须传 config 以支持 Checkpointer）
        config = {"configurable": {"thread_id": session_id}}

        try:
            final_state = self._apply_interrupt(
                await get_graph().ainvoke(initial_state, config)
            )
            await self._save_session(session_id, final_state)
            await self._save_owner(session_id, owner_id)
            logger.info(f"会话 [{session_id}] 初始诊断完成，风险等级: {final_state.get('risk_level')}")
            return final_state
        except Exception as e:
            logger.error(f"会话 [{session_id}] Agent流程执行失败: {e}", exc_info=True)
            raise

    async def process_clarification(
        self,
        session_id: str,
        user_input: str,
        requester_id: Optional[str] = None,
    ) -> dict:
        """处理用户对澄清问题的回答。

        逻辑：
        1. 校验会话确实处于等待澄清状态
        2. 用 Command(resume=...) 恢复被 interrupt 暂停的图
        3. 用户回答由 clarification 节点写入 conversation_history，此处不重复追加
        """
        await self._assert_owner(session_id, requester_id)
        state = await self._load_session(session_id)
        if not state:
            raise ValueError(f"会话 [{session_id}] 不存在或已过期")
        if not state.get("needs_clarification"):
            raise ValueError(f"会话 [{session_id}] 没有待回答的澄清问题")

        config = {"configurable": {"thread_id": session_id}}
        try:
            result = self._apply_interrupt(
                await get_graph().ainvoke(Command(resume=user_input), config)
            )
            await self._save_session(session_id, result)
            logger.info(f"会话 [{session_id}] 澄清处理完成，新动作: {result.get('action')}")
            return result
        except Exception as e:
            logger.error(f"会话 [{session_id}] 澄清处理失败: {e}", exc_info=True)
            raise

    async def followup_chat(
        self,
        session_id: str,
        user_input: str,
        time_location: str = "",
        requester_id: Optional[str] = None,
    ) -> str:
        """诊断完成后的自由追问。不修改核心诊断状态，仅基于已有信息问答。"""
        await self._assert_owner(session_id, requester_id)
        state = await self._load_session(session_id)
        if not state:
            raise ValueError("会话不存在，请重新上传图片开始新的诊断。")

        if not state.get("is_complete"):
            return "诊断流程尚未最终完成，请先回答上面的问题。"

        if state.get("risk_level") == "unreadable":
            return "该图像未能判读，暂无可讨论的诊断结论，请重新上传合格图像。"

        # 1. 先构造客户端：配置错误应直接抛出，不被兜底吞掉
        llm = get_llm()

        # 2. 构建 Prompt
        followup_prompt = build_followup_prompt(state, user_input, time_location)

        # 3. 拼装消息
        history = list(state.get("conversation_history") or [])
        history.append(HumanMessage(content=user_input))
        messages = [
            SystemMessage(content=build_followup_system()),
            *history,
            HumanMessage(content=followup_prompt),
        ]

        # 4. 调用 LLM 并更新历史
        try:
            response = await llm.ainvoke(messages)
        except Exception as e:
            logger.error(f"追问对话失败: {e}")
            return FOLLOWUP_FALLBACK

        history.append(AIMessage(content=response.content))
        state["conversation_history"] = history
        await self._save_session(session_id, state)
        return response.content

    async def get_session_state(self, session_id: str) -> Optional[dict]:
        """获取会话当前状态（用于前端同步）"""
        return await self._load_session(session_id)

    async def clear_session(self, session_id: str) -> bool:
        """清除会话"""
        return await _store_clear_session(get_redis_client, session_id)