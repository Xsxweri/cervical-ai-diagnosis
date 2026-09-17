import uuid
import logging
from typing import Optional, Union, Any
from datetime import timedelta
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage, message_to_dict
from langgraph.types import Command
from Agents.diagnosis_graph import get_graph, get_llm  # 导入图实例和LLM工厂函数
from Agents.state import DiagnosisState
from services.vision_service import analyze_image_from_path  # 导入CV服务
from Prompts import render, raw
from Prompts.clinical_content import FOLLOWUP_FALLBACK
import json
from core.redis_client import get_redis_client

# 配置日志
logger = logging.getLogger(__name__)

SESSION_KEY_PREFIX = "session:"
_MESSAGE_KEYS = ("messages", "conversation_history")

def _serialize_state(state: Union[dict,Any]) -> str:
    """将 DiagnosisState 序列化为 JSON 字符串，特殊处理 BaseMessage 列表"""
    data = dict(state)
    data.pop("__interrupt__", None)
    for key in _MESSAGE_KEYS:
        msgs = data.get(key) or []
        data[key] = [message_to_dict(m) for m in msgs if isinstance(m, BaseMessage)]
    return json.dumps(data, ensure_ascii=False, default=str)

def _deserialize_state(raw: str) -> dict:
    """从 JSON 字符串反序列化为 Dict"""
    data = json.loads(raw)
    for key in _MESSAGE_KEYS:
        if data.get(key):
            data[key] = [_msg_from_dict(d) for d in data[key]]
    return data

def _msg_from_dict(d: dict) -> BaseMessage:
    """根据 type 字段还原具体的 Message 类型"""
    msg_type = d.get("type", "")
    content = d.get("data", {}).get("content", "")
    if msg_type == "human":
        return HumanMessage(content=content)
    elif msg_type == "ai":
        return AIMessage(content=content)
    elif msg_type == "system":
        return SystemMessage(content=content)
    return HumanMessage(content=content)

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

    async def _save_session(self, session_id: str, state: Union[dict,Any]):
        """序列化状态并写入 Redis，同时设置 TTL"""
        key = f"{SESSION_KEY_PREFIX}{session_id}"
        redis_client = await get_redis_client()
        await redis_client.setex(key, int(self.session_ttl.total_seconds()), _serialize_state(state))

    async def _load_session(self, session_id: str) -> Optional[dict]:
        """从Redis读取并反序列化状态"""
        key = f"{SESSION_KEY_PREFIX}{session_id}"
        redis_client = await get_redis_client()
        raw = await redis_client.get(key)
        if raw is None:
            return None
        try:
            return _deserialize_state(raw)
        except Exception as e:
            logger.error(f"反序列化会话 [{session_id}] 失败: {e}")
            return None

    @staticmethod
    def _apply_interrupt(state: dict) -> dict:
        """
        从图返回值中提取 interrupt 载荷，转换为前端可消费的字段
        """
        interrupted = state.pop("__interrupt__", None)
        if interrupted:
            first = interrupted[0] if isinstance(interrupted, (list, tuple)) else interrupted
            payload = getattr(first, "value", None) or {}
            state["clarification_questions"] = payload.get("questions", [])
            state["needs_clarification"] = True
            state["is_complete"] = False
        return state

    async def start_diagnosis(self, image_path: str, session_id:  Optional[str] = None) -> dict :
        """
        启动新诊断流程：执行 CV 分析 -> 初始化状态 -> 触发 Agent 图

        Args:
            image_path: 上传的图片路径
            session_id: 可选会话ID

        Returns:
            更新后的状态对象
        """
        # 1. 生成会话ID
        if not session_id:
            session_id = str(uuid.uuid4())

        # 2. 调用视觉服务进行细胞分析 (这会触发 YOLO + ResNet50 双模型推理)
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
            "is_complete": False
        }

        # 4. 运行 Agent 图
        # 必须传入config以支持LangGraph的Checkpointer
        config = {"configurable": {"thread_id": session_id}}

        try:
            final_state = self._apply_interrupt(await get_graph().ainvoke(initial_state, config))
            await self._save_session(session_id, final_state)
            logger.info(f"会话 [{session_id}] 初始诊断完成，风险等级: {final_state.get('risk_level')}")
            return final_state
        except Exception as e:
            logger.error(f"会话 [{session_id}] Agent流程执行失败: {e}", exc_info=True)
            raise

    async def process_clarification(self, session_id: str, user_input: str) -> dict:
        """
        处理用户对澄清问题的回答。

        逻辑：
        1. 校验会话确实处于等待澄清状态
        2. 用 Command(resume=...) 恢复被 interrupt 暂停的图
        3. 用户回答由 clarification 节点写入 conversation_history，此处不重复追加
        """
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

    async def followup_chat(self, session_id: str, user_input: str, time_location: str = "") -> str:
        """
        诊断完成后的自由追问。
        此方法不修改核心诊断状态，仅基于已有信息进行问答。

        Args:
            session_id: 会话ID
            user_input: 用户的追问
            time_location: 时空上下文字符串 (可选)
        Returns:
            AI的回复文本
        """
        state = await self._load_session(session_id)
        if not state:
            raise ValueError("会话不存在，请重新上传图片开始新的诊断。")

        if not state.get("is_complete"):
            return "诊断流程尚未最终完成，请先回答上面的问题。"

        # 1. 获取 LLM 实例
        llm = get_llm()

        # 2. 构建带有时空上下文的 Prompt
        # 如果传入了 time_location，则使用传入的值，否则使用默认值（兼容旧调用）
        ctx_time_location = time_location or f"<time_location>\n当前时间：未知\n当前地点：未知\n</time_location>"

        # 提取诊断报告摘要（防止上下文过长）
        report_snippet = (state.get("diagnosis_report") or "")[:800]
        recommendations = "\n".join([f"- {r}" for r in (state.get("recommendations") or [])])
        risk_level = state.get("risk_level") or "未知"

        # 构建最终的 Prompt
        # 这里将 time_location 插入到了Prompt的最开始部分
        followup_prompt = render(
            "followup_user",
            time_location=ctx_time_location,
            risk_level=risk_level,
            report_snippet=report_snippet,
            recommendations=recommendations,
            user_input=user_input,
        )

        # 3. 调用LLM并更新历史
        history = list(state.get("conversation_history") or [])
        history.append(HumanMessage(content=user_input))

        messages = [
            SystemMessage(content=raw("followup_system")),
            *history,
            HumanMessage(content=followup_prompt)
        ]

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
        key = f"{SESSION_KEY_PREFIX}{session_id}"
        redis_client = await get_redis_client()
        result = await redis_client.delete(key)
        return result > 0