import uuid
import logging
from typing import Dict, Any, Optional, Tuple
from datetime import datetime
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from Agents.diagnosis_graph import graph, get_llm  # 导入图实例和LLM工厂函数
from Agents.state import DiagnosisState
from services.vision_service import analyze_image_from_path  # 导入CV服务

# 配置日志
logger = logging.getLogger(__name__)


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
        # 内存级会话存储，生产环境建议替换为 Redis
        self.sessions: Dict[str, DiagnosisState] = {}

    def start_diagnosis(self, image_path: str, session_id: Optional[str] = None) -> DiagnosisState:
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
            cv_results = analyze_image_from_path(image_path)
        except Exception as e:
            logger.error(f"CV图像分析失败: {e}")
            raise ValueError(f"图像分析服务异常: {e}")

        # 3. 构建初始状态
        # 注意：这里使用cv_result(单数)以匹配state.py的定义
        initial_state = DiagnosisState(
            image_path=image_path,
            user_query=None,
            conversation_history=[HumanMessage(content="用户上传了宫颈细胞图片，开始AI辅助诊断。")],
            cv_result=cv_results,  # 嵌入CV分析结果
            max_uncertainty=cv_results.get("max_uncertainty", 0.0),
            needs_clarification=cv_results.get("needs_clarification", False),
            retrieved_knowledge=None,
            risk_level=None,
            action=None,  # 初始为空，由第一个节点决定
            clarification_questions=None,
            gradcam_path=None,
            morphology_analysis=None,
            diagnosis_report=None,
            recommendations=None,
            references=None,
            current_step="uncertainty_assessment",
            is_complete=False
        )

        # 4. 运行 Agent 图
        # 必须传入config以支持LangGraph的Checkpointer(MemorySaver)
        config = {"configurable": {"thread_id": session_id}}

        try:
            final_state = graph.invoke(initial_state, config)
            self.sessions[session_id] = final_state
            logger.info(f"会话 [{session_id}] 初始诊断完成，风险等级: {final_state.get('risk_level')}")
            return final_state
        except Exception as e:
            logger.error(f"会话 [{session_id}] Agent流程执行失败: {e}", exc_info=True)
            raise

    def process_clarification(self, session_id: str, user_input: str) -> DiagnosisState:
        """
        处理用户对澄清问题的回答。

        逻辑：
        1. 获取当前挂起的状态
        2. 将用户回答注入历史和状态
        3. 重置 Action 为 re_evaluate，触发重新评估流程
        4. 继续运行图

        Args:
            session_id: 会话ID
            user_input: 用户的回答

        Returns:
            更新后的状态
        """
        state = self.sessions.get(session_id)
        if not state:
            raise ValueError(f"会话 [{session_id}] 不存在或已过期")

        # 1. 更新对话历史
        history = state.get("conversation_history", [])
        history.append(HumanMessage(content=user_input))

        # 2. 更新状态：注入用户回答，并重置流程控制
        state_update = {
            "conversation_history": history,
            "user_query": user_input,  # 这个字段会被Agent节点读取
            "action": "re_evaluate", 
            "needs_clarification": False,
            "current_step": "clarification_resolved"
        }

        # 3. 合并状态并继续执行图
        # 注意：这里使用update，以保持其他字段不变
        for key, value in state_update.items():
            state[key] = value

        config = {"configurable": {"thread_id": session_id}}
        try:
            result = graph.invoke(state, config)
            self.sessions[session_id] = result
            logger.info(f"会话 [{session_id}] 澄清处理完成，新动作: {result.get('action')}")
            return result
        except Exception as e:
            logger.error(f"会话 [{session_id}] 澄清处理失败: {e}", exc_info=True)
            raise

    def followup_chat(self, session_id: str, user_input: str, time_location: str = "") -> str:
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
        state = self.sessions.get(session_id)
        if not state:
            raise ValueError("会话不存在，请重新上传图片开始新的诊断。")

        if not state.get("is_complete"):
            return "诊断流程尚未最终完成，请先回答上面的问题。"

        # 1. 获取 LLM 实例
        llm = get_llm()

        # 2. 构建带有时空上下文的 Prompt
        # 如果传入了 time_location，则使用传入的值，否则使用默认值（为了兼容旧调用）
        ctx_time_location = time_location or f"<time_location>\n当前时间：未知\n当前地点：未知\n</time_location>"

        # 提取诊断报告摘要（防止上下文过长）
        report_snippet = state.get("diagnosis_report", "")[:800]
        recommendations = "\n".join([f"- {r}" for r in state.get("recommendations", [])])
        risk_level = state.get("risk_level", "未知")

        # 构建最终的 Prompt
        # 注意：这里将 time_location 插入到了Prompt的最开始部分
        followup_prompt = f'''{ctx_time_location}

        【角色】你是一位经验丰富的病理科医生助手。
        【任务】请基于已有的诊断报告回答用户的追问。不要改变原有的诊断结论。
        【已有诊断摘要】
        风险等级: {risk_level}
        报告内容: {report_snippet}...
        推荐措施: {recommendations}
    
        【用户追问】
        {user_input}
    
        【回答要求】
        1. 结合当前时间和地点给出人性化的解释。
        2. 如果用户问及具体的医学建议（如“什么时候复查”），请结合报告中的建议回答。
        3. 如果问题超出了报告范围，请建议用户咨询线下专业医生。
        4. 保持语气专业且温和。'''

        # 3. 调用LLM并更新历史
        try:
            # 将追问加入历史
            state["conversation_history"].append(HumanMessage(content=user_input))

            # 构建消息列表
            messages = [
                SystemMessage(content="你正在协助医生进行宫颈细胞学诊断报告的解读。"),
                *state["conversation_history"],
                HumanMessage(content=followup_prompt)
            ]

            response = llm.invoke(messages)

            # 创建 AI 消息并加入历史
            ai_msg = AIMessage(content=response.content)
            state["conversation_history"].append(ai_msg)

            return response.content

        except Exception as e:
            logger.error(f"追问对话失败: {e}")
            return "抱歉，AI医生暂时无法回答这个问题，请稍后再试。"

    def get_session_state(self, session_id: str) -> Optional[DiagnosisState]:
        """获取会话当前状态（用于前端同步）"""
        return self.sessions.get(session_id)

    def clear_session(self, session_id: str) -> bool:
        """清除会话"""
        if session_id in self.sessions:
            del self.sessions[session_id]
            return True
        return False