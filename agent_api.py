from fastapi import FastAPI, UploadFile, File, WebSocket, WebSocketDisconnect, Depends, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from pydantic import BaseModel as PydanticBaseModel
from typing import Optional
import uuid
import json
import os

from Agents.conversation import ConversationManager
from services.vision_service import analyze_image_from_path

class FollowUpRequest(PydanticBaseModel):
    session_id: str
    message: str

# 1. 引入连接管理器 (防止 WebSocket 内存泄漏)
class ConnectionManager:
    def __init__(self):
        self.active_connections: dict[str, WebSocket] = {}

    async def connect(self, session_id: str, websocket: WebSocket):
        await websocket.accept()
        self.active_connections[session_id] = websocket

    def disconnect(self, session_id: str):
        if session_id in self.active_connections:
            del self.active_connections[session_id]

    async def send_progress(self, session_id: str, message: dict):
        if session_id in self.active_connections:
            await self.active_connections[session_id].send_json(message)


manager = ConnectionManager()


# 2. 依赖注入 ConversationManager
def get_conversation_manager():
    # 连接 Redis/DB
    from Agents.conversation import ConversationManager
    return ConversationManager()


app = FastAPI(title="AI Agent宫颈病理诊断系统 v2.0", version="2.0")

# 3. 收紧 CORS 配置
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "https://your-hospital-domain.com"],  # 仅允许白名单
    allow_methods=["*"],
    allow_headers=["*"]
)

# --- 新增：FastAPI 路由 ---
@app.post("/api/agent/followup")
async def agent_followup(
    request: FollowUpRequest,
    conv_manager: ConversationManager = Depends(get_conversation_manager)
):
    """
    处理诊断完成后的用户追问。
    注入 <time_location> 上下文以提供基于当前时间（2026-06-27）和地点（温州）的回答。
    """
    try:
        # 调用 ConversationManager 处理追问，并传入固定的时空上下文
        # 注意：这里将 time_location 作为参数传入，需要同步修改 conversation.py 中的方法签名
        response = conv_manager.followup_chat(
            session_id=request.session_id,
            user_input=request.message,
            time_location="<time_location>\n当前时间：2026-06-27 星期六\n当前地点：浙江省 温州市\n</time_location>"
        )
        return {
            "success": True,
            "response": response
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# 4. 优化后的上传接口 (建议后续升级为 Celery 异步任务)
@app.post("/api/agent/upload")
async def agent_upload(
        file: UploadFile = File(...),
        conv_manager=Depends(get_conversation_manager)
):
    # 文件安全校验
    if not file.filename.endswith(('.svs', '.jpg', '.png')):
        raise HTTPException(status_code=400, detail="不支持的文件格式")

    session_id = str(uuid.uuid4())
    file_path = f"./uploads/{session_id}_{file.filename}"

    # 确保目录存在并保存文件
    os.makedirs("./uploads", exist_ok=True)
    with open(file_path, "wb") as f:
        content = await file.read()
        f.write(content)

    # TODO: 生产环境强烈建议将 analyze_image 和 start_diagnosis 放入 Celery 队列
    # 这里暂时保留同步调用，但应确保 analyze_image 内部不阻塞主线程
    try:
        # from services.vision_service import analyze_image
        # cv_results = await analyze_image(file_path)
        # result = conv_manager.start_diagnosis(session_id, file_path, cv_results)
        result = conv_manager.start_diagnosis(file_path, session_id=session_id)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"诊断启动失败: {str(e)}")

    return {
        "session_id": session_id,
        "status": "complete" if result["is_complete"] else "awaiting_clarification",
        "report": result.get("diagnosis_report"),
        "questions": result.get("clarification_questions"),
    }


# 5. 健壮的 WebSocket 端点
@app.websocket("/ws/agent/chat/{session_id}")
async def agent_websocket(websocket: WebSocket, session_id: str):
    await manager.connect(session_id, websocket)
    try:
        while True:
            message = await websocket.receive_text()
            data = json.loads(message)
            user_input = data.get("message", "")

            # 实际生产中，这里应通过 Redis Pub/Sub 获取后台 Agent 的流式进度
            # 这里仅作逻辑演示
            await manager.send_progress(session_id, {"type": "stream", "content": "正在分析您的补充信息..."})

    except WebSocketDisconnect:
        manager.disconnect(session_id)
    except Exception as e:
        await manager.send_progress(session_id, {"type": "error", "message": str(e)})
        manager.disconnect(session_id)