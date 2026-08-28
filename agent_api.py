from fastapi import FastAPI, UploadFile, File, WebSocket, WebSocketDisconnect, Depends, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel as PydanticBaseModel
from typing import List
import uuid
import json
import os

import io
import zipfile
from PIL import Image
from werkzeug.utils import secure_filename
from functools import lru_cache
from core.auth import verify_token
from contextlib import asynccontextmanager
from Agents.conversation import ConversationManager
from services.vision_service import (
    analyze_pil_image,          # 核心分析函数
    DEVICE, YOLO_WEIGHTS,       # 配置常量
    BINARY_WEIGHTS, MULTI_WEIGHTS,
    MC_DROPOUT_SAMPLES,
    UNCERTAINTY_THRESHOLD,
    ABNORMAL_CONF_THRESHOLD,
)

class FollowUpRequest(PydanticBaseModel):
    session_id: str
    message: str

# 引入连接管理器 (防止 WebSocket 内存泄漏)
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

# 定义文件头字典
MAGIC_BYTES:dict[str,list[bytes]] = {
    'jpg': [b'\xff\xd8\xff'],
    'jpeg': [b'\xff\xd8\xff'],
    'png': [b'\x89PNG\r\n\x1a\n'],
    'tif': [b'II*\x00', b'MM\x00*'],
    'tiff': [b'II*\x00', b'MM\x00*'],
    'bmp': [b'BM'],
    'svs': [b'II*\x00', b'MM\x00*'], # SVS文件通常是TIFF格式的变体
}

# 定义允许的 MIME 类型
ALLOWED_MIME = {
    '.jpg': 'image/jpeg',
    '.jpeg': 'image/jpeg',
    '.png': 'image/png',
    '.tif': 'image/tiff',
    '.tiff': 'image/tiff',
    '.bmp': 'image/bmp',
    '.svs': {'image/tiff', 'image/svs', 'application/octet-stream'}
}

# 文件安全校验函数
def validate_file_content(content: bytes, filename: str, content_type: str):
    ext = os.path.splitext(filename)[1].lower()

    if content_type and ALLOWED_MIME.get(ext) and content_type != ALLOWED_MIME[ext]:
        raise HTTPException(status_code=400, detail="Content-Type 与文件类型不符")

    sigs = MAGIC_BYTES.get(ext)
    if sigs and not any(content.startswith(s) for s in sigs):
        raise HTTPException(status_code=400, detail=f"文件内容与扩展名不匹配: {filename}")
    if ext in {'.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp'}:
        try:
            with Image.open(io.BytesIO(content)) as img:
                img.verify()  # 验证图像完整性
        except Exception:
            raise HTTPException(status_code=400, detail=f"文件不是有效图片: {filename}")

# 依赖注入 ConversationManager
@lru_cache()
def get_conversation_manager():
    return ConversationManager()

@asynccontextmanager
async def lifespan(app: FastAPI):
    from Agents.diagnosis_graph import init_checkpoint_store, close_checkpoint_store, init_kb
    await init_kb()
    await init_checkpoint_store()
    yield
    await close_checkpoint_store()

app = FastAPI(title="AI Agent宫颈病理诊断系统", version="2.0", lifespan=lifespan)

# 3. 收紧 CORS 配置
app.add_middleware(
    CORSMiddleware,
    allow_origins= os.getenv("CORS_ORIGINS", "http://localhost:3000").split(","),  # 仅允许白名单
    allow_methods=["*"],
    allow_headers=["*"]
)

# === FastAPI 接口 ===
@app.post("/analyze", summary="单张图片双模型分析")
async def analyze_image(file: UploadFile = File(...)):
    try:
        data = await file.read()
        image = Image.open(io.BytesIO(data)).convert("RGB")
        return await analyze_pil_image(image)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"图像处理失败: {e}")


@app.post("/analyze_batch", summary="多张图片批量分析")
async def analyze_batch(files: List[UploadFile] = File(...)):
    results = {}
    for file in files:
        try:
            data = await file.read()
            image = Image.open(io.BytesIO(data)).convert("RGB")
            results[file.filename] = await analyze_pil_image(image)
        except Exception as e:
            results[file.filename] = {"出错": str(e), "总细胞数": 0, "异常细胞数": 0, "细胞类": []}
    return results


@app.post("/analyze_zip", summary="ZIP压缩包批量分析")
async def analyze_zip(upload_file: UploadFile = File(...)):
    if not upload_file.filename.endswith(".zip"):
        raise HTTPException(status_code=400, detail="仅接受.zip文件")
    results = {}
    try:
        zip_bytes = await upload_file.read()
        with zipfile.ZipFile(io.BytesIO(zip_bytes)) as z:
            valid_ext = ('.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp')
            filenames = [f for f in z.namelist() if f.lower().endswith(valid_ext)]
            if not filenames:
                raise HTTPException(status_code=400, detail="ZIP中找不到有效的图片")
            for filename in filenames:
                try:
                    with z.open(filename) as img_file:
                        image = Image.open(io.BytesIO(img_file.read())).convert("RGB")
                    results[filename] = await analyze_pil_image(image)
                except Exception as e:
                    results[filename] = {"错误": str(e), "总细胞数": 0, "异常细胞数": 0, "细胞类": []}
    except zipfile.BadZipFile:
        raise HTTPException(status_code=400, detail="无效的ZIP文件")
    return results

@app.get("/health", summary="健康检查")
async def health_check():
    return {
        "status": "ok",
        "device": str(DEVICE),
        "models": {
            "yolo": YOLO_WEIGHTS,
            "二分类": BINARY_WEIGHTS,
            "多分类": MULTI_WEIGHTS
        },
        "config": {
            "mc_dropout_samples": MC_DROPOUT_SAMPLES,
            "不确定阈值": UNCERTAINTY_THRESHOLD,
            "异常的CONF阈值": ABNORMAL_CONF_THRESHOLD
        }
    }

# --- FastAPI 路由 ---
@app.post("/api/agent/followup",dependencies=[Depends(verify_token)])
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
        response = await conv_manager.followup_chat(
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

@app.post("/api/agent/clarify/{session_id}",dependencies=[Depends(verify_token)])
async def agent_clarify(session_id: str, body: dict = Body(...), conv_manager: ConversationManager = Depends(get_conversation_manager)):
    """
    处理用户对诊断报告的澄清请求。
    """
    user_input = body.get("user_input", "")
    if not user_input:
        raise HTTPException(status_code=400, detail="缺少 user_input 字段")
    result = await conv_manager.process_clarification(session_id, user_input)
    return {
        "session_id": session_id,
        "status": "complete" if result["is_complete"] else "awaiting_clarification",
        "report": result.get("diagnosis_report"),
        "questions": result.get("clarification_questions"),
    }

# 4. 优化后的上传接口
@app.post("/api/agent/upload", dependencies=[Depends(verify_token)])
async def agent_upload(
        file: UploadFile = File(...),
        conv_manager=Depends(get_conversation_manager)
):
    allowed_exits = {'.svs', '.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp'}
    max_size = 30 * 1024 * 1024
    safe_name = secure_filename(file.filename or "")
    # 文件安全校验
    if not safe_name or not safe_name.lower().endswith(tuple(allowed_exits)):
        raise HTTPException(status_code=400, detail="不支持的文件格式")

    session_id = str(uuid.uuid4())
    upload_dir = os.path.abspath("./uploads")
    os.makedirs(upload_dir, exist_ok=True)
    file_path = os.path.join(upload_dir, f"{session_id}_{safe_name}")

    if not file_path.startswith(upload_dir):
        raise HTTPException(status_code=400, detail="非法文件路径")

    content = await file.read()
    if len(content) > max_size:
        raise HTTPException(status_code=413, detail="文件超过30MB限制")

    validate_file_content(content, safe_name, file.content_type)
    with open(file_path, "wb") as f:
        f.write(content)

    # 生产环境强烈建议将 analyze_image 和 start_diagnosis 放入 Celery 队列
    try:
        result = await conv_manager.start_diagnosis(file_path, session_id=session_id)
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
            if not user_input:
                await manager.send_progress(session_id, {"type": "error", "message": "消息不能为空"})
                continue
            await manager.send_progress(session_id, {"type": "stream", "content": "正在分析您的补充信息..."})

    except WebSocketDisconnect:
        manager.disconnect(session_id)
    except Exception as e:
        await manager.send_progress(session_id, {"type": "error", "message": str(e)})
        manager.disconnect(session_id)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("agent_api:app", host="0.0.0.0", port=8000, reload=True)