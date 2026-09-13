from fastapi import FastAPI, UploadFile, File, WebSocket, WebSocketDisconnect, Depends, HTTPException, Body, Request, Query
from fastapi.middleware.cors import CORSMiddleware
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address
from pydantic import BaseModel as PydanticBaseModel
from typing import List
import uuid
import json
import os

import io
import zipfile
from datetime import datetime
from zoneinfo import ZoneInfo
from PIL import Image
from werkzeug.utils import secure_filename
from functools import lru_cache
from core.auth import verify_token, create_token, decode_token
from werkzeug.security import check_password_hash
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

class LoginRequest(PydanticBaseModel):
    username: str
    password: str

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

#限流器
limiter = Limiter(key_func=get_remote_address)

ACCOUNTS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "appWeb", 'doctor_accounts.json')
SERVICE_LOCATION = os.getenv("SERVICE_LOCATION", "未知地点")

# 定义文件头字典
MAGIC_BYTES:dict[str,list[bytes]] = {
    '.jpg': [b'\xff\xd8\xff'],
    '.jpeg': [b'\xff\xd8\xff'],
    '.png': [b'\x89PNG\r\n\x1a\n'],
    '.tif': [b'II*\x00', b'MM\x00*'],
    '.tiff': [b'II*\x00', b'MM\x00*'],
    '.bmp': [b'BM'],
    '.svs': [b'II*\x00', b'MM\x00*'], # SVS文件通常是TIFF格式的变体
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

    allowed = ALLOWED_MIME.get(ext)
    if allowed:
        allowed = allowed if isinstance(allowed, (set,frozenset,tuple,list)) else {allowed}
        if content_type and content_type not in allowed:
            raise HTTPException(status_code=400, detail="Content-Type 与文件类型不符")

    sig = MAGIC_BYTES.get(ext)
    if sig and not any(content.startswith(s) for s in sig):
        raise HTTPException(status_code=400, detail=f"文件内容与扩展名不匹配: {filename}")
    if ext in {'.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp'}:
        try:
            with Image.open(io.BytesIO(content)) as img:
                img.verify()  # 验证图像完整性
        except Exception:
            raise HTTPException(status_code=400, detail=f"文件不是有效图片: {filename}")

def build_time_location() -> str:
    tz = ZoneInfo(os.getenv("SERVICE_TIMEZONE", "Asia/Shanghai"))
    now = datetime.now(tz)
    weekday = "星期" + "一二三四五六日"[now.weekday()]
    return (
        f"<time_location>\n"
        f"当前时间：{now.strftime('%Y-%m-%d')} {weekday}\n"
        f"当前地点：{SERVICE_LOCATION}\n"
        f"</time_location>"
    )

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
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# 3. 收紧 CORS 配置
app.add_middleware(
    CORSMiddleware,
    allow_origins= os.getenv("CORS_ORIGINS", "http://localhost:3000").split(","),  # 仅允许白名单
    allow_methods=["*"],
    allow_headers=["*"]
)

# === FastAPI 接口 ===
# 登录接口
@app.post("/api/auth/login", summary="医生登录")
async def login(req: LoginRequest):
    try:
        with open(ACCOUNTS_PATH, 'r', encoding='utf-8') as f:
            accounts = json.load(f)
    except (OSError, json.JSONDecodeError):
        raise HTTPException(status_code=500, detail="账号数据读取失败")
    doctor = next((d for d in accounts if d.get("username") == req.username), None)
    if not doctor or not check_password_hash(doctor.get("password_hash", ""), req.password):
        raise HTTPException(status_code=401, detail="用户名或密码错误")
    token = create_token({"sub": doctor["username"], "uid": doctor.get("id")})
    return{
        "token": token,
        "doctor": {
            "name": doctor.get("name"),
            "department": doctor.get("department"),
            "hospital": doctor.get("hospital")
        },
    }

# 批量/压缩分析的资源上限常量
MAX_BATCH_FILES = 10
MAX_ZIP_FILES = 20
MAX_SINGLE_FILE = 30 * 1024 * 1024  # 单文件最大30MB
MAX_ZIP_UNCOMPRESSED = 200 * 1024 * 1024 # 解压后最大200MB

@app.post("/analyze", summary="单张图片双模型分析", dependencies=[Depends(verify_token)])
@limiter.limit("10/minute")  # 限流：每分钟最多10次请求
async def analyze_image(request: Request,file: UploadFile = File(...)):
    try:
        data = await file.read()
        if len(data) > MAX_SINGLE_FILE:
            raise HTTPException(status_code=413, detail="文件超过30MB限制")
        validate_file_content(data, file.filename or "", file.content_type)
        image = Image.open(io.BytesIO(data)).convert("RGB")
        return await analyze_pil_image(image)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"图像处理失败: {e}")

@app.post("/analyze_batch", summary="多张图片批量分析", dependencies=[Depends(verify_token)])
@limiter.limit("5/minute")  # 限流：每分钟最多5次请求
async def analyze_batch(request: Request, files: List[UploadFile] = File(...)):
    if len(files) > MAX_BATCH_FILES:
        raise HTTPException(status_code=400, detail=f"单次最多上传 {MAX_BATCH_FILES} 个文件")
    results = {}
    for file in files:
        try:
            data = await file.read()
            image = Image.open(io.BytesIO(data)).convert("RGB")
            results[file.filename] = await analyze_pil_image(image)
        except Exception as e:
            results[file.filename] = {"出错": str(e), "总细胞数": 0, "异常细胞数": 0, "细胞类": []}
    return results


@app.post("/analyze_zip", summary="ZIP压缩包批量分析", dependencies=[Depends(verify_token)])
@limiter.limit("3/minute")  # 限流：每分钟最多3
async def analyze_zip(request: Request, upload_file: UploadFile = File(...)):
    if not upload_file.filename.endswith(".zip"):
        raise HTTPException(status_code=400, detail="仅接受.zip文件")
    results = {}
    try:
        zip_bytes = await upload_file.read()
        if len(zip_bytes) > MAX_ZIP_UNCOMPRESSED:
            raise HTTPException(status_code=413, detail="ZIP文件过大")
        if not zip_bytes.startswith(b'PK\x03\x04'):
            raise HTTPException(status_code=400, detail="文件不是有效的ZIP格式")
        with zipfile.ZipFile(io.BytesIO(zip_bytes)) as z:
            valid_ext = ('.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp')
            infos = [i for i in z.infolist() if not i.is_dir() and i.filename.lower().endswith(valid_ext)]
            if not infos:
                raise HTTPException(status_code=400, detail="ZIP中找不到有效的图片")
            if len(infos) > MAX_ZIP_FILES:
                raise HTTPException(status_code=400, detail=f"ZIP内图片数量超过上限({MAX_ZIP_FILES})")
            total_uncompressed = sum(i.file_size for i in infos)
            if total_uncompressed > MAX_ZIP_UNCOMPRESSED:
                raise HTTPException(status_code=400, detail=f"ZIP解压后总大小超过上限({MAX_ZIP_UNCOMPRESSED / (1024*1024)} MB)")
            for info in infos:
                filename = info.filename
                try:
                    with z.open(filename) as img_file:
                        img_bytes = img_file.read()
                    validate_file_content(img_bytes, filename, "image/jpeg")
                    image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
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
@limiter.limit("20/minute")
async def agent_followup(
        request: Request,
        payload: FollowUpRequest,
        conv_manager: ConversationManager = Depends(get_conversation_manager)
):
    """
    处理诊断完成后的用户追问。
    """
    try:
        response = await conv_manager.followup_chat(
            session_id=payload.session_id,
            user_input=payload.message,
            time_location=build_time_location()
        )
        return {
            "success": True,
            "response": response
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/agent/clarify/{session_id}",dependencies=[Depends(verify_token)])
@limiter.limit("20/minute")
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
@limiter.limit("10/minute")
async def agent_upload(
        request: Request,
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
async def agent_websocket(websocket: WebSocket, session_id: str, token: str = Query(...)):
    # 握手前校验 JWT
    try:
        decode_token(token)
    except Exception:
        await websocket.close(code=4401)  # 4401: 自定义未授权关闭码
        return

    await manager.connect(session_id, websocket)
    conv_manager = get_conversation_manager()
    try:
        while True:
            message = await websocket.receive_text()
            try:
                data = json.loads(message)
            except json.JSONDecodeError:
                await manager.send_progress(session_id, {"type": "error", "message": "消息格式错误，需为 JSON"})
                continue
            user_input = (data.get("message") or "").strip()
            if not user_input:
                await manager.send_progress(session_id, {"type": "error", "message": "消息不能为空"})
                continue
            await manager.send_progress(session_id, {"type": "stream", "content": "正在分析您的补充信息..."})

        try:
            answer = await conv_manager.followup_chat(
                session_id=session_id,
                user_input=user_input,
                time_location=build_time_location()
            )
            await manager.send_progress(session_id, {"type": "stream", "content": answer})
            await manager.send_progress(session_id, {"type": "complete"})
        except Exception as e:
            await manager.send_progress(session_id, {"type": "error", "message": str(e)})
    except WebSocketDisconnect:
        manager.disconnect(session_id)
    except Exception:
        manager.disconnect(session_id)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("agent_api:app", host="0.0.0.0", port=8000, reload=True)