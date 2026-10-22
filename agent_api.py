import torch
import torch._dynamo
import os
import uuid
import json
import logging
import time
import asyncio
from fastapi import FastAPI, UploadFile, File, WebSocket, WebSocketDisconnect, Depends, HTTPException, Body, Request, Query
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address
from pydantic import BaseModel as PydanticBaseModel
from typing import List

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
from logging.handlers import RotatingFileHandler
from services.vision_service import (
    analyze_pil_image,          # 核心分析函数
    DEVICE, YOLO_WEIGHTS,       # 配置常量
    BINARY_WEIGHTS, MULTI_WEIGHTS,
    MC_DROPOUT_SAMPLES,
    ABNORMAL_CONF_THRESHOLD,
)
from utils.uncertainty import (
    UNCERTAINTY_HIGH, UNCERTAINTY_MEDIUM, METRIC_NAME, UNCERTAINTY_PERCENTILE,
)

class FollowUpRequest(PydanticBaseModel):
    session_id: str
    message: str

class LoginRequest(PydanticBaseModel):
    username: str
    password: str

def _agent_payload(session_id: str, result: dict) -> dict:
    """统一 Agent 响应体：把不可判读原因、风险口径一并返回给前端。"""
    risk_level = result.get("risk_level")
    if risk_level == "unreadable":
        status = "unreadable"
    elif result.get("is_complete"):
        status = "complete"
    else:
        status = "awaiting_clarification"
    return {
        "session_id": session_id,
        "status": status,
        "risk_level": risk_level,
        "risk_scope": result.get("risk_scope"),
        "decision_reason": result.get("decision_reason"),
        "clarification_rounds": result.get("clarification_rounds", 0),
        "recommendations": result.get("recommendations"),
        "report": result.get("diagnosis_report"),
        "questions": result.get("clarification_questions"),
        "error_message": result.get("error_message"),
    }

# 引入连接管理器 (防止 WebSocket 内存泄漏)
class ConnectionManager:
    def __init__(self):
        self.active_connections: dict[str, WebSocket] = {}

    async def connect(self, session_id: str, websocket: WebSocket, subprotocol: str | None = None):
        await websocket.accept(subprotocol=subprotocol if subprotocol == "bearer" else None)
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

ACCOUNTS_PATH = os.getenv("DOCTOR_ACCOUNTS_PATH", os.path.join(os.path.dirname(os.path.abspath(__file__)), "appWeb", 'doctor_accounts.json'))
SERVICE_LOCATION = os.getenv("SERVICE_LOCATION", "未知地点")

LOG_DIR = os.path.abspath("./logs")
os.makedirs(LOG_DIR, exist_ok=True)
_LOGGING_READY = {"done": False}
def _setup_logging():
    if _LOGGING_READY["done"]:
        return
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s")

    root = logging.getLogger()
    root.setLevel(logging.INFO)
    console = logging.StreamHandler()
    console.setFormatter(fmt)
    root.addHandler(console)
    app_file = RotatingFileHandler(os.path.join(LOG_DIR, "agent_api.log"),
                                   maxBytes=10 * 1024 * 1024, backupCount=5, encoding="utf-8")

    app_file.setFormatter(fmt)
    root.addHandler(app_file)
    logging.getLogger("watchfiles").setLevel(logging.WARNING)

    audit_fmt = logging.Formatter(
            "%(asctime)s %(levelname)s %(message)s",
            datefmt="%Y-%m-%dT%H:%M:%S%z",
        )
    audit_handler = RotatingFileHandler(
        os.path.join(LOG_DIR, "audit.log"),
        maxBytes=10 * 1024 * 1024, backupCount=30, encoding="utf-8")
    audit_handler.setFormatter(audit_fmt)
    audit_logger.addHandler(audit_handler)
    audit_logger.setLevel(logging.INFO)
    audit_logger.propagate = False
    _LOGGING_READY["done"] = True

logger = logging.getLogger("agent_api")
audit_logger = logging.getLogger("audit") 
_setup_logging()

# uploads 清理配置：默认文件存活 6 小时，每 30 分钟扫描一次
UPLOAD_DIR = os.path.abspath("./uploads")
UPLOAD_MAX_AGE_SECONDS = int(os.getenv("UPLOAD_MAX_AGE_SECONDS", str(6 * 3600)))
UPLOAD_CLEANUP_INTERVAL = int(os.getenv("UPLOAD_CLEANUP_INTERVAL", "1800"))

# 定义文件头字典
MAGIC_BYTES:dict[str,list[bytes]] = {
    '.jpg': [b'\xff\xd8\xff'],
    '.jpeg': [b'\xff\xd8\xff'],
    '.png': [b'\x89PNG\r\n\x1a\n'],
    '.tif': [b'II*\x00', b'MM\x00*'],
    '.tiff': [b'II*\x00', b'MM\x00*'],
    '.bmp': [b'BM'],
    # '.svs': [b'II*\x00', b'MM\x00*'], # SVS文件通常是TIFF格式的变体
}

# 定义允许的 MIME 类型
ALLOWED_MIME = {
    '.jpg': 'image/jpeg',
    '.jpeg': 'image/jpeg',
    '.png': 'image/png',
    '.tif': 'image/tiff',
    '.tiff': 'image/tiff',
    '.bmp': 'image/bmp',
    # '.svs': {'image/tiff', 'image/svs', 'application/octet-stream'}
}

def _requester_id(token_payload: dict) -> str:
    """从 JWT payload 提取稳定的操作者标识（优先 uid，回退 username）。"""
    return str(token_payload.get("uid") or token_payload.get("sub") or "")

def _internal_error(e: Exception) -> HTTPException:
    """内部错误：服务端留全量堆栈 + 关联编号，客户端只拿到编号。
    避免借 500 响应泄露内部路径、SQL、堆栈等实现细节。
    """
    err_id = uuid.uuid4().hex[:12]
    logger.exception(f"[{err_id}] 内部错误: {e}")
    return HTTPException(status_code=500, detail=f"服务内部错误，请稍后重试（错误编号 {err_id}）")

# 文件安全校验函数
def validate_file_content(content: bytes, filename: str, content_type: str | None = None):
    ext = os.path.splitext(filename)[1].lower()

    allowed = ALLOWED_MIME.get(ext)
    if allowed and content_type:
        allowed_set = allowed if isinstance(allowed, (set, frozenset, tuple, list)) else {allowed}
        if content_type not in allowed_set:
            raise HTTPException(status_code=400, detail="Content-Type 与文件类型不符")

    sig = MAGIC_BYTES.get(ext)
    if sig and not any(content.startswith(s) for s in sig):
        raise HTTPException(status_code=400, detail=f"文件内容与扩展名不匹配: {filename}")
    if ext in {'.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp'}:
        try:
            with Image.open(io.BytesIO(content)) as img:
                img.verify()
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

def _cleanup_uploads():
    """删除 uploads/ 中超过 UPLOAD_MAX_AGE_SECONDS 的文件，防止磁盘只增不减"""
    if not os.path.isdir(UPLOAD_DIR):
        return
    now = time.time()
    for name in os.listdir(UPLOAD_DIR):
        path = os.path.join(UPLOAD_DIR, name)
        try:
            if os.path.isfile(path) and now - os.path.getmtime(path) > UPLOAD_MAX_AGE_SECONDS:
                os.remove(path)
        except OSError as e:
            logger.warning(f"清理上传文件失败 {path}: {e}")

async def _uploads_cleanup_loop():
    """周期性清理任务，随应用生命周期启动/取消"""
    while True:
        try:
            await asyncio.sleep(UPLOAD_CLEANUP_INTERVAL)
            await asyncio.to_thread(_cleanup_uploads)
        except asyncio.CancelledError:
            break
        except Exception as e:
            logger.warning(f"uploads 清理任务异常: {e}")

@asynccontextmanager
async def lifespan(app: FastAPI):
    from Agents.diagnosis_graph import init_checkpoint_store, close_checkpoint_store, init_kb
    from services.vision_service import get_model_manager
    await asyncio.to_thread(get_model_manager)
    await init_kb()
    await init_checkpoint_store()
    cleanup_task = asyncio.create_task(_uploads_cleanup_loop())
    yield
    cleanup_task.cancel()
    await close_checkpoint_store()

app = FastAPI(title="AI Agent宫颈病理诊断系统", version="2.0", lifespan=lifespan)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    err_id = uuid.uuid4().hex[:12]
    logger.exception(f"[{err_id}] 未捕获异常: {exc}")
    return JSONResponse(
        status_code=500,
        content={"detail": f"服务内部错误，请稍后重试（错误编号 {err_id}）"},
        headers=_cors_headers_for(request),
    )

def _cors_headers_for(request: Request) -> dict:
    """为越过 CORSMiddleware 的响应（如 500）按白名单手动补 CORS 头。"""
    origin = request.headers.get("origin")
    if not origin:
        return {}
    allowed = [o.strip() for o in os.getenv("CORS_ORIGINS", "").split(",") if o.strip()]
    if origin not in allowed:
        return {}
    return {"Access-Control-Allow-Origin": origin, "Vary": "Origin"}

# 医疗合规审计日志：记录 IP、方法、路径、状态码、延迟
@app.middleware("http")
async def audit_log_middleware(request: Request, call_next):
    start = time.perf_counter()
    client_ip = request.client.host if request.client else "unknown"

    def _who() -> str:
        """操作者身份；未认证请求记为 anonymous。"""
        return getattr(request.state, "requester_id", "") or "anonymous"
    
    try:
        response = await call_next(request)
    except Exception as e:
        err_id = uuid.uuid4().hex[:12]
        latency = (time.perf_counter() - start) * 1000
        logger.exception(f"[{err_id}] 未捕获异常: {e}")
        audit_logger.error(
            f"operator={_who()} ip={client_ip} {request.method} {request.url.path} "
            f"500 {latency:.1f}ms err={err_id}"
        )
        return JSONResponse(
            status_code=500,
            content={"detail": f"服务内部错误，请稍后重试（错误编号 {err_id}）"},
            headers=_cors_headers_for(request),        # ← 新增
        )
    latency = (time.perf_counter() - start) * 1000
    audit_logger.info(
        f"operator={_who()} ip={client_ip} {request.method} {request.url.path} "
        f"{response.status_code} {latency:.1f}ms"
    )
    return response

app.add_middleware(
    CORSMiddleware,
    allow_origins= os.getenv("CORS_ORIGINS", "http://localhost:3000").split(","),  # 仅允许白名单
    allow_methods=["*"],
    allow_headers=["*"]
)


# === FastAPI 接口 ===
# 登录接口
@app.post("/api/auth/login", summary="医生登录")
@limiter.limit("5/minute")
async def login(request: Request, req: LoginRequest):
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

def _error_result(detail: str) -> dict:
    """单图失败时的占位结果，字段与 _analyze_pil_image_sync 完全一致。"""
    return {
        "total_cells": 0,
        "abnormal_cells": 0,
        "cells": [],
        "highest_risk": "none",
        "interpretable": False,
        "unreadable_reason": None,
        "risk_level": None,
        "risk_scope": None,
        "max_uncertainty": 0.0,
        "uncertainty_metric": METRIC_NAME,
        "uncertainty_aggregation": f"p{UNCERTAINTY_PERCENTILE}",
        "uncertainty_thresholds": {"high": UNCERTAINTY_HIGH, "medium": UNCERTAINTY_MEDIUM},
        "needs_clarification": False,
        "error": detail,
    }

# 批量/压缩分析的资源上限常量
MAX_BATCH_FILES = 10
MAX_ZIP_FILES = 20
MAX_SINGLE_FILE = 30 * 1024 * 1024  # 单文件最大30MB
MAX_ZIP_FILE = 50 * 1024 * 1024
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
        logger.exception("图像处理失败")
        raise HTTPException(status_code=400, detail="图像处理失败，请确认文件为有效的细胞图像")

@app.post("/analyze_batch", summary="多张图片批量分析", dependencies=[Depends(verify_token)])
@limiter.limit("5/minute")
async def analyze_batch(request: Request, files: List[UploadFile] = File(...)):
    if len(files) > MAX_BATCH_FILES:
        raise HTTPException(status_code=400, detail=f"单次最多上传 {MAX_BATCH_FILES} 个文件")

    async def _one(file: UploadFile):
        name = file.filename or "unnamed"
        try:
            data = await file.read()
            if len(data) > MAX_SINGLE_FILE:
                raise HTTPException(status_code=413, detail="文件超过30MB限制")
            validate_file_content(data, name, file.content_type)
            image = Image.open(io.BytesIO(data)).convert("RGB")
            return name, await analyze_pil_image(image)
        except HTTPException as e:
            return name, _error_result(str(e.detail))
        except Exception:
            logger.exception(f"批量分析失败: {name}")
            return name, _error_result("图像分析失败")

    # GPU 下并发过高会 OOM，用信号量限流
    sem = asyncio.Semaphore(int(os.getenv("ANALYZE_CONCURRENCY", "2")))

    async def _guarded(file):
        async with sem:
            return await _one(file)

    return {name: res for name, res in await asyncio.gather(*(_guarded(f) for f in files))}


@app.post("/analyze_zip", summary="ZIP压缩包批量分析", dependencies=[Depends(verify_token)])
@limiter.limit("3/minute")  # 限流：每分钟最多3
async def analyze_zip(request: Request, upload_file: UploadFile = File(...)):
    if not upload_file.filename.endswith(".zip"):
        raise HTTPException(status_code=400, detail="仅接受.zip文件")
    results = {}
    try:
        zip_bytes = await upload_file.read()
        if len(zip_bytes) > MAX_ZIP_FILE:
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
            
            # 先把字节全部读出并校验（同步、快速），随后即可关闭 ZIP
            # 这样长时间推理期间不再持有 zip 句柄，也避免并发时交错访问
            payloads = {}
            for info in infos:
                filename = info.filename
                try:
                    with z.open(filename) as img_file:
                        img_bytes = img_file.read()
                    validate_file_content(img_bytes, filename)
                    payloads[filename] = img_bytes
                except HTTPException as e:
                    results[filename] = _error_result(str(e.detail))
                except Exception:
                    logger.exception(f"ZIP 内图片读取失败: {filename}")
                    results[filename] = _error_result("图像读取失败")
            
            # 并发推理（GPU 下需限流，避免显存 OOM）
            sem = asyncio.Semaphore(int(os.getenv("ANALYZE_CONCURRENCY", "2")))
            async def _one_entry(filename: str, img_bytes: bytes):
                try:
                    image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
                    return filename, await analyze_pil_image(image)
                except Exception:
                    logger.exception(f"ZIP 内图片分析失败: {filename}")
                    return filename, _error_result("图像分析失败")

            async def _guarded(filename: str, img_bytes: bytes):
                async with sem:
                    return await _one_entry(filename, img_bytes)

            if payloads:
                for name, res in await asyncio.gather(
                    *(_guarded(fn, b) for fn, b in payloads.items())
                ):
                    results[name] = res
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
            "不确定阈值_high": UNCERTAINTY_HIGH,
            "不确定阈值_medium": UNCERTAINTY_MEDIUM,
            "异常的CONF阈值": ABNORMAL_CONF_THRESHOLD,
        }
    }

# --- FastAPI 路由 ---
@app.post("/api/agent/followup")
@limiter.limit("20/minute")
async def agent_followup(
        request: Request,
        payload: FollowUpRequest,
        doctor: dict = Depends(verify_token),
        conv_manager: ConversationManager = Depends(get_conversation_manager)
):
    """
    处理诊断完成后的用户追问。
    """
    try:
        response = await conv_manager.followup_chat(
            session_id=payload.session_id,
            user_input=payload.message,
            time_location=build_time_location(),
            requester_id=_requester_id(doctor),
        )
        return {"success": True, "response": response}
    except PermissionError as e:
        raise HTTPException(status_code=403, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise _internal_error(e)

@app.post("/api/agent/clarify/{session_id}")
@limiter.limit("20/minute")
async def agent_clarify(
    request: Request, 
    session_id: str, 
    body: dict = Body(...), 
    doctor: dict = Depends(verify_token),
    conv_manager: ConversationManager = Depends(get_conversation_manager)):
    """
    处理用户对诊断报告的澄清请求。
    """
    user_input = body.get("user_input", "")
    if not user_input:
        raise HTTPException(status_code=400, detail="缺少 user_input 字段")
    try:
        result = await conv_manager.process_clarification(
            session_id, user_input, requester_id=_requester_id(doctor))
    except PermissionError as e:
        raise HTTPException(status_code=403, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise _internal_error(e)
    return _agent_payload(session_id, result)

# 4. 优化后的上传接口
@app.post("/api/agent/upload", dependencies=[Depends(verify_token)])
@limiter.limit("10/minute")
async def agent_upload(
        request: Request,
        file: UploadFile = File(...),
        doctor: dict = Depends(verify_token), 
        conv_manager=Depends(get_conversation_manager)
):
    allowed_exits = {'.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp'}
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
        result = await conv_manager.start_diagnosis(file_path, session_id=session_id, owner_id=_requester_id(doctor))
    except Exception as e:
        logger.exception("诊断启动失败")
        raise _internal_error(e)

    return _agent_payload(session_id, result)

def _extract_ws_token(websocket: WebSocket) -> str:
    """从 Sec-WebSocket-Protocol 子协议中取 JWT，避免 token 出现在 URL query 里。

    前端写法: new WebSocket(url, ["bearer", token])
    浏览器会把它们拼成: "bearer, <token>"
    """
    raw = websocket.headers.get("sec-websocket-protocol", "")
    for part in raw.split(","):
        part = part.strip()
        if part and part != "bearer":
            return part
    return ""

# 5. 健壮的 WebSocket 端点
@app.websocket("/ws/agent/chat/{session_id}")
async def agent_websocket(websocket: WebSocket, session_id: str):
    # 握手前校验 JWT
    token = _extract_ws_token(websocket) or websocket.query_params.get("token", "")
    if not token:
        await websocket.close(code=4401)
        return
    try:
        token_payload = decode_token(token)
    except Exception:
        await websocket.close(code=4401)
        return
    requester_id = _requester_id(token_payload)
    audit_logger.info(
        f"operator={requester_id or 'anonymous'} ip="
        f"{websocket.client.host if websocket.client else 'unknown'} "
        f"WS /ws/agent/chat session={session_id} connected"
    )

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
                    time_location=build_time_location(),
                    requester_id=requester_id
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
    uvicorn.run(
        "agent_api:app", host="0.0.0.0", port=8000,
        reload=True,
        reload_excludes=["logs/*", "uploads/*", "knowledge_db/*", "outputs/*", "runs/*", "*.log"],
    )