"""Agent 路由：诊断追问、澄清、上传、WebSocket。"""

import json
import os
import uuid

from fastapi import (
    APIRouter,
    Body,
    Depends,
    File,
    HTTPException,
    Request,
    UploadFile,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.responses import FileResponse
from werkzeug.utils import secure_filename

from Agents.conversation import ConversationManager
from api.config import ALLOWED_EXTS, MAX_SINGLE_FILE, UPLOAD_DIR
from api.logging_setup import audit_logger, logger
from api.runtime import build_time_location, diagnosis_mode, get_arq_pool, get_conversation_manager, limiter, manager
from api.schemas import FollowUpRequest, _agent_payload, _session_payload
from api.security import _internal_error, _requester_id, validate_file_content
from core.auth import decode_token, verify_token
from core.redis_client import get_redis_client
from core import task_store

router = APIRouter(tags=["agent"])


@router.post("/api/agent/followup")
@limiter.limit("20/minute")
async def agent_followup(
    request: Request,
    payload: FollowUpRequest,
    doctor: dict = Depends(verify_token),
    conv_manager: ConversationManager = Depends(get_conversation_manager),
):
    """处理诊断完成后的用户追问。"""
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


@router.post("/api/agent/clarify/{session_id}")
@limiter.limit("20/minute")
async def agent_clarify(
    request: Request,
    session_id: str,
    body: dict = Body(...),
    doctor: dict = Depends(verify_token),
    conv_manager: ConversationManager = Depends(get_conversation_manager),
):
    """处理用户对澄清问题的回答。"""
    user_input = body.get("user_input", "")
    if not user_input:
        raise HTTPException(status_code=400, detail="缺少 user_input 字段")
    try:
        result = await conv_manager.process_clarification(
            session_id, user_input, requester_id=_requester_id(doctor)
        )
    except PermissionError as e:
        raise HTTPException(status_code=403, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise _internal_error(e)
    return _agent_payload(session_id, result)


@router.post("/api/agent/upload", status_code=202, summary="上传图像并投递诊断任务")
@limiter.limit("10/minute")
async def agent_upload(
    request: Request,
    file: UploadFile = File(...),
    doctor: dict = Depends(verify_token),
    conv_manager: ConversationManager = Depends(get_conversation_manager),
):
    """
    上传图像并投递诊断任务
    """
    safe_name = secure_filename(file.filename or "")
    if not safe_name or not safe_name.lower().endswith(ALLOWED_EXTS):
        raise HTTPException(status_code=400, detail="不支持的文件格式")

    session_id = str(uuid.uuid4())
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    file_path = os.path.join(UPLOAD_DIR, f"{session_id}_{safe_name}")
    if not file_path.startswith(UPLOAD_DIR):
        raise HTTPException(status_code=400, detail="非法文件路径")

    content = await file.read()
    if len(content) > MAX_SINGLE_FILE:
        raise HTTPException(status_code=413, detail="文件超过30MB限制")

    validate_file_content(content, safe_name, file.content_type)
    with open(file_path, "wb") as f:
        f.write(content)

    owner_id = _requester_id(doctor)

    if diagnosis_mode() != "queue":
        try:
            result = await conv_manager.start_diagnosis(
                file_path, session_id=session_id, owner_id=owner_id
            )
        except Exception as e:
            logger.exception("诊断启动失败")
            raise _internal_error(e)
        return _agent_payload(session_id, result)

    client = await get_redis_client()
    if not await client.exists(HEARTBEAT_KEY):
        raise HTTPException(
            status_code=503, detail="诊断服务暂不可用（无可用 worker），请联系信息科"
        )
    
    # --- queue 模式 ---
    queue = os.getenv("ARQ_QUEUE_NAME", "arq:diagnosis")
    await task_store.set_status(session_id, task_store.QUEUED, owner_id=owner_id)
    try:
        job = await get_arq_pool().enqueue_job(
            "run_diagnosis",
            session_id,
            file_path,
            owner_id,
            _job_id=f"diag:{session_id}",
            _queue_name=queue,
        )
    except Exception as e:
        # 不静默降级为同步：否则"队列坏了"会表现成"偶尔很慢"
        await task_store.set_status(session_id, task_store.FAILED, error=f"投递失败: {e}")
        logger.exception("诊断任务投递失败")
        raise HTTPException(status_code=503, detail="诊断队列不可用，请稍后重试")

    if job is None:
        logger.warning("任务已存在，复用: %s", session_id)

    audit_logger.info(
        f"operator={owner_id} ip={request.client.host if request.client else 'unknown'} "
        f"UPLOAD queued session={session_id} file={safe_name}"
    )
    return {
        "session_id": session_id,
        "task_status": task_store.QUEUED,
        "queue": queue,
        "poll_interval_ms": 1500,
    }

@router.get("/api/agent/task/{session_id}", summary="查询诊断任务状态")
@limiter.limit("120/minute")     # 轮询接口需要宽松限流
async def agent_task_status(
    request: Request,
    session_id: str,
    doctor: dict = Depends(verify_token),
    conv_manager: ConversationManager = Depends(get_conversation_manager),
):
    """
    查询诊断任务状态
    """
    status = await task_store.get_status(session_id)

    def _assert_owner(status: dict) -> None:
        owner = str(status.get("owner_id") or "")
        if owner and owner != _requester_id(doctor):
            raise HTTPException(status_code=403, detail="无权访问该任务")

    if status is None:
        # 同步模式下没有 task 记录，回退查会话（兼容旧客户端与历史会话）
        state = await conv_manager.get_session_state(
            session_id, requester_id=_requester_id(doctor)
        )
        if state:
            return {**_session_payload(session_id, state), "task_status": task_store.DONE}
        raise HTTPException(status_code=404, detail="任务不存在或已过期")

    if status["status"] == task_store.FAILED:
        return {
            "task_status": task_store.FAILED,
            "session_id": session_id,
            "error": status.get("error") or "诊断失败",
        }

    if status["status"] != task_store.DONE:
        return {
            "task_status": status["status"],
            "session_id": session_id,
        }

    # 完成：组装与会话读取一致的 payload
    state = await conv_manager.get_session_state(
        session_id, requester_id=_requester_id(doctor)
    )
    if not state:
        raise HTTPException(status_code=404, detail="任务已完成但会话不可读（可能已过期）")
    return {**_session_payload(session_id, state), "task_status": task_store.DONE}

@router.get("/api/agent/session/{session_id}", summary="读取会话完整结果")
@limiter.limit("30/minute")
async def agent_session(
    request: Request,
    session_id: str,
    doctor: dict = Depends(verify_token),
    conv_manager: ConversationManager = Depends(get_conversation_manager),
):
    """
    读取会话完整报告与细胞明细
    """
    try:
        state = await conv_manager.get_session_state(
            session_id, requester_id=_requester_id(doctor)
        )
    except PermissionError as e:
        raise HTTPException(status_code=403, detail=str(e))
    except Exception as e:
        raise _internal_error(e)

    if not state:
        raise HTTPException(status_code=404, detail="会话不存在或已过期")
    return _session_payload(session_id, state)


@router.get("/api/agent/session/{session_id}/image", summary="读取会话原图")
async def agent_session_image(
    session_id: str,
    doctor: dict = Depends(verify_token),
    conv_manager: ConversationManager = Depends(get_conversation_manager),
):
    """
    读取会话的原图, 用于报告页的证据链核对
    """
    try:
        state = await conv_manager.get_session_state(
            session_id, requester_id=_requester_id(doctor)
        )
    except PermissionError as e:
        raise HTTPException(status_code=403, detail=str(e))
    except Exception as e:
        raise _internal_error(e)

    if not state:
        raise HTTPException(status_code=404, detail="会话不存在或已过期")

    path = state.get("image_path")
    if not path or not os.path.isfile(path):
        raise HTTPException(status_code=404, detail="原图已清理或不存在")

    # 双重保险：即使状态被篡改，也不允许读到 uploads 之外的文件
    if not os.path.abspath(path).startswith(UPLOAD_DIR):
        raise HTTPException(status_code=403, detail="非法文件路径")

    return FileResponse(path)

GRADCAM_DIR = os.path.abspath("outputs/gradcam")

@router.post("/api/agent/session/{session_id}/gradcam", summary="按需生成 Grad-CAM")
@limiter.limit("5/minute")   # 生成涉及 GPU 推理，限流比读取更严
async def agent_generate_gradcam(
    request: Request,
    session_id: str,
    doctor: dict = Depends(verify_token),
    conv_manager: ConversationManager = Depends(get_conversation_manager),
):
    """按需为已完成的会话生成 Grad-CAM 叠加图。

    幂等：已生成则直接返回路径，不重复推理。
    失败统一返回 404 + 明确原因（热力图是解释性附加物，不应让接口 500）。
    """
    try:
        path = await conv_manager.generate_gradcam(
            session_id, requester_id=_requester_id(doctor)
        )
    except PermissionError as e:
        raise HTTPException(status_code=403, detail=str(e))
    except ValueError as e:
        # 会话不存在 / 流程未完成
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.exception("Grad-CAM 生成失败")
        raise _internal_error(e)

    if not path:
        raise HTTPException(
            status_code=404,
            detail="无法生成热力图：可能未检出可解释的异常细胞，或图像已被清理",
        )

    return {
        "session_id": session_id,
        "gradcam_available": True,
        "gradcam_analysis": None,   # 由 GET 读取会话时返回，此处不重复查
    }

@router.get("/api/agent/session/{session_id}/gradcam", summary="读取 Grad-CAM 叠加图")
@limiter.limit("30/minute")
async def agent_session_gradcam(
    request: Request,
    session_id: str,
    doctor: dict = Depends(verify_token),
    conv_manager: ConversationManager = Depends(get_conversation_manager),
):
    """读取会话的 Grad-CAM 叠加图（仅当 ENABLE_GRADCAM 开启时存在）。"""
    try:
        state = await conv_manager.get_session_state(
            session_id, requester_id=_requester_id(doctor)
        )
    except PermissionError as e:
        raise HTTPException(status_code=403, detail=str(e))
    except Exception as e:
        raise _internal_error(e)

    if not state:
        raise HTTPException(status_code=404, detail="会话不存在或已过期")

    path = state.get("gradcam_path")
    if not path or not os.path.isfile(path):
        raise HTTPException(status_code=404, detail="热力图未生成或已清理")

    # 双重保险：即使状态被篡改，也不允许读到 outputs/gradcam 之外的文件
    if not os.path.abspath(path).startswith(GRADCAM_DIR):
        raise HTTPException(status_code=403, detail="非法文件路径")

    return FileResponse(path)
def _extract_ws_token(websocket: WebSocket) -> str:
    """
    从 WebSocket 请求头中提取 JWT
    """
    raw = websocket.headers.get("sec-websocket-protocol", "")
    for part in raw.split(","):
        part = part.strip()
        if part and part != "bearer":
            return part
    return ""


@router.websocket("/ws/agent/chat/{session_id}")
async def agent_websocket(websocket: WebSocket, session_id: str):
    # 握手前校验 JWT
    # 子协议优先；`?token=` 仅为灰度期兼容旧客户端保留
    # 待确认所有客户端均已升级后，删除 `or websocket.query_params...`，
    # 让凭证彻底不经过 URL（避免进入访问日志中）
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
    offered = [
        p.strip()
        for p in websocket.headers.get("sec-websocket-protocol", "").split(",")
        if p.strip()
    ]

    await manager.connect(session_id, websocket, subprotocol="bearer" if "bearer" in offered else None)
    
    conv_manager = get_conversation_manager()
    try:
        while True:
            message = await websocket.receive_text()
            try:
                data = json.loads(message)
            except json.JSONDecodeError:
                await manager.send_progress(
                    session_id, {"type": "error", "message": "消息格式错误，需为 JSON"}
                )
                continue
            user_input = (data.get("message") or "").strip()
            if not user_input:
                await manager.send_progress(
                    session_id, {"type": "error", "message": "消息不能为空"}
                )
                continue
            await manager.send_progress(
                session_id, {"type": "stream", "content": "正在分析您的补充信息..."}
            )

            try:
                answer = await conv_manager.followup_chat(
                    session_id=session_id,
                    user_input=user_input,
                    time_location=build_time_location(),
                    requester_id=requester_id,
                )
                await manager.send_progress(session_id, {"type": "stream", "content": answer})
                await manager.send_progress(session_id, {"type": "complete"})
            except Exception as e:
                await manager.send_progress(session_id, {"type": "error", "message": str(e)})
    except WebSocketDisconnect:
        manager.disconnect(session_id)
    except Exception:
        manager.disconnect(session_id)