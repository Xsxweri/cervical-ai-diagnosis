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
from api.runtime import build_time_location, get_conversation_manager, limiter, manager
from api.schemas import FollowUpRequest, _agent_payload, _session_payload
from api.security import _internal_error, _requester_id, validate_file_content
from core.auth import decode_token, verify_token

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


@router.post("/api/agent/upload", dependencies=[Depends(verify_token)])
@limiter.limit("10/minute")
async def agent_upload(
    request: Request,
    file: UploadFile = File(...),
    doctor: dict = Depends(verify_token),
    conv_manager: ConversationManager = Depends(get_conversation_manager),
):
    """上传细胞图像并启动诊断流程。"""
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

    # 生产环境强烈建议将 analyze_image 和 start_diagnosis 放入 Celery 队列
    try:
        result = await conv_manager.start_diagnosis(
            file_path, session_id=session_id, owner_id=_requester_id(doctor)
        )
    except Exception as e:
        logger.exception("诊断启动失败")
        raise _internal_error(e)

    return _agent_payload(session_id, result)

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