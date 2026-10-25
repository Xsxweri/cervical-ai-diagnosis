"""视觉分析路由：单图 / 批量 / ZIP / 健康检查"""

import asyncio
import io
import os
import zipfile
from typing import List

from fastapi import APIRouter, Depends, File, HTTPException, Request, UploadFile
from PIL import Image

from api.config import (
    ALLOWED_EXTS,
    MAX_BATCH_FILES,
    MAX_SINGLE_FILE,
    MAX_ZIP_FILES,
    MAX_ZIP_FILE,
    MAX_ZIP_UNCOMPRESSED,
)
from api.logging_setup import logger
from api.runtime import limiter
from api.schemas import _error_result
from api.security import validate_file_content
from core.auth import verify_token
from services.vision_service import (
    ABNORMAL_CONF_THRESHOLD,
    BINARY_WEIGHTS,
    DEVICE,
    MC_DROPOUT_SAMPLES,
    MULTI_WEIGHTS,
    MULTI_LABEL_SPACE,
    DUAL_MODE,
    PRIMARY_SPACE,
    AUX_SPACE,
    YOLO_WEIGHTS,
    analyze_pil_image,
)
from utils.uncertainty import UNCERTAINTY_HIGH, UNCERTAINTY_MEDIUM

router = APIRouter(tags=["vision"])


@router.post("/analyze", summary="单张图片双模型分析", dependencies=[Depends(verify_token)])
@limiter.limit("10/minute")
async def analyze_image(request: Request, file: UploadFile = File(...)):
    try:
        data = await file.read()
        if len(data) > MAX_SINGLE_FILE:
            raise HTTPException(status_code=413, detail="文件超过30MB限制")
        validate_file_content(data, file.filename or "", file.content_type)
        image = Image.open(io.BytesIO(data)).convert("RGB")
        return await analyze_pil_image(image)
    except HTTPException:
        raise
    except Exception:
        logger.exception("图像处理失败")
        raise HTTPException(status_code=400, detail="图像处理失败，请确认文件为有效的细胞图像")


@router.post("/analyze_batch", summary="多张图片批量分析", dependencies=[Depends(verify_token)])
@limiter.limit("5/minute")
async def analyze_batch(request: Request, files: List[UploadFile] = File(...)):
    if len(files) > MAX_BATCH_FILES:
        raise HTTPException(status_code=400, detail=f"单次最多上传 {MAX_BATCH_FILES} 个文件")

    async def _one(item: UploadFile):
        name = item.filename or "unnamed"
        try:
            data = await item.read()
            if len(data) > MAX_SINGLE_FILE:
                raise HTTPException(status_code=413, detail="文件超过30MB限制")
            validate_file_content(data, name, item.content_type)
            image = Image.open(io.BytesIO(data)).convert("RGB")
            return name, await analyze_pil_image(image)
        except HTTPException as e:
            return name, _error_result(str(e.detail))
        except Exception:
            logger.exception(f"批量分析失败: {name}")
            return name, _error_result("图像分析失败")

    # GPU 下并发过高会 OOM，用信号量限流
    sem = asyncio.Semaphore(int(os.getenv("ANALYZE_CONCURRENCY", "2")))

    async def _guarded(item: UploadFile):
        async with sem:
            return await _one(item)

    return {name: res for name, res in await asyncio.gather(*(_guarded(f) for f in files))}


@router.post("/analyze_zip", summary="ZIP压缩包批量分析", dependencies=[Depends(verify_token)])
@limiter.limit("3/minute")
async def analyze_zip(request: Request, upload_file: UploadFile = File(...)):
    if not upload_file.filename.endswith(".zip"):
        raise HTTPException(status_code=400, detail="仅接受.zip文件")

    results: dict = {}
    try:
        zip_bytes = await upload_file.read()
        if len(zip_bytes) > MAX_ZIP_FILE:
            raise HTTPException(status_code=413, detail="ZIP文件过大")
        if not zip_bytes.startswith(b"PK\x03\x04"):
            raise HTTPException(status_code=400, detail="文件不是有效的ZIP格式")

        with zipfile.ZipFile(io.BytesIO(zip_bytes)) as z:
            infos = [
                i for i in z.infolist()
                if not i.is_dir() and i.filename.lower().endswith(ALLOWED_EXTS)
            ]
            if not infos:
                raise HTTPException(status_code=400, detail="ZIP中找不到有效的图片")
            if len(infos) > MAX_ZIP_FILES:
                raise HTTPException(
                    status_code=400, detail=f"ZIP内图片数量超过上限({MAX_ZIP_FILES})"
                )
            total_uncompressed = sum(i.file_size for i in infos)
            if total_uncompressed > MAX_ZIP_UNCOMPRESSED:
                raise HTTPException(
                    status_code=400,
                    detail=f"ZIP解压后总大小超过上限({MAX_ZIP_UNCOMPRESSED / (1024 * 1024)} MB)",
                )

            # 先把字节全部读出并校验（同步、快速），随后即可关闭 ZIP
            # 这样长时间推理期间不再持有 zip 句柄，也避免并发时交错访问
            payloads: dict[str, bytes] = {}
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


@router.get("/health", summary="存活探针")
async def health_check():
    """
    公开存活探针。

    返回结果：
      {"status": "ok"}
    """
    return {"status": "ok"}


@router.get(
    "/api/system/info",
    summary="系统信息（需鉴权）",
    dependencies=[Depends(verify_token)],
)
async def system_info():
    """返回设备、模型权重路径、阈值与 RAG 可用性。"""
    # 检测知识库可用性
    from Agents.knowledge import UnavailableKnowledgeBase, get_kb

    try:
        rag_available = not isinstance(get_kb(), UnavailableKnowledgeBase)
    except RuntimeError:
        # init_kb 尚未完成
        rag_available = False
    return {
        "status": "ok",
        "device": str(DEVICE),
        "models": {
            "yolo": YOLO_WEIGHTS,
            "二分类": BINARY_WEIGHTS,
            "多分类": MULTI_WEIGHTS,
            "多分类标签空间": MULTI_LABEL_SPACE,
            "dual模式": DUAL_MODE,
            "主空间": PRIMARY_SPACE,
            "辅助空间": AUX_SPACE,
        },
        "config": {
            "mc_dropout_samples": MC_DROPOUT_SAMPLES,
            "不确定阈值_high": UNCERTAINTY_HIGH,
            "不确定阈值_medium": UNCERTAINTY_MEDIUM,
            "异常的CONF阈值": ABNORMAL_CONF_THRESHOLD,
        },
        "rag_available": rag_available,
    }