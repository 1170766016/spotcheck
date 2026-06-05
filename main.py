"""
SpotCheck AI — 设备屏显智能点检系统

FastAPI 主入口，提供图片上传与 AI 识别 API。
"""
import os
os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"
import io
import time
import uuid
import base64
from contextlib import asynccontextmanager

import cv2
import numpy as np
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from config import BASE_DIR, UPLOAD_DIR, FRONTEND_DIR, SERVER_CONFIG
from ai_engine import ocr_engine
from ai_engine.preprocessor import preprocess_image, create_annotated_image
from ai_engine.parser import parse_ocr_results, get_raw_texts


# ============================================================
# 应用生命周期
# ============================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用启动时预热 OCR 引擎。"""
    print("=" * 50)
    print("  SpotCheck AI 正在启动...")
    print("=" * 50)
    ocr_engine.warmup()
    print("=" * 50)
    print("  [OK] SpotCheck AI 启动完成")
    print(f"  [INFO] 访问地址: http://localhost:{SERVER_CONFIG['port']}")
    print("=" * 50)
    yield


# ============================================================
# FastAPI 应用
# ============================================================

app = FastAPI(
    title="SpotCheck AI",
    description="制造业设备屏显智能点检系统",
    version="0.1.0",
    lifespan=lifespan,
)

# CORS 中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# API 路由
# ============================================================

@app.post("/api/recognize")
async def recognize_image(image: UploadFile = File(...)):
    """
    核心接口：上传设备屏显图片，返回提取的参数列表。

    Returns:
        {
            "success": true,
            "data": {
                "parameters": [
                    {"name": "模具温度", "value": "180", "unit": "℃", ...}
                ],
                "raw_texts": [...],
                "annotated_image": "base64...",
                "stats": {...}
            }
        }
    """
    total_start = time.time()

    # 验证文件类型
    if not image.content_type or not image.content_type.startswith("image/"):
        raise HTTPException(400, "请上传图片文件（JPG/PNG）")

    try:
        # 读取图片数据
        image_bytes = await image.read()
        if len(image_bytes) == 0:
            raise HTTPException(400, "图片文件为空")

        print("\n" + "="*60)
        print("  [API] 开始图像分析请求...")
        print("="*60)

        # Step 1: 图像预处理
        preprocess_start = time.time()
        processed_img, preprocess_info = preprocess_image(image_bytes)
        preprocess_time = round((time.time() - preprocess_start) * 1000, 1)
        print(f"[API] [Step 1] 图像预处理完成 | 原图尺寸: {preprocess_info['original_size']} -> 预处理尺寸: {preprocess_info['processed_size']} | 耗时: {preprocess_time}ms")

        # Step 2: OCR 识别
        print("[API] [Step 2] 开始运行 OCR 识别...")
        ocr_results, ocr_stats = ocr_engine.recognize(processed_img)
        print(f"[API] [Step 2] OCR 识别完成 | 文字行数量: {ocr_stats['text_count']} | OCR 模型加载耗时: {ocr_engine._model_load_time}ms | OCR 推理执行耗时: {ocr_stats['ocr_time_ms']}ms")

        # Step 3: 参数解析
        print("[API] [Step 3] 开始参数提取解析...")
        parse_start = time.time()
        parameters = parse_ocr_results(ocr_results)
        raw_texts = get_raw_texts(ocr_results)
        parse_time = round((time.time() - parse_start) * 1000, 1)
        print(f"[API] [Step 3] 参数解析完成 | 提取到的参数数量: {len(parameters)} | 耗时: {parse_time}ms")

        # Step 4: 生成标注图片
        print("[API] [Step 4] 开始生成可视化标注图片...")
        annotate_start = time.time()
        annotated_img = create_annotated_image(
            processed_img, ocr_results, parameters
        )
        # 编码为 base64
        _, img_buffer = cv2.imencode(
            ".jpg", annotated_img, [cv2.IMWRITE_JPEG_QUALITY, 85]
        )
        annotated_b64 = base64.b64encode(img_buffer).decode("utf-8")
        annotate_time = round((time.time() - annotate_start) * 1000, 1)
        print(f"[API] [Step 4] 标注图片生成完成 | 耗时: {annotate_time}ms")

        # 原图也转为 base64（用于前端对比展示）
        _, orig_buffer = cv2.imencode(
            ".jpg", processed_img, [cv2.IMWRITE_JPEG_QUALITY, 85]
        )
        original_b64 = base64.b64encode(orig_buffer).decode("utf-8")

        # 总耗时
        total_time = round((time.time() - total_start) * 1000, 1)
        print("="*60)
        print(f"  [API] 图像分析处理成功！总耗时: {total_time}ms")
        print("="*60 + "\n")

        # 保存上传的原始图片（用于后续追溯）
        save_name = f"{uuid.uuid4().hex[:12]}.jpg"
        save_path = os.path.join(UPLOAD_DIR, save_name)
        with open(save_path, "wb") as f:
            f.write(image_bytes)

        return JSONResponse({
            "success": True,
            "data": {
                "parameters": parameters,
                "raw_texts": raw_texts,
                "original_image": original_b64,
                "annotated_image": annotated_b64,
                "stats": {
                    "total_time_ms": total_time,
                    "preprocess_time_ms": preprocess_info["preprocess_time_ms"],
                    "ocr_time_ms": ocr_stats["ocr_time_ms"],
                    "parse_time_ms": parse_time,
                    "annotate_time_ms": annotate_time,
                    "text_count": ocr_stats["text_count"],
                    "param_count": len(parameters),
                    "original_size": preprocess_info["original_size"],
                    "processed_size": preprocess_info["processed_size"],
                },
            },
        })

    except HTTPException:
        raise
    except Exception as e:
        print(f"[ERROR] 识别失败: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(500, f"识别处理失败: {str(e)}")


@app.get("/api/health")
async def health_check():
    """健康检查接口。"""
    return {"status": "ok", "service": "SpotCheck AI"}


# ============================================================
# 前端静态文件服务
# ============================================================

# 前端首页
@app.get("/")
async def serve_index():
    index_path = os.path.join(FRONTEND_DIR, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return {"message": "SpotCheck AI API is running. Frontend not found."}


# 静态文件
app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")


# ============================================================
# 启动入口
# ============================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host=SERVER_CONFIG["host"],
        port=SERVER_CONFIG["port"],
        reload=SERVER_CONFIG["reload"],
    )
