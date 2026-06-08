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
import json
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from config import BASE_DIR, UPLOAD_DIR, FRONTEND_DIR, SERVER_CONFIG, OCR_TUNING_DEFAULTS, IMAGE_TUNING_DEFAULTS
from ai_engine import ocr_engine
from ai_engine.preprocessor import preprocess_image, create_annotated_image, auto_tune_params, load_saved_params, save_params_to_file
from ai_engine.parser import parse_ocr_results, get_raw_texts


# ============================================================
# 应用生命周期
# ============================================================

# ============================================================
# 全局已保存参数（启动时加载）
# ============================================================
_saved_global_params: dict | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用启动时预热 OCR 引擎并加载已保存参数。"""
    global _saved_global_params
    print("=" * 50)
    print("  SpotCheck 正在启动...")
    print("=" * 50)

    # 加载已保存的参数
    _saved_global_params = load_saved_params()
    if _saved_global_params:
        print(f"  [OK] 已加载保存的默认参数")
    else:
        print(f"  [INFO] 无已保存的参数文件，使用系统默认值")

    ocr_engine.warmup()
    print("=" * 50)
    print("  [OK] SpotCheck 启动完成")
    print(f"  [INFO] 访问地址: http://localhost:{SERVER_CONFIG['port']}")
    print("=" * 50)
    yield


# ============================================================
# FastAPI 应用
# ============================================================

app = FastAPI(
    title="SpotCheck",
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
async def recognize_image(
    image: UploadFile = File(...),
    debug: bool = False,
    params: str = Form(None)
):
    """
    核心接口：上传设备屏显图片，返回提取的参数列表。

    Query Parameters:
        debug: 是否返回详细诊断信息（默认 False）

    Returns:
        {
            "success": true,
            "data": {
                "parameters": [
                    {"name": "模具温度", "value": "180", "unit": "℃", "confidence": 0.95, "source": "spatial"}
                ],
                "raw_texts": [...],
                "annotated_image": "base64...",
                "original_image": "base64...",
                "stats": {
                    "total_time_ms": 1234,
                    "preprocess_time_ms": 45,
                    "ocr_time_ms": 890,
                    "parse_time_ms": 50,
                    "annotate_time_ms": 45,
                    "text_count": 12
                },
                "diagnostics": {  # 仅当 debug=true 时
                    "original_size": "1920x1080",
                    "processed_size": "960x540",
                    "brightness_level": "normal",
                    "is_screen_display": true,
                    "clarity_level": "normal",
                    "applied_enhancements": ["CLAHE_contrast", "edge_enhance"]
                }
            },
            "error": null
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

        # 解析自定义参数（如无前端参数则使用已保存的全局参数）
        custom_params = None
        ocr_params = None
        if params:
            try:
                custom_params = json.loads(params)
                print(f"[API] [后台日志] 收到自定义图像预处理参数:")
                for k, v in custom_params.items():
                    print(f"  - {k}: {v}")
            except Exception as e:
                print(f"[API] [警告] 自定义参数解析失败: {e}")

        # 如果前端没有发送参数，使用已保存的全局参数
        if custom_params is None and _saved_global_params:
            custom_params = dict(_saved_global_params)
            print(f"[API] [后台日志] 使用已保存的全局默认参数: {list(custom_params.keys())}")

        # 提取 OCR 专用参数
        if custom_params:
            ocr_keys = list(OCR_TUNING_DEFAULTS.keys())
            ocr_params = {k: custom_params[k] for k in ocr_keys if k in custom_params}
            if ocr_params:
                print(f"[API] [OCR 参数] 提取到 OCR 运行时参数: {ocr_params}")

        print("\n" + "="*70)
        print("  [API] 开始图像分析请求" + (" [DEBUG 模式]" if debug else "") + "...")
        print("="*70)

        # Step 1: 图像预处理
        preprocess_start = time.time()
        processed_img, preprocess_info = preprocess_image(image_bytes, debug=debug, params=custom_params)
        preprocess_time = round((time.time() - preprocess_start) * 1000, 1)
        print(f"[API] [预处理] 原图: {preprocess_info['original_size']} → 处理后: {preprocess_info['processed_size']} | 增强: {preprocess_info['applied_enhancements']} | 耗时: {preprocess_time}ms")

        # Step 2: OCR 识别
        print("[API] [OCR 识别] 启动模型推理...")
        ocr_results, ocr_stats = ocr_engine.recognize(processed_img, ocr_params=ocr_params)
        print(f"[API] [OCR 识别] 识别到 {ocr_stats['text_count']} 个文字框 | 耗时: {ocr_stats['ocr_time_ms']}ms")

        # Step 3: 参数解析
        print("[API] [参数提取] 开始智能参数配对...")
        parse_start = time.time()
        parameters = parse_ocr_results(ocr_results, debug=debug)
        raw_texts = get_raw_texts(ocr_results)
        parse_time = round((time.time() - parse_start) * 1000, 1)
        print(f"[API] [参数提取] 提取到 {len(parameters)} 个参数 | 耗时: {parse_time}ms")

        # Step 4: 生成标注图片
        print("[API] [可视化] 生成标注图片...")
        annotate_start = time.time()
        annotated_img = create_annotated_image(
            processed_img, ocr_results, parameters
        )
        # 编码为 base64
        _, img_buffer = cv2.imencode(
            ".jpg", annotated_img, [cv2.IMWRITE_JPEG_QUALITY, 90]
        )
        annotated_b64 = base64.b64encode(img_buffer).decode("utf-8")
        annotate_time = round((time.time() - annotate_start) * 1000, 1)
        print(f"[API] [可视化] 标注图片生成完成 | 耗时: {annotate_time}ms")

        # 原图也转为 base64（用于前端对比展示）
        _, orig_buffer = cv2.imencode(
            ".jpg", processed_img, [cv2.IMWRITE_JPEG_QUALITY, 92]
        )
        original_b64 = base64.b64encode(orig_buffer).decode("utf-8")

        # 总耗时
        total_time = round((time.time() - total_start) * 1000, 1)
        print("="*70)
        print(f"✅ [完成] 总耗时: {total_time}ms (预处理{preprocess_time}ms + OCR{ocr_stats['ocr_time_ms']}ms + 提取{parse_time}ms + 可视化{annotate_time}ms)")
        print("="*70 + "\n")

        # 保存上传的原始图片（使用原文件名，重复则覆盖）
        original_filename = image.filename or "unknown.jpg"
        save_path = os.path.join(UPLOAD_DIR, original_filename)
        with open(save_path, "wb") as f:
            f.write(image_bytes)

        # 保存 OCR 识别原始文本到同名 .txt 文件
        stem = os.path.splitext(original_filename)[0]
        txt_path = os.path.join(UPLOAD_DIR, f"{stem}.txt")
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(f"# 文件: {original_filename}\n")
            f.write(f"# 识别时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"# 识别文字数: {ocr_stats['text_count']}\n")
            f.write("#" + "=" * 50 + "\n\n")
            for item in raw_texts:
                text = item.get("text", "") if isinstance(item, dict) else str(item)
                conf = item.get("confidence", "") if isinstance(item, dict) else ""
                if conf:
                    f.write(f"[{conf:.3f}] {text}\n")
                else:
                    f.write(f"{text}\n")

        # 构建响应体
        response_data = {
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
        }

        # 如果启用调试模式，添加诊断信息
        if debug:
            response_data["data"]["diagnostics"] = {
                "original_size": preprocess_info.get("original_size"),
                "processed_size": preprocess_info.get("processed_size"),
                "brightness_level": preprocess_info.get("brightness_level"),
                "contrast_level": preprocess_info.get("contrast_level"),
                "clarity_level": preprocess_info.get("clarity_level"),
                "is_screen_display": preprocess_info.get("is_screen_display"),
                "applied_enhancements": preprocess_info.get("applied_enhancements", []),
            }

        return JSONResponse(response_data)

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
    return {"status": "ok", "service": "SpotCheck"}


@app.get("/api/saved-params")
async def get_saved_params():
    """获取已保存的全局参数。"""
    global _saved_global_params
    if _saved_global_params:
        return {"success": True, "data": _saved_global_params}
    return {"success": True, "data": None}


@app.post("/api/save-params")
async def save_params(params: str = Form(...)):
    """将当前参数永久保存为全局默认值。"""
    global _saved_global_params
    try:
        parsed = json.loads(params)
        if save_params_to_file(parsed):
            _saved_global_params = parsed
            print(f"[API] [保存参数] 已永久保存: {list(parsed.keys())}")
            return {"success": True, "message": "参数已保存，后续所有图片将使用此参数"}
        return JSONResponse({"success": False, "message": "保存失败"}, status_code=500)
    except Exception as e:
        return JSONResponse({"success": False, "message": str(e)}, status_code=400)


@app.post("/api/auto-tune")
async def auto_tune(image: UploadFile = File(...)):
    """根据上传的图片自动分析并推荐最佳参数。"""
    global _saved_global_params
    try:
        image_bytes = await image.read()
        recommended = auto_tune_params(image_bytes, saved_params=_saved_global_params)
        print(f"[API] [自动调优] 完成，推荐参数: {recommended}")
        return {"success": True, "data": recommended}
    except Exception as e:
        print(f"[API] [自动调优] 失败: {e}")
        import traceback
        traceback.print_exc()
        return JSONResponse({"success": False, "message": str(e)}, status_code=500)


# ============================================================
# 前端静态文件服务
# ============================================================

# 前端首页
@app.get("/")
async def serve_index():
    index_path = os.path.join(FRONTEND_DIR, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return {"message": "SpotCheck API is running. Frontend not found."}


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
