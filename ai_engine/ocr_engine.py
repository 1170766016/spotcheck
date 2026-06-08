"""
OCR 识别引擎模块

封装 PaddleOCR，提供设备屏显图片的文字识别功能。
模型在首次调用时加载（单例），后续调用复用已加载的模型。
支持运行时动态调整 OCR 参数（检测阈值、输入尺寸、批大小等）。
"""
import time
import numpy as np
from paddleocr import PaddleOCR

from config import OCR_CONFIG, OCR_TUNING_DEFAULTS

# ============================================================
# OCR 引擎单例
# ============================================================
_ocr_instance: PaddleOCR | None = None
_model_load_time: float = 0
_current_overrides: dict = {}


def get_ocr_engine(overrides: dict = None) -> PaddleOCR:
    """
    获取 PaddleOCR 引擎实例（懒加载单例）。
    如果传入 overrides 且与当前覆盖参数不同，则重新创建实例。
    """
    global _ocr_instance, _model_load_time, _current_overrides

    # 检查是否需要重建引擎
    need_recreate = False
    if _ocr_instance is None:
        need_recreate = True
    elif overrides and overrides != _current_overrides:
        need_recreate = True

    if need_recreate:
        print("[OCR] 正在加载 PaddleOCR 模型（首次加载较慢）...")
        start = time.time()

        # 合并基础配置与运行时覆盖
        merged_config = {**OCR_CONFIG}
        if overrides:
            merged_config.update(overrides)
            _current_overrides = dict(overrides)
        else:
            _current_overrides = {}

        _ocr_instance = PaddleOCR(**merged_config)
        _model_load_time = round((time.time() - start) * 1000, 1)
        print(f"[OCR] 模型加载完成，耗时 {_model_load_time}ms")
        if overrides:
            print(f"[OCR] 应用运行时参数覆盖: {overrides}")

    return _ocr_instance


def update_runtime_config(ocr_params: dict):
    """
    更新 OCR 运行时参数。
    如果参数发生变化，下次调用 get_ocr_engine() 时会重建实例。
    """
    global _ocr_instance, _current_overrides

    if not ocr_params:
        return

    # 只接受已知的可调参数
    valid_overrides = {}
    for key in OCR_TUNING_DEFAULTS:
        if key in ocr_params:
            valid_overrides[key] = ocr_params[key]

    if valid_overrides and valid_overrides != _current_overrides:
        # 标记需要重建（下次 get_ocr_engine 时触发）
        _current_overrides = valid_overrides
        _ocr_instance = None  # 清除旧实例以触发重建
        print(f"[OCR] 运行时参数已更新，引擎将在下次识别时重建: {valid_overrides}")


def recognize(image: np.ndarray, ocr_params: dict = None) -> tuple[list, dict]:
    """
    对图像执行 OCR 识别。

    Args:
        image: OpenCV 格式的图像 (BGR, numpy array)
        ocr_params: 可选的运行时 OCR 参数覆盖

    Returns:
        (ocr_results, stats)
        - ocr_results: 标准化结果
          格式: [[ [box, (text, confidence)], ... ]]
          box: [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]
        - stats: {"ocr_time_ms", "text_count"}
    """
    # 如果有新的 OCR 参数，先更新配置（可能触发引擎重建）
    if ocr_params:
        update_runtime_config(ocr_params)

    engine = get_ocr_engine(ocr_params)

    start = time.time()
    results = list(engine.predict(image))
    ocr_time = round((time.time() - start) * 1000, 1)

    # 标准化 OCR 结果，兼容 PP-OCRv5 / PP-OCRv4 / PaddleX 新格式和旧版格式
    normalized_results = []
    text_count = 0
    
    if results:
        for page_res in results:
            page_lines = []
            if page_res is None:
                normalized_results.append([])
                continue
                
            has_dict_keys = False
            if hasattr(page_res, "get"):
                try:
                    has_dict_keys = "rec_texts" in page_res
                except:
                    pass

            if has_dict_keys or hasattr(page_res, "rec_texts"):
                if has_dict_keys:
                    texts = page_res.get("rec_texts", [])
                    scores = page_res.get("rec_scores", [])
                    polys = page_res.get("dt_polys", page_res.get("rec_polys", []))
                else:
                    texts = getattr(page_res, "rec_texts", [])
                    scores = getattr(page_res, "rec_scores", [])
                    polys = getattr(page_res, "dt_polys", getattr(page_res, "rec_polys", []))
                
                for box, text, score in zip(polys, texts, scores):
                    if hasattr(box, "tolist"):
                        box_list = box.tolist()
                    else:
                        box_list = list(box)
                    page_lines.append([box_list, (text, float(score))])
            else:
                # 已经是旧版的行列表格式
                page_lines = page_res
            
            normalized_results.append(page_lines)
        
        if normalized_results and normalized_results[0]:
            text_count = len(normalized_results[0])
    else:
        normalized_results = []

    stats = {
        "ocr_time_ms": ocr_time,
        "text_count": text_count,
    }

    return normalized_results, stats


def warmup():
    """
    预热 OCR 引擎（加载模型 + 空跑一次）。
    在服务启动时调用，避免第一个请求太慢。
    """
    engine = get_ocr_engine()

    # 空跑一张小图让引擎完全初始化（用最小尺寸减少预热时间）
    dummy = np.zeros((32, 100, 3), dtype=np.uint8)
    # 在上面写一些文字以触发完整的 det+rec 流程
    import cv2
    cv2.putText(dummy, "OK", (10, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    list(engine.predict(dummy))
    print("[OCR] 引擎预热完成")

