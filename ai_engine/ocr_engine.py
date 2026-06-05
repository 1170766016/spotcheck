"""
OCR 识别引擎模块

封装 PaddleOCR，提供设备屏显图片的文字识别功能。
模型在首次调用时加载（单例），后续调用复用已加载的模型。
"""
import time
import numpy as np
from paddleocr import PaddleOCR

from config import OCR_CONFIG

# ============================================================
# OCR 引擎单例
# ============================================================
_ocr_instance: PaddleOCR | None = None
_model_load_time: float = 0


def get_ocr_engine() -> PaddleOCR:
    """获取 PaddleOCR 引擎实例（懒加载单例）。"""
    global _ocr_instance, _model_load_time

    if _ocr_instance is None:
        print("[OCR] 正在加载 PaddleOCR 模型（首次加载较慢）...")
        start = time.time()
        _ocr_instance = PaddleOCR(**OCR_CONFIG)
        _model_load_time = round((time.time() - start) * 1000, 1)
        print(f"[OCR] 模型加载完成，耗时 {_model_load_time}ms")

    return _ocr_instance


def recognize(image: np.ndarray) -> tuple[list, dict]:
    """
    对图像执行 OCR 识别。

    Args:
        image: OpenCV 格式的图像 (BGR, numpy array)

    Returns:
        (ocr_results, stats)
        - ocr_results: PaddleOCR 原始结果
          格式: [[ [box, (text, confidence)], ... ]]
          box: [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]
        - stats: {"ocr_time_ms", "text_count"}
    """
    engine = get_ocr_engine()

    start = time.time()
    results = engine.ocr(image)
    ocr_time = round((time.time() - start) * 1000, 1)

    # 标准化 OCR 结果，兼容 PP-OCRv5 / PaddleX 新格式和旧版格式
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

    # 空跑一张小图让引擎完全初始化
    dummy = np.zeros((64, 200, 3), dtype=np.uint8)
    # 在上面写一些文字以触发完整的 det+rec 流程
    import cv2
    cv2.putText(dummy, "warmup", (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    engine.ocr(dummy)
    print("[OCR] 引擎预热完成")
