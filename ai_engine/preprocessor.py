"""
图像预处理模块

对上传的设备屏显照片进行预处理，提升 OCR 识别准确率和速度。
处理包括：EXIF 方向矫正、尺寸缩放、对比度增强、去噪等。
"""
import cv2
import numpy as np
from PIL import Image, ExifTags
import io
import time

from config import IMAGE_CONFIG


def preprocess_image(image_bytes: bytes) -> tuple[np.ndarray, dict]:
    """
    预处理图片，返回处理后的图像和处理信息。

    Args:
        image_bytes: 原始图片二进制数据

    Returns:
        (processed_image, info_dict)
        - processed_image: OpenCV 格式的图像 (BGR)
        - info_dict: 处理信息 {"original_size", "processed_size", "preprocess_time_ms"}
    """
    start_time = time.time()
    info = {}

    # Step 1: 解码图片并处理 EXIF 方向
    image = _decode_with_exif(image_bytes)
    info["original_size"] = f"{image.shape[1]}x{image.shape[0]}"

    # Step 2: 缩放到合理尺寸
    image = _resize_image(image, IMAGE_CONFIG["max_size"])

    # Step 3: 增强对比度
    if IMAGE_CONFIG["enhance_contrast"]:
        image = _enhance_contrast(image)

    # Step 4: 去噪
    if IMAGE_CONFIG["denoise"]:
        image = _denoise(image)

    info["processed_size"] = f"{image.shape[1]}x{image.shape[0]}"
    info["preprocess_time_ms"] = round((time.time() - start_time) * 1000, 1)

    return image, info


def _decode_with_exif(image_bytes: bytes) -> np.ndarray:
    """解码图片并根据 EXIF 信息矫正方向（手机拍照常有方向问题）。"""
    # 用 PIL 读取以正确处理 EXIF
    pil_image = Image.open(io.BytesIO(image_bytes))

    # 处理 EXIF 方向
    try:
        exif = pil_image._getexif()
        if exif:
            orientation_key = None
            for key, val in ExifTags.TAGS.items():
                if val == "Orientation":
                    orientation_key = key
                    break
            if orientation_key and orientation_key in exif:
                orientation = exif[orientation_key]
                if orientation == 3:
                    pil_image = pil_image.rotate(180, expand=True)
                elif orientation == 6:
                    pil_image = pil_image.rotate(270, expand=True)
                elif orientation == 8:
                    pil_image = pil_image.rotate(90, expand=True)
    except (AttributeError, KeyError):
        pass

    # 转换为 RGB（去掉 Alpha 通道等）
    if pil_image.mode != "RGB":
        pil_image = pil_image.convert("RGB")

    # PIL -> OpenCV (RGB -> BGR)
    cv_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    return cv_image


def _resize_image(image: np.ndarray, max_size: int) -> np.ndarray:
    """等比缩放，使最长边不超过 max_size。"""
    h, w = image.shape[:2]
    if max(h, w) <= max_size:
        return image

    scale = max_size / max(h, w)
    new_w = int(w * scale)
    new_h = int(h * scale)
    return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)


def _enhance_contrast(image: np.ndarray) -> np.ndarray:
    """使用 CLAHE 自适应直方图均衡增强对比度。"""
    # 转到 LAB 色彩空间，只增强亮度通道
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)

    clahe = cv2.createCLAHE(
        clipLimit=IMAGE_CONFIG["clahe_clip_limit"],
        tileGridSize=IMAGE_CONFIG["clahe_grid_size"],
    )
    l_enhanced = clahe.apply(l_channel)

    lab_enhanced = cv2.merge([l_enhanced, a_channel, b_channel])
    return cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)


def _denoise(image: np.ndarray) -> np.ndarray:
    """轻度去噪，保留文字边缘。"""
    return cv2.fastNlMeansDenoisingColored(image, None, 6, 6, 7, 21)


def create_annotated_image(
    image: np.ndarray,
    ocr_results: list,
    parameters: list,
) -> np.ndarray:
    """
    在原图上绘制识别结果标注。

    Args:
        image: 原始图像 (BGR)
        ocr_results: PaddleOCR 原始结果
        parameters: 解析后的参数列表

    Returns:
        标注后的图像
    """
    annotated = image.copy()

    if not ocr_results or not ocr_results[0]:
        return annotated

    # 收集已匹配的文本（用于颜色区分）
    matched_texts = set()
    for param in parameters:
        if param.get("name"):
            matched_texts.add(param["name"])
        raw_value = param.get("value", "")
        unit = param.get("unit", "")
        if raw_value:
            matched_texts.add(raw_value)
            if unit:
                matched_texts.add(raw_value + unit)
                matched_texts.add(raw_value + " " + unit)

    for line in ocr_results[0]:
        box = line[0]
        text = line[1][0]
        confidence = line[1][1]

        # 转换为整数坐标
        pts = np.array(box, dtype=np.int32)

        # 根据匹配状态选择颜色
        is_matched = any(t in text or text in t for t in matched_texts)

        if is_matched:
            color = (0, 220, 180)   # 青绿色 - 已匹配
            thickness = 2
        else:
            color = (128, 128, 128)  # 灰色 - 未匹配
            thickness = 1

        # 绘制多边形边框
        cv2.polylines(annotated, [pts], True, color, thickness)

        # 绘制文本标签背景
        x_min = int(min(p[0] for p in box))
        y_min = int(min(p[1] for p in box))
        label = f"{text} ({confidence:.0%})"

        # 计算文本大小
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.45
        (label_w, label_h), baseline = cv2.getTextSize(
            label, font, font_scale, 1
        )

        # 绘制标签背景
        label_y = max(y_min - 4, label_h + 4)
        cv2.rectangle(
            annotated,
            (x_min, label_y - label_h - 4),
            (x_min + label_w + 4, label_y + 2),
            color,
            -1,
        )
        # 绘制标签文字
        cv2.putText(
            annotated, label,
            (x_min + 2, label_y - 2),
            font, font_scale, (0, 0, 0), 1, cv2.LINE_AA,
        )

    return annotated
