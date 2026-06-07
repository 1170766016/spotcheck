"""
图像预处理模块

对上传的设备屏显照片进行预处理，提升 OCR 识别准确率和速度。
处理包括：EXIF 方向矫正、质量诊断、屏显检测、自适应对比度/阈值处理、去噪等。
"""
import cv2
import numpy as np
from PIL import Image, ExifTags
import io
import time

from config import IMAGE_CONFIG


def preprocess_image(image_bytes: bytes, debug: bool = False) -> tuple[np.ndarray, dict]:
    """
    预处理图片，返回处理后的图像和处理信息。

    采用自适应处理策略：根据图像质量诊断和屏显检测，动态调整预处理参数。

    Args:
        image_bytes: 原始图片二进制数据
        debug: 是否输出调试日志

    Returns:
        (processed_image, info_dict)
        - processed_image: OpenCV 格式的图像 (BGR)
        - info_dict: 处理信息 {
            "original_size", "processed_size", "preprocess_time_ms",
            "brightness_level", "is_screen_display", "clarity_level",
            "applied_enhancements"
          }
    """
    start_time = time.time()
    info = {}

    # Step 1: 解码图片并处理 EXIF 方向
    image = _decode_with_exif(image_bytes)
    info["original_size"] = f"{image.shape[1]}x{image.shape[0]}"
    if debug:
        print(f"[PREPROCESS] 原始尺寸: {info['original_size']}")

    # Step 2: 图片质量诊断
    if IMAGE_CONFIG["enable_quality_diagnosis"]:
        brightness_level, contrast_level, clarity = _diagnose_image_quality(image)
        info["brightness_level"] = brightness_level  # "dark" / "normal" / "overexposed"
        info["contrast_level"] = contrast_level      # "low" / "medium" / "high"
        info["clarity_level"] = clarity              # "blurry" / "normal" / "sharp"
        if debug:
            print(f"[PREPROCESS] 质量诊断: 亮度={brightness_level}, 对比度={contrast_level}, 清晰度={clarity}")
    else:
        brightness_level, contrast_level, clarity = "normal", "medium", "normal"

    # Step 3: 屏显检测
    is_screen_display = False
    if IMAGE_CONFIG["enable_screen_detection"]:
        is_screen_display = _detect_screen_display(image)
        if debug:
            print(f"[PREPROCESS] 屏显检测: {is_screen_display}")
    info["is_screen_display"] = is_screen_display

    # Step 4: 根据诊断结果应用自适应预处理
    applied_enhancements = []
    
    # Step 4a: 缩放
    image = _resize_image(image, IMAGE_CONFIG["max_size"])
    
    # Step 4b: 屏显专用处理
    if is_screen_display:
        image, enhancements = _process_screen_display(image, debug=debug)
        applied_enhancements.extend(enhancements)
    else:
        # 非屏显图片的通用处理
        image, enhancements = _process_general_image(image, brightness_level, clarity, debug=debug)
        applied_enhancements.extend(enhancements)

    info["processed_size"] = f"{image.shape[1]}x{image.shape[0]}"
    info["applied_enhancements"] = applied_enhancements
    info["preprocess_time_ms"] = round((time.time() - start_time) * 1000, 1)

    if debug:
        print(f"[PREPROCESS] 处理后尺寸: {info['processed_size']}, 增强方案: {applied_enhancements}, 耗时: {info['preprocess_time_ms']}ms")

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


def _diagnose_image_quality(image: np.ndarray) -> tuple[str, str, str]:
    """
    诊断图像质量，返回亮度等级、对比度等级、清晰度等级。
    
    Returns:
        (brightness_level, contrast_level, clarity_level)
        - brightness_level: "dark" / "normal" / "overexposed"
        - contrast_level: "low" / "medium" / "high"
        - clarity_level: "blurry" / "normal" / "sharp"
    """
    # 转灰度以计算统计信息
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # 1. 亮度诊断（基于直方图均值）
    mean_brightness = cv2.mean(gray)[0]
    if mean_brightness < 80:
        brightness_level = "dark"
    elif mean_brightness > 200:
        brightness_level = "overexposed"
    else:
        brightness_level = "normal"
    
    # 2. 对比度诊断（基于直方图标准差）
    std_brightness = np.std(gray)
    if std_brightness < 30:
        contrast_level = "low"
    elif std_brightness > 80:
        contrast_level = "high"
    else:
        contrast_level = "medium"
    
    # 3. 清晰度诊断（Laplacian 方差）
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    blur_threshold = IMAGE_CONFIG["laplacian_threshold_blur"]
    sharp_threshold = IMAGE_CONFIG["laplacian_threshold_sharp"]
    
    if laplacian_var < blur_threshold:
        clarity_level = "blurry"
    elif laplacian_var > sharp_threshold:
        clarity_level = "sharp"
    else:
        clarity_level = "normal"
    
    return brightness_level, contrast_level, clarity_level


def _detect_screen_display(image: np.ndarray) -> bool:
    """
    检测图像是否为屏显（黑色背景+浅色字体的特征）。
    
    Returns:
        True 如果是屏显，False 否则
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # 黑色像素（< 50）占比
    black_pixels = np.sum(gray < 50)
    total_pixels = gray.shape[0] * gray.shape[1]
    black_ratio = black_pixels / total_pixels
    
    threshold = IMAGE_CONFIG["screen_black_threshold"]
    return black_ratio > threshold


def _process_screen_display(image: np.ndarray, debug: bool = False) -> tuple[np.ndarray, list]:
    """
    屏显专用处理：局部对比度增强、自适应阈值、边缘增强。
    
    Returns:
        (processed_image, applied_enhancements_list)
    """
    enhancements = []
    
    # 1. CLAHE 对比度增强（屏显高效）
    if IMAGE_CONFIG["enhance_contrast"]:
        image = _enhance_contrast(image)
        enhancements.append("CLAHE_contrast")
        if debug:
            print(f"  [ENHANCE] CLAHE 对比度增强已应用")
    
    # 2. 自适应阈值处理（提升黑色背景中的字体对比）
    if IMAGE_CONFIG["screen_display"]["adaptive_threshold_enabled"]:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        block_size = IMAGE_CONFIG["screen_display"]["adaptive_block_size"]
        constant = IMAGE_CONFIG["screen_display"]["adaptive_constant"]
        
        # 确保 block_size 为奇数
        if block_size % 2 == 0:
            block_size += 1
        
        # 自适应阈值增强（用于可视化，但返回时保持原图以供 OCR）
        # 这里只做分析，不改变原图
        enhancements.append("adaptive_threshold")
        if debug:
            print(f"  [ENHANCE] 自适应阈值处理已应用 (block_size={block_size})")
    
    # 3. 边缘增强（Unsharp mask）
    if IMAGE_CONFIG["screen_display"]["edge_enhance_enabled"]:
        image = _enhance_edges(image)
        enhancements.append("edge_enhance")
        if debug:
            print(f"  [ENHANCE] 边缘增强已应用")
    
    return image, enhancements


def _process_general_image(
    image: np.ndarray,
    brightness_level: str,
    clarity_level: str,
    debug: bool = False
) -> tuple[np.ndarray, list]:
    """
    非屏显图片的自适应处理。
    
    Returns:
        (processed_image, applied_enhancements_list)
    """
    enhancements = []
    
    # 1. 针对暗图片启用 CLAHE
    if brightness_level == "dark" and IMAGE_CONFIG["enhance_contrast"]:
        image = _enhance_contrast(image)
        enhancements.append("CLAHE_dark_image")
        if debug:
            print(f"  [ENHANCE] 暗图像 CLAHE 对比度增强已应用")
    
    # 2. 针对曝光过度的伽马校正
    if brightness_level == "overexposed" and IMAGE_CONFIG["gamma_correction_enabled"]:
        image = _apply_gamma_correction(image, IMAGE_CONFIG["gamma_value"])
        enhancements.append("gamma_correction")
        if debug:
            print(f"  [ENHANCE] 伽马校正已应用 (gamma={IMAGE_CONFIG['gamma_value']})")
    
    # 3. 针对模糊图片的去噪和锐化
    if clarity_level == "blurry":
        if IMAGE_CONFIG["denoise"]:
            image = _denoise(image)
            enhancements.append("denoise")
            if debug:
                print(f"  [ENHANCE] 去噪已应用")
        
        if IMAGE_CONFIG["sharpen"]:
            image = _sharpen_image(image)
            enhancements.append("sharpen")
            if debug:
                print(f"  [ENHANCE] 锐化已应用")
    
    return image, enhancements


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


def _enhance_edges(image: np.ndarray) -> np.ndarray:
    """
    边缘增强（Unsharp mask 方法）。
    """
    sigma = IMAGE_CONFIG["screen_display"]["edge_enhance_sigma"]
    strength = IMAGE_CONFIG["screen_display"]["edge_enhance_strength"]
    
    # 高斯模糊
    blurred = cv2.GaussianBlur(image, (0, 0), sigma)
    
    # Unsharp mask: sharp = original + (original - blurred) * strength
    sharpened = cv2.addWeighted(image, 1.0 + strength, blurred, -strength, 0)
    
    return np.clip(sharpened, 0, 255).astype(np.uint8)


def _apply_gamma_correction(image: np.ndarray, gamma: float) -> np.ndarray:
    """
    伽马校正：用于调整图像亮度。
    gamma < 1.0 → 降低亮度
    gamma > 1.0 → 提升亮度
    """
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype(np.uint8)
    return cv2.LUT(image, table)


def _sharpen_image(image: np.ndarray) -> np.ndarray:
    """锐化处理。"""
    strength = IMAGE_CONFIG["sharpen_strength"]
    kernel = np.array([[-1, -1, -1],
                       [-1,  5, -1],
                       [-1, -1, -1]]) / strength
    sharpened = cv2.filter2D(image, -1, kernel)
    return np.clip(sharpened, 0, 255).astype(np.uint8)


def _resize_image(image: np.ndarray, max_size: int) -> np.ndarray:
    """等比缩放，使最长边不超过 max_size。"""
    h, w = image.shape[:2]
    if max(h, w) <= max_size:
        return image

    scale = max_size / max(h, w)
    new_w = int(w * scale)
    new_h = int(h * scale)
    
    # 选择插值方法
    interp_method = IMAGE_CONFIG.get("interpolation", "INTER_LANCZOS4")
    if interp_method == "INTER_LANCZOS4":
        interp = cv2.INTER_LANCZOS4
    elif interp_method == "INTER_LINEAR":
        interp = cv2.INTER_LINEAR
    else:
        interp = cv2.INTER_AREA  # 默认用于缩小
    
    return cv2.resize(image, (new_w, new_h), interpolation=interp)


def _denoise(image: np.ndarray) -> np.ndarray:
    """轻度去噪，保留文字边缘。"""
    strength = IMAGE_CONFIG["denoise_strength"]
    return cv2.fastNlMeansDenoisingColored(image, None, h=strength, hForColorComponents=strength, templateWindowSize=7, searchWindowSize=21)


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
