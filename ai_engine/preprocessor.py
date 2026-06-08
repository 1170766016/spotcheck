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

from config import IMAGE_CONFIG, OCR_TUNING_DEFAULTS, IMAGE_TUNING_DEFAULTS


# ============================================================
# 已保存的用户参数文件路径
# ============================================================
import os
SAVED_PARAMS_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "saved_params.json")


def load_saved_params() -> dict | None:
    """从 saved_params.json 加载已保存的用户参数。"""
    import json
    if os.path.exists(SAVED_PARAMS_PATH):
        try:
            with open(SAVED_PARAMS_PATH, "r", encoding="utf-8") as f:
                params = json.load(f)
            print(f"[PREPROCESS] 已加载保存的参数文件: {SAVED_PARAMS_PATH}")
            return params
        except Exception as e:
            print(f"[PREPROCESS] 加载参数文件失败: {e}")
    return None


def save_params_to_file(params: dict) -> bool:
    """将参数保存到 saved_params.json。"""
    import json
    try:
        with open(SAVED_PARAMS_PATH, "w", encoding="utf-8") as f:
            json.dump(params, f, ensure_ascii=False, indent=2)
        print(f"[PREPROCESS] 参数已保存到: {SAVED_PARAMS_PATH}")
        return True
    except Exception as e:
        print(f"[PREPROCESS] 保存参数失败: {e}")
        return False


def auto_tune_params(image_bytes: bytes, saved_params: dict = None) -> dict:
    """
    根据图片内容自动分析并推荐最佳预处理参数。
    采用“处理 → 评估 → 调整”策略：直接在内部处理图片，
    评估处理后的文字可读性质量，不够好就迭代调整参数。

    Args:
        image_bytes: 原始图片二进制数据
        saved_params: 已保存的用户参数（作为基础，在其上做微调）

    Returns:
        推荐的参数字典
    """
    import copy

    params = copy.deepcopy(saved_params) if saved_params else {}

    # 解码图片
    image = _decode_with_exif(image_bytes)
    gray_orig = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    h, w = gray_orig.shape[:2]
    orig_max = max(h, w)

    # ================================================================
    # 基础图像分析
    # ================================================================
    black_ratio = np.sum(gray_orig < 50) / (h * w)
    white_ratio = np.sum(gray_orig > 200) / (h * w)
    mean_brightness = cv2.mean(gray_orig)[0]

    # Sobel 边缘强度（模糊检测）
    sobelx = cv2.Sobel(gray_orig, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray_orig, cv2.CV_64F, 0, 1, ksize=3)
    edge_strength = np.mean(np.sqrt(sobelx**2 + sobely**2))

    # 判断图片类型
    is_dark_screen = black_ratio > 0.3
    is_mostly_dark = black_ratio > 0.6

    print(f"[AUTO-TUNE] 原图: 亮度={mean_brightness:.1f}, 黑底={black_ratio:.2f}, "
          f"白字={white_ratio:.2f}, 边缘={edge_strength:.1f}, 尺寸={w}x{h}")

    # ================================================================
    # 质量评估函数：评估处理后的图片对 OCR 的友好程度
    # 分数越高越好
    # ================================================================
    def evaluate_ocr_readability(img_gray):
        """评估灰度图的 OCR 可读性（0~100 分）。"""
        h2, w2 = img_gray.shape
        
        # 1. Otsu 二值化质量：类间方差越大 = 文字背景分离越好
        _, binary = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        fg_ratio = np.sum(binary > 128) / (h2 * w2)
        # 理想文字占比 5%~40%，太少或太多都不好
        text_quality = max(0, 1.0 - abs(fg_ratio - 0.2) * 3) if fg_ratio < 0.6 else 0.3
        
        # 2. 文字边缘清晰度
        edges = cv2.Canny(img_gray, 50, 150)
        edge_density = np.sum(edges > 0) / (h2 * w2)
        # 理想边缘密度 2%~15%
        edge_quality = min(1.0, edge_density * 10) if edge_density < 0.2 else max(0.3, 1.0 - (edge_density - 0.2) * 5)
        
        # 3. 对比度：文字和背景的亮度差异
        # 取 Otsu 阈值两侧的平均亮度差
        thresh_val = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[0]
        bg_mean = np.mean(img_gray[img_gray <= thresh_val]) if np.sum(img_gray <= thresh_val) > 100 else 0
        fg_mean = np.mean(img_gray[img_gray > thresh_val]) if np.sum(img_gray > thresh_val) > 100 else 255
        contrast = (fg_mean - bg_mean) / 255.0  # 0~1
        
        # 综合分数
        score = (text_quality * 30 + edge_quality * 30 + contrast * 40)
        return score, {"text_quality": text_quality, "edge_quality": edge_quality, 
                       "contrast": contrast, "fg_ratio": fg_ratio, "edge_density": edge_density}

    # ================================================================
    # 第一阶段：确定基础处理（反相）
    # ================================================================
    # 先评估原图
    base_score, base_metrics = evaluate_ocr_readability(gray_orig)
    print(f"[AUTO-TUNE] 原图评估: 分数={base_score:.1f}, {base_metrics}")

    best_params = dict(params)
    best_score = base_score

    # 尝试全局反相
    if is_dark_screen:
        gray_inverted = 255 - gray_orig
        inv_score, inv_metrics = evaluate_ocr_readability(gray_inverted)
        print(f"[AUTO-TUNE] 全局反相评估: 分数={inv_score:.1f}, {inv_metrics}")
        if inv_score > best_score:
            best_params["invert_mode"] = "invert_black_bg"
            best_score = inv_score
            print(f"[AUTO-TUNE] ✓ 全局反相提升 {inv_score - base_score:.1f} 分")
        else:
            # 尝试局部反相
            ksize = max(21, (min(h, w) // 10) | 1)
            local_mean = cv2.boxFilter(gray_orig, -1, (ksize, ksize))
            mask = local_mean < 80
            gray_local = gray_orig.copy()
            gray_local[mask] = 255 - gray_orig[mask]
            local_score, local_metrics = evaluate_ocr_readability(gray_local)
            print(f"[AUTO-TUNE] 局部反相评估: 分数={local_score:.1f}, {local_metrics}")
            if local_score > best_score:
                best_params["invert_mode"] = "local_black_bg"
                best_params["invert_black_bg_thresh"] = 80
                best_score = local_score
                print(f"[AUTO-TUNE] ✓ 局部反相提升 {local_score - base_score:.1f} 分")
            else:
                best_params["invert_mode"] = "none"
                print(f"[AUTO-TUNE] ✗ 反相未提升，保持原样")
    else:
        best_params["invert_mode"] = "none"

    # 应用当前最佳反相，得到基础图
    test_image = image.copy()
    invert_mode = best_params.get("invert_mode", "none")
    if invert_mode == "invert_black_bg" and is_mostly_dark:
        test_image = 255 - test_image
    elif invert_mode == "local_black_bg":
        gray_test = cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY)
        ksize = max(21, (min(h, w) // 10) | 1)
        local_mean = cv2.boxFilter(gray_test, -1, (ksize, ksize))
        mask = local_mean < 80
        for c in range(3):
            test_image[:, :, c] = np.where(mask, 255 - test_image[:, :, c], test_image[:, :, c])

    test_gray = cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY)
    current_score, _ = evaluate_ocr_readability(test_gray)
    print(f"[AUTO-TUNE] 反相后分数: {current_score:.1f}")

    # ================================================================
    # 第二阶段：尝试 CLAHE（对屏显几乎总是有益）
    # ================================================================
    clahe_candidates = [
        {"clip": 2.0, "grid": 16},
        {"clip": 2.5, "grid": 24},
        {"clip": 3.0, "grid": 24},
        {"clip": 3.5, "grid": 32},
        {"clip": 4.0, "grid": 32},
    ]
    
    best_clahe_score = current_score
    best_clahe = None

    for candidate in clahe_candidates:
        lab = cv2.cvtColor(test_image, cv2.COLOR_BGR2LAB)
        l_ch, a_ch, b_ch = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=candidate["clip"], tileGridSize=(candidate["grid"], candidate["grid"]))
        l_enhanced = clahe.apply(l_ch)
        enhanced = cv2.merge([l_enhanced, a_ch, b_ch])
        enhanced_bgr = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
        enhanced_gray = cv2.cvtColor(enhanced_bgr, cv2.COLOR_BGR2GRAY)
        
        score, metrics = evaluate_ocr_readability(enhanced_gray)
        print(f"[AUTO-TUNE] CLAHE(clip={candidate['clip']},grid={candidate['grid']}): "
              f"分数={score:.1f} {'✓' if score > best_clahe_score else '✗'}")
        
        if score > best_clahe_score + 1.0:  # 至少提升 1 分才认为有意义
            best_clahe_score = score
            best_clahe = candidate

    if best_clahe:
        best_params["contrast_mode"] = "clahe"
        best_params["clahe_clip_limit"] = best_clahe["clip"]
        best_params["clahe_grid_size"] = best_clahe["grid"]
        current_score = best_clahe_score
        print(f"[AUTO-TUNE] ✓ 最佳 CLAHE: clip={best_clahe['clip']}, grid={best_clahe['grid']}, 分数={best_clahe_score:.1f}")
    else:
        best_params["contrast_mode"] = "none"
        print(f"[AUTO-TUNE] ✗ CLAHE 未提升分数，保持原样")

    # ================================================================
    # 第三阶段：尝试锐化（仅在边缘较弱时）
    # ================================================================
    if edge_strength < 15:
        sharpen_candidates = [
            {"mode": "laplacian", "strength": 0.8},
            {"mode": "laplacian", "strength": 1.2},
            {"mode": "unsharp_mask", "strength": 1.0},
        ]
        
        # 在当前最佳处理基础上尝试锐化
        # 重建当前最佳图像
        curr_best = test_image.copy()
        if best_clahe:
            lab = cv2.cvtColor(curr_best, cv2.COLOR_BGR2LAB)
            l_ch, a_ch, b_ch = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=best_clahe["clip"], tileGridSize=(best_clahe["grid"], best_clahe["grid"]))
            l_enhanced = clahe.apply(l_ch)
            curr_best = cv2.cvtColor(cv2.merge([l_enhanced, a_ch, b_ch]), cv2.COLOR_LAB2BGR)
        
        curr_gray = cv2.cvtColor(curr_best, cv2.COLOR_BGR2GRAY)
        curr_score, _ = evaluate_ocr_readability(curr_gray)
        best_sharpen_score = curr_score
        best_sharpen = None
        
        for candidate in sharpen_candidates:
            test_img = curr_best.copy()
            if candidate["mode"] == "laplacian":
                laplacian = cv2.Laplacian(test_img, cv2.CV_64F)
                sharpened = test_img - candidate["strength"] * laplacian
                test_img = np.clip(sharpened, 0, 255).astype(np.uint8)
            elif candidate["mode"] == "unsharp_mask":
                blurred = cv2.GaussianBlur(test_img, (5, 5), 1.0)
                sharpened = cv2.addWeighted(test_img, 1.0 + candidate["strength"], blurred, -candidate["strength"], 0)
                test_img = np.clip(sharpened, 0, 255).astype(np.uint8)
            
            test_gray = cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)
            score, _ = evaluate_ocr_readability(test_gray)
            print(f"[AUTO-TUNE] 锐化({candidate['mode']},str={candidate['strength']}): "
                  f"分数={score:.1f} {'✓' if score > best_sharpen_score else '✗'}")
            
            if score > best_sharpen_score + 0.5:
                best_sharpen_score = score
                best_sharpen = candidate
        
        if best_sharpen:
            best_params["sharpen_enabled"] = True
            best_params["sharpen_mode"] = best_sharpen["mode"]
            best_params["sharpen_strength"] = best_sharpen["strength"]
            current_score = best_sharpen_score
            print(f"[AUTO-TUNE] ✓ 锐化提升: {best_sharpen['mode']}, 分数={best_sharpen_score:.1f}")
        else:
            best_params["sharpen_enabled"] = False
    else:
        best_params["sharpen_enabled"] = False

    # ================================================================
    # 第四阶段：设置保守的固定参数
    # ================================================================
    best_params["gamma_enabled"] = False        # 自动模式不开伽马（风险太高）
    best_params["deblur_enabled"] = False        # 去重影容易过度处理
    best_params["denoise_enabled"] = False       # 去噪容易模糊文字
    best_params["grayscale_enabled"] = False
    best_params["morphology_enabled"] = False
    best_params["screen_black_threshold"] = 0.4

    # 分辨率
    if orig_max > 2000:
        best_params["max_size"] = 0
    elif orig_max > 1200:
        best_params["max_size"] = 1280
    else:
        best_params["max_size"] = 0

    final_score = current_score
    improvement = final_score - base_score
    print(f"[AUTO-TUNE] 最终: 分数 {base_score:.1f} → {final_score:.1f} (提升 {improvement:.1f}), 参数: {best_params}")
    return best_params


def _get_merged_config(params: dict = None) -> dict:
    """合并默认配置与传入的参数。"""
    import copy
    cfg = copy.deepcopy(IMAGE_CONFIG)
    if not params:
        return cfg
    
    # 细致覆盖
    for k, v in params.items():
        if isinstance(v, dict) and k in cfg and isinstance(cfg[k], dict):
            cfg[k].update(v)
        else:
            cfg[k] = v
    return cfg


def preprocess_image(image_bytes: bytes, debug: bool = False, params: dict = None) -> tuple[np.ndarray, dict]:
    """
    预处理图片，返回处理后的图像和处理信息。

    采用自适应处理策略：根据图像质量诊断和屏显检测，动态调整预处理参数。
    如果提供了 params 自定义参数，则启用自定义调优管道。

    Args:
        image_bytes: 原始图片二进制数据
        debug: 是否输出调试日志
        params: 前端传入的自定义调优参数

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

    # Step 2: 合并配置
    cfg = _get_merged_config(params)

    # Step 3: 图片质量诊断
    brightness_level, contrast_level, clarity = _diagnose_image_quality(image, cfg)
    info["brightness_level"] = brightness_level  # "dark" / "normal" / "overexposed"
    info["contrast_level"] = contrast_level      # "low" / "medium" / "high"
    info["clarity_level"] = clarity              # "blurry" / "normal" / "sharp"
    if debug:
        print(f"[PREPROCESS] 质量诊断: 亮度={brightness_level}, 对比度={contrast_level}, 清晰度={clarity}")

    # Step 3b: 屏显检测
    is_screen_display = _detect_screen_display(image, cfg)
    if debug:
        print(f"[PREPROCESS] 屏显检测: {is_screen_display}")
    info["is_screen_display"] = is_screen_display

    # Step 4: 缩放（max_size=0 表示不缩放，使用原图分辨率）
    if cfg["max_size"] > 0:
        image = _resize_image(image, cfg["max_size"], cfg.get("interpolation", "INTER_LANCZOS4"))
        if debug:
            print(f"[PREPROCESS] 已缩放到最长边 {cfg['max_size']}px")
    else:
        if debug:
            print(f"[PREPROCESS] max_size=0，保持原图分辨率: {image.shape[1]}x{image.shape[0]}")

    applied_enhancements = []
    
    # 检测是否启用了自定义预处理
    use_custom_pipeline = False
    if params:
        custom_keys = {"grayscale_enabled", "invert_mode", "deblur_enabled", 
                       "sharpen_enabled", "contrast_mode", "denoise_enabled", 
                       "gamma_enabled", "morphology_enabled"}
        if any(k in params for k in custom_keys):
            use_custom_pipeline = True

    if use_custom_pipeline:
        if debug:
            print("[PREPROCESS] 启用自定义图像参数调优管道...")
        image, enhancements = _apply_custom_pipeline(image, cfg, debug=debug)
        applied_enhancements.extend(enhancements)
    else:
        # Step 4b: 屏显专用处理
        if is_screen_display:
            image, enhancements = _process_screen_display(image, cfg, debug=debug)
            applied_enhancements.extend(enhancements)
        else:
            # 非屏显图片的通用处理
            image, enhancements = _process_general_image(image, brightness_level, clarity, cfg, debug=debug)
            applied_enhancements.extend(enhancements)

    info["processed_size"] = f"{image.shape[1]}x{image.shape[0]}"
    info["applied_enhancements"] = applied_enhancements
    info["preprocess_time_ms"] = round((time.time() - start_time) * 1000, 1)

    if debug:
        print(f"[PREPROCESS] 处理后尺寸: {info['processed_size']}, 增强方案: {applied_enhancements}, 耗时: {info['preprocess_time_ms']}ms")

    return image, info


def _apply_custom_pipeline(image: np.ndarray, cfg: dict, debug: bool = False) -> tuple[np.ndarray, list]:
    """应用自定义调试图像预处理管道。"""
    applied = []
    
    # 1. 灰度化
    if cfg.get("grayscale_enabled", False):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = cv2.merge([gray, gray, gray])
        applied.append("grayscale")
        if debug:
            print("  [CUSTOM] 已应用灰度化")
            
    # 2. 对比度增强
    contrast_mode = cfg.get("contrast_mode", "none")
    if contrast_mode == "clahe":
        clip_limit = cfg.get("clahe_clip_limit", 2.5)
        grid_size = int(cfg.get("clahe_grid_size", 32))
        grid_size = max(2, grid_size)
        # 转到 LAB 色彩空间，只增强亮度通道
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(grid_size, grid_size))
        l_enhanced = clahe.apply(l_channel)
        lab_enhanced = cv2.merge([l_enhanced, a_channel, b_channel])
        image = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)
        applied.append(f"clahe_contrast(clip={clip_limit},grid={grid_size})")
        if debug:
            print(f"  [CUSTOM] 已应用 CLAHE 对比度增强 (clip={clip_limit}, grid={grid_size})")
    elif contrast_mode == "linear":
        alpha = cfg.get("linear_alpha", 1.0)
        beta = cfg.get("linear_beta", 0.0)
        image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
        applied.append(f"linear_contrast(alpha={alpha},beta={beta})")
        if debug:
            print(f"  [CUSTOM] 已应用线性对比度 (alpha={alpha}, beta={beta})")

    # 3. 伽马校正
    if cfg.get("gamma_enabled", False):
        gamma = cfg.get("gamma_value", 1.0)
        inv_gamma = 1.0 / max(0.1, gamma)
        table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype(np.uint8)
        image = cv2.LUT(image, table)
        applied.append(f"gamma(val={gamma})")
        if debug:
            print(f"  [CUSTOM] 已应用伽马校正 (gamma={gamma})")

    # 4. 去重影 / 去模糊 (Anti-Shake Directional Unsharp Mask)
    if cfg.get("deblur_enabled", False):
        direction = cfg.get("deblur_direction", "both")
        strength = cfg.get("deblur_strength", 1.0)
        h, w = image.shape[:2]
        if direction == "horizontal":
            ksize_w = max(3, (w // 100) | 1)
            # 水平方向高斯模糊
            blurred = cv2.GaussianBlur(image, (ksize_w, 1), 0)
            sharpened = cv2.addWeighted(image, 1.0 + strength, blurred, -strength, 0)
            image = np.clip(sharpened, 0, 255).astype(np.uint8)
            applied.append(f"deblur_horizontal(k={ksize_w},s={strength})")
        elif direction == "vertical":
            ksize_h = max(3, (h // 100) | 1)
            # 垂直方向高斯模糊
            blurred = cv2.GaussianBlur(image, (1, ksize_h), 0)
            sharpened = cv2.addWeighted(image, 1.0 + strength, blurred, -strength, 0)
            image = np.clip(sharpened, 0, 255).astype(np.uint8)
            applied.append(f"deblur_vertical(k={ksize_h},s={strength})")
        else: # both
            ksize = max(3, (min(h, w) // 100) | 1)
            blurred = cv2.GaussianBlur(image, (ksize, ksize), 0)
            sharpened = cv2.addWeighted(image, 1.0 + strength, blurred, -strength, 0)
            image = np.clip(sharpened, 0, 255).astype(np.uint8)
            applied.append(f"deblur_both(k={ksize},s={strength})")
        if debug:
            print(f"  [CUSTOM] 已应用抖动去模糊 (dir={direction}, str={strength})")

    # 5. 锐化增强
    if cfg.get("sharpen_enabled", False):
        mode = cfg.get("sharpen_mode", "laplacian")
        strength = cfg.get("sharpen_strength", 1.0)
        if mode == "laplacian":
            laplacian = cv2.Laplacian(image, cv2.CV_64F)
            sharpened = image - strength * laplacian
            image = np.clip(sharpened, 0, 255).astype(np.uint8)
            applied.append(f"sharpen_laplacian(s={strength})")
        elif mode == "unsharp_mask":
            blurred = cv2.GaussianBlur(image, (5, 5), 1.0)
            sharpened = cv2.addWeighted(image, 1.0 + strength, blurred, -strength, 0)
            image = np.clip(sharpened, 0, 255).astype(np.uint8)
            applied.append(f"sharpen_unsharp_mask(s={strength})")
        elif mode == "kernel":
            identity = np.array([[0, 0, 0],
                                 [0, 1, 0],
                                 [0, 0, 0]])
            kernel = np.array([[-1, -1, -1],
                               [-1,  9, -1],
                               [-1, -1, -1]])
            kernel_adj = identity * (1.0 - strength) + kernel * strength
            sharpened = cv2.filter2D(image, -1, kernel_adj)
            image = np.clip(sharpened, 0, 255).astype(np.uint8)
            applied.append(f"sharpen_kernel(s={strength})")
        if debug:
            print(f"  [CUSTOM] 已应用锐化 (mode={mode}, str={strength})")

    # 6. 自适应去噪 (双边滤波)
    if cfg.get("denoise_enabled", False):
        strength = cfg.get("denoise_strength", 15)
        # 双边滤波以保留边缘
        image = cv2.bilateralFilter(image, 9, strength * 2, strength * 2)
        applied.append(f"denoise_bilateral(s={strength})")
        if debug:
            print(f"  [CUSTOM] 已应用双边去噪 (str={strength})")

    # 7. 形态学处理
    if cfg.get("morphology_enabled", False):
        m_type = cfg.get("morphology_type", "open")
        m_size = int(cfg.get("morphology_size", 2))
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (m_size, m_size))
        if m_type == "dilate":
            image = cv2.dilate(image, kernel)
        elif m_type == "erode":
            image = cv2.erode(image, kernel)
        elif m_type == "open":
            image = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
        elif m_type == "close":
            image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
        applied.append(f"morphology_{m_type}(size={m_size})")
        if debug:
            print(f"  [CUSTOM] 已应用形态学处理 (type={m_type}, size={m_size})")

    # 8. 反相处理 (Invert Colors)
    invert_mode = cfg.get("invert_mode", "none")
    if invert_mode == "always":
        image = 255 - image
        applied.append("invert_always")
        if debug:
            print("  [CUSTOM] 已应用强制反相")
    elif invert_mode == "invert_black_bg":
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        black_pixels = np.sum(gray < 50)
        black_ratio = black_pixels / (gray.shape[0] * gray.shape[1])
        if black_ratio > cfg.get("screen_black_threshold", 0.4):
            image = 255 - image
            applied.append("invert_auto_black_bg")
            if debug:
                print(f"  [CUSTOM] 检测到黑底比例 {black_ratio:.2f} > 阈值，已应用自动全局反相")
    elif invert_mode == "local_black_bg":
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        h, w = image.shape[:2]
        ksize = max(21, (min(h, w) // 10) | 1)
        local_mean = cv2.boxFilter(gray, -1, (ksize, ksize))
        thresh = cfg.get("invert_black_bg_thresh", 80)
        mask = local_mean < thresh
        
        inverted_image = image.copy()
        for c in range(3):
            inverted_image[:, :, c] = np.where(mask, 255 - image[:, :, c], image[:, :, c])
        image = inverted_image
        applied.append("invert_local_black_bg")
        if debug:
            print(f"  [CUSTOM] 已应用局部黑底反相 (thresh={thresh})")
            
    return image, applied


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


def _diagnose_image_quality(image: np.ndarray, cfg: dict = IMAGE_CONFIG) -> tuple[str, str, str]:
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
    blur_threshold = cfg["laplacian_threshold_blur"]
    sharp_threshold = cfg["laplacian_threshold_sharp"]
    
    if laplacian_var < blur_threshold:
        clarity_level = "blurry"
    elif laplacian_var > sharp_threshold:
        clarity_level = "sharp"
    else:
        clarity_level = "normal"
    
    return brightness_level, contrast_level, clarity_level


def _detect_screen_display(image: np.ndarray, cfg: dict = IMAGE_CONFIG) -> bool:
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
    
    threshold = cfg["screen_black_threshold"]
    return black_ratio > threshold


def _process_screen_display(image: np.ndarray, cfg: dict = IMAGE_CONFIG, debug: bool = False) -> tuple[np.ndarray, list]:
    """
    屏显专用处理：局部对比度增强、自适应阈值、边缘增强。
    
    Returns:
        (processed_image, applied_enhancements_list)
    """
    enhancements = []
    
    # 1. CLAHE 对比度增强（屏显高效）
    if cfg["enhance_contrast"]:
        image = _enhance_contrast(image, cfg)
        enhancements.append("CLAHE_contrast")
        if debug:
            print(f"  [ENHANCE] CLAHE 对比度增强已应用")
    
    # 2. 自适应阈值处理（提升黑色背景中的字体对比）
    if cfg["screen_display"]["adaptive_threshold_enabled"]:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        block_size = cfg["screen_display"]["adaptive_block_size"]
        constant = cfg["screen_display"]["adaptive_constant"]
        
        # 确保 block_size 为奇数
        if block_size % 2 == 0:
            block_size += 1
        
        # 自适应阈值增强（用于可视化，但返回时保持原图以供 OCR）
        # 这里只做分析，不改变原图
        enhancements.append("adaptive_threshold")
        if debug:
            print(f"  [ENHANCE] 自适应阈值处理已应用 (block_size={block_size})")
    
    # 3. 边缘增强（Unsharp mask）
    if cfg["screen_display"]["edge_enhance_enabled"]:
        image = _enhance_edges(image, cfg)
        enhancements.append("edge_enhance")
        if debug:
            print(f"  [ENHANCE] 边缘增强已应用")
    
    return image, enhancements


def _process_general_image(
    image: np.ndarray,
    brightness_level: str,
    clarity_level: str,
    cfg: dict = IMAGE_CONFIG,
    debug: bool = False
) -> tuple[np.ndarray, list]:
    """
    非屏显图片的自适应处理。
    
    Returns:
        (processed_image, applied_enhancements_list)
    """
    enhancements = []
    
    # 1. 针对暗图片启用 CLAHE
    if brightness_level == "dark" and cfg["enhance_contrast"]:
        image = _enhance_contrast(image, cfg)
        enhancements.append("CLAHE_dark_image")
        if debug:
            print(f"  [ENHANCE] 暗图像 CLAHE 对比度增强已应用")
    
    # 2. 针对曝光过度的伽马校正
    if brightness_level == "overexposed" and cfg["gamma_correction_enabled"]:
        image = _apply_gamma_correction(image, cfg["gamma_value"])
        enhancements.append("gamma_correction")
        if debug:
            print(f"  [ENHANCE] 伽马校正已应用 (gamma={cfg['gamma_value']})")
    
    # 3. 针对模糊图片的去噪和锐化
    if clarity_level == "blurry":
        if cfg["denoise"]:
            image = _denoise(image, cfg["denoise_strength"])
            enhancements.append("denoise")
            if debug:
                print(f"  [ENHANCE] 去噪已应用")
        
        if cfg["sharpen"]:
            image = _sharpen_image(image, cfg["sharpen_strength"])
            enhancements.append("sharpen")
            if debug:
                print(f"  [ENHANCE] 锐化已应用")
    
    return image, enhancements


def _enhance_contrast(image: np.ndarray, cfg: dict = IMAGE_CONFIG) -> np.ndarray:
    """使用 CLAHE 自适应直方图均衡增强对比度。"""
    # 转到 LAB 色彩空间，只增强亮度通道
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)

    clahe = cv2.createCLAHE(
        clipLimit=cfg["clahe_clip_limit"],
        tileGridSize=cfg["clahe_grid_size"],
    )
    l_enhanced = clahe.apply(l_channel)

    lab_enhanced = cv2.merge([l_enhanced, a_channel, b_channel])
    return cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)


def _enhance_edges(image: np.ndarray, cfg: dict = IMAGE_CONFIG) -> np.ndarray:
    """
    边缘增强（Unsharp mask 方法）。
    """
    sigma = cfg["screen_display"]["edge_enhance_sigma"]
    strength = cfg["screen_display"]["edge_enhance_strength"]
    
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
    inv_gamma = 1.0 / max(0.1, gamma)
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype(np.uint8)
    return cv2.LUT(image, table)


def _sharpen_image(image: np.ndarray, strength: float = 1.0) -> np.ndarray:
    """锐化处理。"""
    kernel = np.array([[-1, -1, -1],
                       [-1,  5, -1],
                       [-1, -1, -1]]) / max(0.1, strength)
    sharpened = cv2.filter2D(image, -1, kernel)
    return np.clip(sharpened, 0, 255).astype(np.uint8)


def _resize_image(image: np.ndarray, max_size: int, interpolation: str = "INTER_LANCZOS4") -> np.ndarray:
    """等比缩放，使最长边不超过 max_size。"""
    h, w = image.shape[:2]
    if max(h, w) <= max_size:
        return image

    scale = max_size / max(h, w)
    new_w = int(w * scale)
    new_h = int(h * scale)
    
    # 选择插值方法
    if interpolation == "INTER_LANCZOS4":
        interp = cv2.INTER_LANCZOS4
    elif interpolation == "INTER_LINEAR":
        interp = cv2.INTER_LINEAR
    else:
        interp = cv2.INTER_AREA  # 默认用于缩小
    
    return cv2.resize(image, (new_w, new_h), interpolation=interp)


def _denoise(image: np.ndarray, strength: int = 15) -> np.ndarray:
    """轻度去噪，保留文字边缘。"""
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
