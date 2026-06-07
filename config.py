"""
SpotCheck AI 系统配置
"""
import os

# ============================================================
# 路径配置
# ============================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
FRONTEND_DIR = os.path.join(BASE_DIR, "frontend")
MODELS_DIR = os.path.join(BASE_DIR, "models")

# 确保必要目录存在
os.makedirs(UPLOAD_DIR, exist_ok=True)

# ============================================================
# 模型选择开关
# ============================================================
# 可选模型类型:
# - "mobile" : PP-OCRv4 Mobile 模型（CPU 速度极快，端到端约 4.3 秒，推荐 CPU 部署）
# - "server" : PP-OCRv5 Server 模型（识别精度极高，但 CPU 推理较慢，推荐 GPU 部署）
ACTIVE_MODEL = "mobile"

# ============================================================
# OCR 引擎配置
# ============================================================
if ACTIVE_MODEL == "mobile":
    OCR_CONFIG = {
        "text_detection_model_name": "PP-OCRv4_mobile_det",
        "text_detection_model_dir": os.path.join(MODELS_DIR, "PP-OCRv4_mobile_det"),
        "text_recognition_model_name": "PP-OCRv4_mobile_rec",
        "text_recognition_model_dir": os.path.join(MODELS_DIR, "PP-OCRv4_mobile_rec"),
        "text_det_limit_side_len": 640,                                                # 限制输入尺寸 640 以加速检测
    }
else:  # "server"
    OCR_CONFIG = {
        "text_detection_model_name": "PP-OCRv5_server_det",
        "text_detection_model_dir": os.path.join(MODELS_DIR, "PP-OCRv5_server_det"),
        "text_recognition_model_name": "PP-OCRv5_server_rec",
        "text_recognition_model_dir": os.path.join(MODELS_DIR, "PP-OCRv5_server_rec"),
        "text_det_limit_side_len": 960,                                                # 限制输入尺寸 960 以保证大图高精度
    }

# 基础优化参数
OCR_CONFIG.update({
    "use_doc_orientation_classify": False,                                          # 跳过方向分类（加速）
    "use_doc_unwarping": False,                                                    # 跳过文档矫正（加速）
    "use_textline_orientation": False,                                             # 跳过文本行方向（加速）
    "text_det_thresh": 0.3,                                                        # 检测阈值
    "text_det_box_thresh": 0.5,                                                    # 框置信度阈值
    "text_recognition_batch_size": 16,                                             # 识别批量大小（增大以提高吞吐）
    "enable_mkldnn": False,                                                        # 禁用 MKLDNN 以避免 PIR 兼容性报错
    "device": "cpu",                                                               # 在 CPU 上运行
})

# ============================================================
# 图像预处理配置
# ============================================================
IMAGE_CONFIG = {
    # 基础分辨率配置
    "max_size": 960,               # 最大边长
    "interpolation": "INTER_LANCZOS4",  # 插值方法（INTER_AREA/INTER_LINEAR/INTER_LANCZOS4）
    
    # 质量诊断与自适应处理
    "enable_quality_diagnosis": True,   # 启用图片质量诊断
    "enable_screen_detection": True,    # 启用屏显检测
    "screen_black_threshold": 0.4,      # 黑色区域占比阈值（> 0.4 判定为屏显）
    
    # 对比度增强配置（根据诊断结果自动调整）
    "enhance_contrast": True,      # 启用对比度增强
    "clahe_clip_limit": 2.5,       # CLAHE 对比度限制（屏显用较温和的值）
    "clahe_grid_size": (32, 32),   # CLAHE 网格大小（屏显优化）
    
    # 屏显特化处理
    "screen_display": {
        "adaptive_threshold_enabled": True,      # 自适应阈值处理
        "adaptive_block_size": 31,               # 自适应阈值块大小
        "adaptive_constant": 5,                  # 自适应阈值常数
        "edge_enhance_enabled": True,            # 边缘增强（Unsharp mask）
        "edge_enhance_sigma": 1.0,               # 高斯模糊 sigma
        "edge_enhance_strength": 1.5,            # 增强强度
        "invert_enabled": False,                 # 是否允许反色处理
    },
    
    # 亮度/对比度校正
    "gamma_correction_enabled": False,  # 伽马校正（针对曝光过度）
    "gamma_value": 0.7,             # 伽马值（< 1.0 降低亮度）
    "contrast_reduction": False,     # 对比度压缩
    
    # 去噪与锐化
    "denoise": False,               # 自适应去噪（检测到模糊时启用）
    "denoise_strength": 15,         # 去噪强度
    "sharpen": False,               # 自适应锐化
    "sharpen_strength": 1.0,        # 锐化强度
    
    # 清晰度阈值
    "laplacian_threshold_blur": 50,  # Laplacian 方差（< 50 判定为模糊）
    "laplacian_threshold_sharp": 200, # Laplacian 方差（> 200 判定为过锐）
}

# ============================================================
# 参数解析配置
# ============================================================
PARSER_CONFIG = {
    "min_confidence": 0.5,         # 最低置信度阈值
    "row_tolerance_ratio": 0.7,    # 同行判定容差（相对字高）
    "max_pair_distance": 500,      # 最大配对距离（像素）
}

# ============================================================
# 服务配置
# ============================================================
SERVER_CONFIG = {
    "host": "0.0.0.0",
    "port": 8000,
    "reload": False,
}
