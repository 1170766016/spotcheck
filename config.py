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
    "max_size": 960,               # 最大边长（从 1920 降到 960，减小推理图像）
    "enhance_contrast": False,     # 设备屏显本身高对比度，跳过 CLAHE 增强（节省 ~85ms）
    "clahe_clip_limit": 2.0,       # CLAHE 对比度限制（备用）
    "clahe_grid_size": (8, 8),     # CLAHE 网格大小（备用）
    "denoise": False,              # 是否去噪（设置为 False 以防止 fastNlMeansDenoisingColored 耗时过长导致接口挂起）
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
