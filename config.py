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
# OCR 引擎配置
# ============================================================
OCR_CONFIG = {
    "lang": "ch",                                                          # 中英文混合识别
    "text_detection_model_dir": os.path.join(MODELS_DIR, "PP-OCRv5_server_det"),   # 本地检测模型
    "text_recognition_model_dir": os.path.join(MODELS_DIR, "PP-OCRv5_server_rec"), # 本地识别模型
    "use_doc_orientation_classify": False,                                  # 跳过方向分类（加速）
    "use_doc_unwarping": False,                                             # 跳过文档矫正（加速）
    "use_textline_orientation": False,                                      # 跳过文本行方向（加速）
    "text_det_thresh": 0.3,                                                 # 检测阈值
    "text_det_box_thresh": 0.5,                                             # 框置信度阈值
    "text_det_limit_side_len": 960,                                         # 检测输入尺寸限制（加速）
    "text_recognition_batch_size": 8,                                       # 识别批量大小
    "enable_mkldnn": False,                                                 # 禁用 MKLDNN 以避免 PIR 兼容性报错
    "device": "cpu",                                                        # 在 CPU 上运行
}

# ============================================================
# 图像预处理配置
# ============================================================
IMAGE_CONFIG = {
    "max_size": 1920,              # 最大边长（像素）
    "enhance_contrast": True,      # 是否增强对比度
    "clahe_clip_limit": 2.0,       # CLAHE 对比度限制
    "clahe_grid_size": (8, 8),     # CLAHE 网格大小
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
