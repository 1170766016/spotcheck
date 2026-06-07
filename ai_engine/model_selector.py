"""
模型自适应选择模块

功能：
1. 根据图像质量自动选择模型（移动版 vs 服务器版）
2. 自适应分辨率调整
3. 质量评分系统
"""

from typing import Dict, Any, Tuple
try:
    import cv2
except ImportError:
    cv2 = None
import numpy as np
from config import PARSER_CONFIG


class ModelSelector:
    """模型选择器。"""
    
    # 模型配置
    MODELS = {
        "mobile": {
            "name": "PP-OCRv4 Mobile",
            "det_model": "models/PP-OCRv4_mobile_det",
            "rec_model": "models/PP-OCRv4_mobile_rec",
            "description": "轻量级，快速，适合质量好的图像",
            "min_quality_score": 0.6,
            "max_resolution": (1920, 1080),
        },
        "server": {
            "name": "PP-OCRv4 Server",
            "det_model": "models/PP-OCRv4_server_det",  # 如果有的话
            "rec_model": "models/PP-OCRv4_server_rec",  # 如果有的话
            "description": "完整版，精度高，适合质量差的图像",
            "min_quality_score": 0.0,
            "max_resolution": (4096, 2160),
        },
    }
    
    @staticmethod
    def calculate_quality_score(
        image: np.ndarray,
        brightness_level: str = "normal",
        clarity_level: str = "normal",
        is_screen_display: bool = False,
        debug: bool = False
    ) -> Tuple[float, Dict[str, Any]]:
        """
        计算图像质量评分 [0, 1]。
        
        综合考虑：
        - 亮度等级（过暗/过亮降分）
        - 清晰度等级（模糊降分）
        - 屏显类型
        - 图像统计特性
        
        Returns:
            (quality_score, quality_details)
        """
        score = 1.0
        details = {
            "brightness_level": brightness_level,
            "clarity_level": clarity_level,
            "is_screen_display": is_screen_display,
            "subscores": {},
        }
        
        # 1. 亮度评分
        brightness_score = 1.0
        if brightness_level == "dark":
            brightness_score = 0.7
        elif brightness_level == "overexposed":
            brightness_score = 0.75
        details["subscores"]["brightness"] = brightness_score
        score *= brightness_score
        
        # 2. 清晰度评分
        clarity_score = 1.0
        if clarity_level == "blurry":
            clarity_score = 0.6
        details["subscores"]["clarity"] = clarity_score
        score *= clarity_score
        
        # 3. 屏显类型加分（通常屏显质量稳定）
        if is_screen_display:
            score *= 1.05  # 屏显给予小幅加分
            details["subscores"]["screen_bonus"] = 1.05
        
        # 4. 图像统计特性
        if image is not None and cv2 is not None:
            try:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                
                # 对比度评分
                std = np.std(gray)
                contrast_score = min(std / 80, 1.0)  # 标准差 80 为理想值
                details["subscores"]["contrast"] = contrast_score
                score *= 0.9 * contrast_score + 0.1  # 权重 90%
                
                # 饱和度评分（彩色度）
                hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
                saturation = np.mean(hsv[:, :, 1])
                saturation_score = min(saturation / 200, 1.0)  # 饱和度 200 为理想值
                details["subscores"]["saturation"] = saturation_score
                score *= 0.95 * saturation_score + 0.05
            except Exception as e:
                if debug:
                    print(f"[QUALITY] 图像统计计算失败: {e}")
        
        # 限制在 [0, 1] 范围内
        score = max(0, min(1, score))
        
        if debug:
            print(f"[QUALITY] 质量评分: {score:.3f}")
            for sub_metric, sub_score in details["subscores"].items():
                print(f"  ├─ {sub_metric}: {sub_score:.3f}")
        
        return score, details
    
    @staticmethod
    def select_model(quality_score: float, debug: bool = False) -> Dict[str, Any]:
        """
        根据质量评分选择最适合的模型。
        
        Returns:
            {
                "model_type": "mobile" | "server",
                "model_config": {...},
                "recommendation": "理由说明"
            }
        """
        recommendation = ""
        
        if quality_score >= 0.75:
            model_type = "mobile"
            recommendation = "图像质量优秀，使用轻量级模型以提升速度"
        elif quality_score >= 0.5:
            model_type = "mobile"
            recommendation = "图像质量良好，使用轻量级模型"
        elif quality_score >= 0.3:
            model_type = "server"
            recommendation = "图像质量一般，使用完整模型以提升精度"
        else:
            model_type = "server"
            recommendation = "图像质量较差，使用完整模型处理"
        
        result = {
            "model_type": model_type,
            "model_config": ModelSelector.MODELS[model_type],
            "quality_score": quality_score,
            "recommendation": recommendation,
        }
        
        if debug:
            print(f"[MODEL] 选择: {ModelSelector.MODELS[model_type]['name']}")
            print(f"[MODEL] 原因: {recommendation}")
        
        return result
    
    @staticmethod
    def adjust_resolution(
        image: np.ndarray,
        quality_score: float,
        target_model: str = "mobile",
        debug: bool = False
    ) -> np.ndarray:
        """
        根据质量和目标模型自适应调整分辨率。
        
        - 高质量 + 移动模型：可以降低分辨率以加速
        - 低质量 + 服务器模型：可能需要提高分辨率以改善细节
        """
        h, w = image.shape[:2]
        original_size = (w, h)
        
        max_res = ModelSelector.MODELS[target_model]["max_resolution"]
        max_w, max_h = max_res
        
        if cv2 is None:
            # OpenCV 不可用，返回原图
            if debug:
                print(f"[RESOLUTION] OpenCV 不可用，返回原图")
            return image
        
        # 如果图像已经超过最大分辨率，需要缩小
        if w > max_w or h > max_h:
            scale = min(max_w / w, max_h / h)
            new_w, new_h = int(w * scale), int(h * scale)
            image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
            
            if debug:
                print(f"[RESOLUTION] 缩小: {original_size} → {(new_w, new_h)}")
        
        # 高质量 + 移动模型：可以尝试降低分辨率
        elif target_model == "mobile" and quality_score > 0.8:
            if w > 1280 or h > 720:
                scale = 0.8
                new_w, new_h = int(w * scale), int(h * scale)
                image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
                
                if debug:
                    print(f"[RESOLUTION] 优化: {original_size} → {(new_w, new_h)}")
        
        return image
    
    @staticmethod
    def get_model_recommendation_report(
        image: np.ndarray,
        brightness_level: str,
        clarity_level: str,
        is_screen_display: bool,
        debug: bool = False
    ) -> Dict[str, Any]:
        """
        获取完整的模型选择建议报告。
        """
        # 计算质量评分
        quality_score, quality_details = ModelSelector.calculate_quality_score(
            image, brightness_level, clarity_level, is_screen_display, debug=debug
        )
        
        # 选择模型
        model_selection = ModelSelector.select_model(quality_score, debug=debug)
        
        # 调整分辨率
        adjusted_image = ModelSelector.adjust_resolution(
            image.copy(),
            quality_score,
            target_model=model_selection["model_type"],
            debug=debug
        )
        
        return {
            "quality_score": quality_score,
            "quality_details": quality_details,
            "model_selection": model_selection,
            "adjusted_image": adjusted_image,
            "original_shape": image.shape,
            "adjusted_shape": adjusted_image.shape,
        }
