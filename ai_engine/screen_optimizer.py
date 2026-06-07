"""
屏显参数特化优化模块

功能：
1. ROI（感兴趣区域）自动提取
2. 参数聚类分组
3. 屏显特定的 OCR 后处理
4. 参数去重和合并
"""

import numpy as np
try:
    import cv2
except ImportError:
    cv2 = None
from typing import List, Dict, Tuple, Any
from config import IMAGE_CONFIG


class ScreenDisplayOptimizer:
    """屏显参数优化器。"""
    
    @staticmethod
    def extract_roi(image: np.ndarray, debug: bool = False) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        自动提取 ROI（只含有文字的区域）。
        
        对于屏显图片（黑色背景），通过检测非黑色像素来定位内容区域。
        
        Returns:
            (roi_image, roi_info)
            - roi_image: 提取的 ROI 图像
            - roi_info: ROI 信息 {"x", "y", "width", "height", "area_ratio"}
        """
        if cv2 is None:
            # OpenCV 不可用，返回整个图像
            h, w = image.shape[:2]
            return image, {
                "x": 0, "y": 0, "width": w, "height": h,
                "area_ratio": 1.0, "has_content": False,
                "message": "OpenCV 不可用，返回整个图像"
            }
        
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        
        # 二值化：找出非黑色区域
        # 黑色像素 (< 50)，非黑色像素 (>= 50)
        _, binary = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY)
        
        # 形态学处理：填充小洞、移除小噪声
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)
        
        # 寻找轮廓并获取最大连通区域
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            # 没有找到内容，返回整个图像
            return image, {
                "x": 0, "y": 0, "width": w, "height": h,
                "area_ratio": 1.0, "has_content": False
            }
        
        # 获取最大的矩形区域
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, roi_w, roi_h = cv2.boundingRect(largest_contour)
        
        # 添加边距（避免裁剪太紧）
        margin = 10
        x = max(0, x - margin)
        y = max(0, y - margin)
        roi_w = min(w - x, roi_w + 2 * margin)
        roi_h = min(h - y, roi_h + 2 * margin)
        
        roi_image = image[y:y+roi_h, x:x+roi_w]
        area_ratio = (roi_w * roi_h) / (w * h)
        
        roi_info = {
            "x": x,
            "y": y,
            "width": roi_w,
            "height": roi_h,
            "area_ratio": area_ratio,
            "has_content": area_ratio < 0.95,  # 如果 ROI 不是整个图像，则有内容
        }
        
        if debug:
            print(f"[ROI] 提取区域: ({x}, {y}) {roi_w}x{roi_h}, 占比: {area_ratio:.1%}")
        
        return roi_image, roi_info
    
    @staticmethod
    def cluster_parameters(
        parameters: List[Dict[str, Any]],
        debug: bool = False
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        按参数类型对参数进行聚类分组。
        
        Returns:
            {
                "temperature": [...],
                "pressure": [...],
                "speed": [...],
                "others": [...]
            }
        """
        clusters = {
            "temperature": [],
            "pressure": [],
            "speed": [],
            "electrical": [],
            "time": [],
            "others": [],
        }
        
        # 参数类型关键词
        type_keywords = {
            "temperature": ["temp", "温度", "℃", "°c"],
            "pressure": ["pressure", "压力", "mpa", "pa", "bar"],
            "speed": ["speed", "rpm", "转速", "r/min"],
            "electrical": ["voltage", "current", "power", "电压", "电流", "功率", "v", "a", "w"],
            "time": ["time", "时间", "s", "sec", "min", "h"],
        }
        
        for param in parameters:
            name = param.get("name", "").lower()
            unit = param.get("unit", "").lower()
            value = str(param.get("value", "")).lower()
            
            # 根据参数名和单位分类
            classified = False
            for cluster_type, keywords in type_keywords.items():
                if any(kw in name or kw in unit or kw in value for kw in keywords):
                    clusters[cluster_type].append(param)
                    classified = True
                    break
            
            if not classified:
                clusters["others"].append(param)
        
        if debug:
            for cluster_type, params in clusters.items():
                if params:
                    print(f"[CLUSTER] {cluster_type}: {len(params)} 个参数")
        
        return clusters
    
    @staticmethod
    def deduplicate_parameters(
        parameters: List[Dict[str, Any]],
        debug: bool = False
    ) -> List[Dict[str, Any]]:
        """
        去重和合并相同的参数。
        
        如果识别出多个几乎相同的参数（参数名相同，值接近），
        保留置信度最高的。
        """
        if not parameters:
            return []
        
        # 按参数名分组
        name_groups = {}
        for param in parameters:
            name = param.get("name", "").strip().lower()
            if name not in name_groups:
                name_groups[name] = []
            name_groups[name].append(param)
        
        deduplicated = []
        removed_count = 0
        
        for name, group in name_groups.items():
            if len(group) == 1:
                deduplicated.append(group[0])
            else:
                # 多个相同参数，选择置信度最高的
                best = max(group, key=lambda p: p.get("confidence", 0))
                deduplicated.append(best)
                removed_count += len(group) - 1
                
                if debug:
                    print(f"[DEDUP] 参数 '{name}': 移除 {len(group) - 1} 个重复项，保留置信度最高的")
        
        if debug and removed_count > 0:
            print(f"[DEDUP] 总共去重: {removed_count} 个参数")
        
        return deduplicated
    
    @staticmethod
    def enhance_ocr_results(
        ocr_results: list,
        image: np.ndarray,
        debug: bool = False
    ) -> list:
        """
        屏显特定的 OCR 后处理。
        
        对于低对比度或特殊字体的文字框，进行增强处理。
        """
        if not ocr_results or not ocr_results[0]:
            return ocr_results
        
        enhanced = []
        
        for line in ocr_results[0]:
            text, confidence = line[1] if len(line) > 1 else ("", 0.0)
            
            # 后处理：
            # 1. 数字识别增强（某些字体 0/O, 1/l/I 容易混淆）
            text = ScreenDisplayOptimizer._enhance_digit_recognition(text)
            
            # 2. 常见 OCR 错误修正
            text = ScreenDisplayOptimizer._correct_common_errors(text)
            
            # 更新置信度（如果有修正，降低一些）
            if text != line[1][0]:
                confidence *= 0.95  # 修正后置信度略降
            
            # 重新组织结果
            enhanced.append([line[0], (text, confidence)])
        
        if enhanced:
            ocr_results[0] = enhanced
        
        return ocr_results
    
    @staticmethod
    def _enhance_digit_recognition(text: str) -> str:
        """数字识别增强。"""
        # 在可能是参数名-值分隔的位置，用冒号替换其他符号
        text = text.replace("：", ":")  # 中文冒号 → 英文冒号
        text = text.replace("：", ":")
        
        # 常见的 0/O 混淆（在数字上下文中）
        # 但这需要更多上下文，暂时保守处理
        
        return text
    
    @staticmethod
    def _correct_common_errors(text: str) -> str:
        """修正常见 OCR 错误。"""
        corrections = {
            "℃": ["c", "°c"],  # 温度单位
            "℃": ["C"],
            "MPa": ["Mpa", "mpa", "M Pa"],
            "rpm": ["r/min", "R/min"],
        }
        
        for correct, errors in corrections.items():
            for error in errors:
                if error in text:
                    text = text.replace(error, correct)
        
        return text
