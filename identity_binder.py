# -*- coding: utf-8 -*-
import collections
from shapely.geometry import Point, Polygon
import cv2

class IdentityBinder:
    """
    基于抓拍门 (Capture Zone) 与多帧投票 (Majority Voting) 的纯视觉号码绑定引擎。
    """
    def __init__(self, capture_poly_pts=None, det_model_dir=None, rec_model_dir=None):
        """
        capture_poly_pts: 抓拍门多边形顶点像素坐标列表 [[x1, y1], [x2, y2], ...]
        """
        # 默认设置一个画面下方的多边形作为抓拍门（离摄像头最近且清晰的必经通道）
        if capture_poly_pts is None:
            capture_poly_pts = [(200, 500), (1080, 500), (1280, 720), (0, 720)]
        self.capture_zone = Polygon(capture_poly_pts)
        
        # 本地 OCR 模型初始化路径
        self.det_model_dir = det_model_dir or 'e:/mycode/companyworkspace/spotcheck/models/PP-OCRv4_mobile_det'
        self.rec_model_dir = rec_model_dir or 'e:/mycode/companyworkspace/spotcheck/models/PP-OCRv4_mobile_rec'
        self.ocr = None
        
        # 投票缓存：{track_id: [num1, num2, ...]}
        self.ocr_cache = collections.defaultdict(list)

    def _init_ocr(self):
        """延迟加载 OCR 模型，防止初始化其他模块时阻塞"""
        if self.ocr is None:
            try:
                from paddleocr import PaddleOCR
                self.ocr = PaddleOCR(
                    det_model_dir=self.det_model_dir,
                    rec_model_dir=self.rec_model_dir,
                    use_angle_cls=False,
                    lang="ch",
                    use_gpu=True
                )
                print("[IdentityBinder] 成功加载本地 PP-OCRv4 识别引擎。")
            except Exception as e:
                print(f"[IdentityBinder] 加载 PaddleOCR 失败，将使用虚拟号码绑定模式。错误: {e}")
                self.ocr = "DUMMY"

    def is_in_capture_zone(self, u, v) -> bool:
        """判断脚底像素坐标是否处于抓拍门多边形内"""
        point = Point(u, v)
        return self.capture_zone.contains(point)

    def process_ocr(self, frame, person):
        """
        在抓拍门内对目标进行高频截图与 OCR 识别，并执行多帧投票绑定。
        """
        # 1. 如果已经成功绑定，则无需再做昂贵的 OCR 推理
        if person.identity is not None:
            return

        # 延迟加载 OCR
        self._init_ocr()
        
        # 2. 判断是否在抓拍门内
        bbox = person.bbox  # [xmin, ymin, xmax, ymax]
        if bbox is None:
            return
            
        foot_u = (bbox[0] + bbox[2]) / 2.0
        foot_v = bbox[3]
        if not self.is_in_capture_zone(foot_u, foot_v):
            return

        # 3. 裁剪人体的上半身区域（衣服号码通常在胸前/背后）
        h, w = frame.shape[:2]
        xmin, ymin, xmax, ymax = map(int, bbox)
        
        # 边界合法性保护
        xmin = max(0, xmin)
        ymin = max(0, ymin)
        xmax = min(w, xmax)
        ymax = min(h, ymax)
        
        person_h = ymax - ymin
        # 取人体框的 15% 到 50% 高度区域作为上半身裁剪框
        crop_ymin = int(ymin + person_h * 0.12)
        crop_ymax = int(ymin + person_h * 0.50)
        
        crop_img = frame[crop_ymin:crop_ymax, xmin:xmax]
        if crop_img.size == 0:
            return

        # 4. 执行 OCR 检测与识别
        detected_number = None
        if self.ocr == "DUMMY":
            # 虚拟环境无 GPU / 库报错时的兜底测试数字
            detected_number = f"T{person.track_id:02d}"
        elif self.ocr is not None:
            try:
                result = self.ocr.ocr(crop_img, cls=False)
                if result and result[0]:
                    for line in result[0]:
                        text = line[1][0].strip()
                        confidence = line[1][1]
                        
                        # 规则过滤：过滤非数字和置信度较低的干扰项。通常衣服号码为两位数字（如 "08"）
                        # 允许 1 到 3 位纯数字，以应对部分单数或百位数标号
                        if text.isdigit() and (1 <= len(text) <= 3) and confidence > 0.82:
                            detected_number = text
                            break
            except Exception as e:
                print(f"[IdentityBinder] OCR 运行出错: {e}")

        # 5. 缓存识别结果，并进行多数投票
        if detected_number:
            self.ocr_cache[person.track_id].append(detected_number)
            
        # 当某 track_id 累计被成功 OCR 识别达 8 次以上时进行判定
        votes = self.ocr_cache[person.track_id]
        if len(votes) >= 8:
            counter = collections.Counter(votes)
            most_common_num, count = counter.most_common(1)[0]
            ratio = count / len(votes)
            
            # 若占比超过 75%，则判定身份成功，完成绑定
            if ratio > 0.75:
                person.identity = most_common_num
                print(f"[IdentityBinder] 绑定成功: 追踪 ID {person.track_id} <=> 衣服号码 {most_common_num} (投票比例: {ratio*100:.1f}%)")
                # 清除当前 ID 缓存
                if person.track_id in self.ocr_cache:
                    del self.ocr_cache[person.track_id]
            # 如果投票太分散说明起皱/遮挡严重，修剪缓存只保留最新 4 条，等待后续帧继续累加投票
            elif len(votes) > 12:
                self.ocr_cache[person.track_id] = votes[-4:]
