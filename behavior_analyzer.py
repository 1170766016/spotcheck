# -*- coding: utf-8 -*-
import numpy as np
import cv2

class IPMMapper:
    """
    单目逆透视映射 (IPM) 变换。
    将画面中的像素坐标映射为工厂地面的 2D 物理地图坐标（米）。
    """
    def __init__(self, src_pts=None, dst_pts=None):
        """
        src_pts: 像素平面中四个点的坐标 [[u1, v1], [u2, v2], [u3, v3], [u4, v4]]
        dst_pts: 对应的物理地面 2D 坐标（米） [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
        """
        # 如果未指定，提供一个基于常见监控倾角的示例标定矩阵（2m 宽 x 8m 长的通道）
        if src_pts is None:
            src_pts = np.float32([[400, 500], [880, 500], [1150, 950], [130, 950]])
        if dst_pts is None:
            dst_pts = np.float32([[0, 0], [2, 0], [2, 8], [0, 8]])
            
        self.H = cv2.getPerspectiveTransform(
            np.array(src_pts, dtype=np.float32),
            np.array(dst_pts, dtype=np.float32)
        )

    def to_world(self, u, v):
        """
        将像素点 (u, v) 映射为物理世界坐标 (X_real, Y_real)
        """
        pts = np.array([[[u, v]]], dtype=np.float32)
        warped = cv2.perspectiveTransform(pts, self.H)
        return float(warped[0][0][0]), float(warped[0][0][1])


class BehaviorRules:
    """
    基于人体骨骼关键点和边界框的规则分析引擎
    """
    @staticmethod
    def check_fall(bbox, keypoints):
        """
        摔倒检测算法：
        1. 宽高比发生突变：人体框从竖长变横宽，即 w / h > 1.05
        2. 头部高度骤降：鼻子（关键点 0）的高度（Y坐标）显著低于髋部（关键点 11, 12）
        3. 躯干倾斜角：肩膀中点到髋部中点的连线与地平线夹角 < 30°
        """
        # bbox: [xmin, ymin, xmax, ymax]
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        if h <= 0:
            return False
            
        aspect_ratio = w / h
        
        # keypoints: [[x, y, score], ...] 共 17 个点
        # 0: 鼻子, 5: 左肩, 6: 右肩, 11: 左髋, 12: 右髋
        if len(keypoints) < 13:
            return False
            
        nose_y = keypoints[0][1]
        left_hip_y = keypoints[11][1]
        right_hip_y = keypoints[12][1]
        hip_y = (left_hip_y + right_hip_y) / 2.0
        
        # 图像坐标系中 Y 轴向下为正。nose_y > hip_y 说明头部在画面中的位置比髋部还低（人倒下了）
        cond_ratio = aspect_ratio > 1.05
        cond_height = (nose_y > hip_y) and (keypoints[0][2] > 0.3)
        
        # 躯干倾斜角度计算
        shoulder_center = (np.array(keypoints[5][:2]) + np.array(keypoints[6][:2])) / 2.0
        hip_center = (np.array(keypoints[11][:2]) + np.array(keypoints[12][:2])) / 2.0
        torso_vector = shoulder_center - hip_center  # 从髋部指向肩膀的向量
        
        # 计算与水平方向 (1, 0) 的夹角
        v_horizontal = np.array([1, 0])
        cos_theta = np.dot(torso_vector, v_horizontal) / (np.linalg.norm(torso_vector) * np.linalg.norm(v_horizontal) + 1e-6)
        angle = np.degrees(np.arccos(np.clip(abs(cos_theta), -1.0, 1.0)))  # 与水平线的夹角
        cond_angle = angle < 35.0  # 夹角小于 35 度说明人体处于横向平躺或倾斜状态
        
        # 三者中满足任意两个核心指标，即可判定为摔倒
        fall_votes = sum([cond_ratio, cond_height, cond_angle])
        return fall_votes >= 2

    @staticmethod
    def check_head_down(keypoints):
        """
        低头检测算法：
        计算鼻子到双肩中点（颈部）的连线与垂直方向的夹角。
        """
        if len(keypoints) < 7:
            return False
            
        # 0: 鼻子, 5: 左肩, 6: 右肩
        nose = np.array(keypoints[0][:2])
        left_shoulder = np.array(keypoints[5][:2])
        right_shoulder = np.array(keypoints[6][:2])
        
        # 如果置信度太低，不进行计算
        if keypoints[0][2] < 0.4 or keypoints[5][2] < 0.4 or keypoints[6][2] < 0.4:
            return False
            
        neck = (left_shoulder + right_shoulder) / 2.0
        vector_neck_to_nose = nose - neck  # 从颈部指向鼻子的向量
        vector_vertical = np.array([0, -1])  # 垂直向上向量（正常站立时鼻子在上方）
        
        cos_theta = np.dot(vector_neck_to_nose, vector_vertical) / (
            np.linalg.norm(vector_neck_to_nose) * np.linalg.norm(vector_vertical) + 1e-6
        )
        angle = np.degrees(np.arccos(np.clip(cos_theta, -1.0, 1.0)))
        
        # 如果俯角（低头角度）大于 40°，判定为低头状态
        return angle > 40.0
