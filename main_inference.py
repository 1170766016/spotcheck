# -*- coding: utf-8 -*-
import os
import sys
import time
import argparse
import numpy as np
import cv2

# 将当前目录和可能的 PaddleDetection 路径加入 sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from video_reader import RTSPStreamReader
from behavior_analyzer import IPMMapper, BehaviorRules
from occlusion_manager import OcclusionManager, TrackedPerson
from identity_binder import IdentityBinder

class FactoryMonitorSystem:
    """
    工厂人员异常行为监控与身份追踪主系统。
    支持：
    1. 真实 AI 模型推理模式 (加载 PP-YOLOE + PP-TinyPose + ByteTrack + PaddleOCR)
    2. 全图形化物理抗干扰行为仿真演示模式 (可在任何无 GPU、无模型的环境下直接运行测试)
    """
    def __init__(self, args):
        self.args = args
        self.ipm = IPMMapper()
        self.occlusion_mgr = OcclusionManager()
        self.identity_binder = IdentityBinder(
            det_model_dir=args.ocr_det_dir,
            rec_model_dir=args.ocr_rec_dir
        )
        self.tracked_persons = {}
        
        # 尝试加载 Paddle 推理组件
        self.use_real_model = False
        if args.video != "simulation":
            self.use_real_model = self._init_paddle_detectors()
            if not self.use_real_model:
                print("\n[!] 警告: 本地未找到部署模型权重或环境未正确安装，系统将自动进入【全图形化行为仿真与抗干扰演示模式】！")
        else:
            print("\n[INFO] 系统已选择启动【全图形化行为仿真与抗干扰演示模式】。")

    def _init_paddle_detectors(self):
        """初始化 PaddleDetection 与 PaddleOCR 的推理预测器"""
        paddledet_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "PaddleDetection")
        if not os.path.exists(paddledet_path):
            print(f"[!] 找不到 PaddleDetection 目录: {paddledet_path}")
            return False
            
        sys.path.append(paddledet_path)
        sys.path.append(os.path.join(paddledet_path, "deploy", "python"))
        
        # 验证模型文件夹是否存在
        if not os.path.exists(self.args.det_model) or not os.path.exists(self.args.keypoint_model):
            print(f"[!] 检测模型或姿态模型目录不存在: \n- 检测: {self.args.det_model}\n- 姿态: {self.args.keypoint_model}")
            return False
            
        try:
            # 动态导入官方 deploy 推理类
            from predict import Detector
            from keypoint_infer import KeyPointDetector
            
            # 初始化人体检测器 (内置了 ByteTrack 关联)
            self.detector = Detector(
                model_dir=self.args.det_model,
                device=self.args.device.upper(),
                run_mode='fluid',
                threshold=0.5
            )
            # 初始化关键点估计器 (PP-TinyPose)
            self.pose_detector = KeyPointDetector(
                model_dir=self.args.keypoint_model,
                device=self.args.device.upper(),
                run_mode='fluid'
            )
            print("[INFO] 成功加载 PP-YOLOE 行人追踪模型和 PP-TinyPose 姿态评估模型。")
            return True
        except Exception as e:
            print(f"[!] 导入或加载 Paddle 预测器失败: {e}")
            return False

    def run(self):
        if self.use_real_model:
            self._run_real_inference()
        else:
            self._run_simulation()

    def _run_real_inference(self):
        """真实视频流/摄像头模型推理主循环"""
        print(f"[INFO] 启动真实模型推理，视频源: {self.args.video}")
        reader = RTSPStreamReader(self.args.video).start()
        
        cv2.namedWindow("Factory Monitor System (AI Inference)", cv2.WINDOW_NORMAL)
        
        while reader.isOpened():
            frame = reader.read()
            if frame is None:
                time.sleep(0.01)
                continue
                
            current_time = time.time()
            
            # 1. 运行人体检测与跟踪 (ByteTrack)
            # 官方 Detector.predict 会输出 bbox 框坐标及对应的 track_id
            det_results = self.detector.predict(frame, threshold=0.5)
            
            # 2. 如果有人，提取关键点并进行业务研判
            if det_results and 'boxes' in det_results and len(det_results['boxes']) > 0:
                boxes = det_results['boxes']
                
                # 过滤出类别为行人的目标（通常 class_id == 0）
                pedestrian_boxes = [box for box in boxes if int(box[0]) == 0]
                
                if len(pedestrian_boxes) > 0:
                    # 运行 PP-TinyPose 关键点估计
                    # 官方 KeyPointDetector.predict 输入为原图和所有人体的框
                    kpts_results = self.pose_detector.predict(frame, pedestrian_boxes)
                    
                    # 遍历追踪到的人员进行判定
                    for i, box in enumerate(pedestrian_boxes):
                        # box 格式: [class_id, score, xmin, ymin, xmax, ymax, track_id]
                        track_id = int(box[6]) if len(box) > 6 else i
                        bbox = box[2:6]
                        kpts = kpts_results['keypoints'][i] if i < len(kpts_results['keypoints']) else None
                        
                        if kpts is None:
                            continue
                            
                        # 更新或创建人员业务状态
                        if track_id not in self.tracked_persons:
                            self.tracked_persons[track_id] = TrackedPerson(track_id)
                        
                        person = self.tracked_persons[track_id]
                        person.bbox = bbox
                        person.last_seen_time = current_time
                        person.is_occluded = False
                        
                        # IPM 物理坐标映射
                        foot_u = (bbox[0] + bbox[2]) / 2.0
                        foot_v = bbox[3]
                        old_pos = person.world_pos
                        person.world_pos = self.ipm.to_world(foot_u, foot_v)
                        
                        # 计算速度
                        if old_pos is not None:
                            dist = np.linalg.norm(np.array(person.world_pos) - np.array(old_pos))
                            person.velocity = dist * 25.0  # 假设为 25 fps
                            
                        # 行为分析
                        # 2.1 摔倒判定
                        is_fall = BehaviorRules.check_fall(bbox, kpts)
                        if is_fall:
                            cv2.putText(frame, f"FALL ALERT - ID {track_id}", (int(bbox[0]), int(bbox[1]-10)),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                            print(f"[告警] 检测到 ID {track_id} 摔倒！")
                            
                        # 2.2 低头玩手机判定
                        is_head_down = BehaviorRules.check_head_down(kpts)
                        if is_head_down and person.velocity < 0.2:
                            if not person.headdown_timer.is_running:
                                person.headdown_timer.start_time = current_time
                                person.headdown_timer.is_running = True
                            else:
                                elapsed = current_time - person.headdown_timer.start_time
                                if elapsed > 15.0:  # 持续 15 秒判定为玩手机
                                    cv2.putText(frame, f"PLAY PHONE - ID {track_id}", (int(bbox[0]), int(bbox[3]+20)),
                                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                        else:
                            person.headdown_timer.is_running = False
                            
                        # 2.3 抓拍区 OCR 绑定
                        self.identity_binder.process_ocr(frame, person)
                        
                        # 可视化绘制
                        self._draw_skeleton(frame, bbox, kpts, person)
            
            # 3. 驱动空间静态锚点及抗遮挡计时
            self.occlusion_mgr.update_anchors(self.tracked_persons, current_time)
            
            # 4. 静止不动状态判定与告警绘制
            for person in self.tracked_persons.values():
                if person.still_timer.is_running and not person.is_occluded:
                    still_time = current_time - person.still_timer.start_time + person.still_timer.accumulated_time
                    if still_time > 15.0:  # 演示用：15秒即触发告警。生产环境可改为180秒
                        bbox = person.bbox
                        if bbox is not None:
                            cv2.putText(frame, f"STILL WARNING - {person.identity or person.track_id} ({still_time:.1f}s)", 
                                        (int(bbox[0]), int(bbox[1]-30)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # 绘制抓拍门 ROI
            pts = np.array(self.identity_binder.capture_zone.exterior.coords, dtype=np.int32)
            cv2.polylines(frame, [pts], True, (0, 255, 0), 2)
            cv2.putText(frame, "Capture Zone (OCR)", (int(pts[0][0]), int(pts[0][1]-10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # 显示画面
            cv2.imshow("Factory Monitor System (AI Inference)", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
        reader.stop()
        cv2.destroyAllWindows()

    def _run_simulation(self):
        """物理抗干扰行为仿真演示模式"""
        print("[INFO] 启动全场景交互仿真演示程序。")
        print("控制面板 (在画面弹窗下按键):")
        print("- [F] 模拟/恢复 工人 A 突然摔倒 (Fall)")
        print("- [H] 模拟/恢复 工人 A 低头玩手机 (Head Down)")
        print("- [C] 模拟/恢复 区域内多人聚集 (Crowd Gathering)")
        print("- [O] 触发路人相交遮挡 (验证空间锚点与ID Switch计时继承)")
        print("- [R] 重置所有仿真参数")
        print("- [Q] 退出程序")
        
        cv2.namedWindow("Factory Monitor Simulator", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Factory Monitor Simulator", 1280, 720)
        
        frame_w, frame_h = 1280, 720
        fps = 25
        frame_interval = 1.0 / fps
        
        # 仿真参数
        time_sim = 0.0
        
        # 模拟的人体轨迹：定义工人 A (ID=1)
        a_x, a_y = 50, 400
        a_w, a_h = 100, 220
        a_id = 1
        
        # 模拟路人 B (ID=2)
        b_x, b_y = 1200, 400
        b_w, b_h = 95, 215
        b_id = 2
        
        is_fall_simulated = False
        is_headdown_simulated = False
        is_crowd_simulated = False
        is_occlusion_simulated = False
        occlusion_start_time = 0.0
        occlusion_done = False
        
        # 物理坐标转换模拟：为仿真数据注册
        self.tracked_persons[a_id] = TrackedPerson(a_id)
        
        while True:
            # 创建黑色背景画布，代表工厂通道
            canvas = np.zeros((frame_h, frame_w, 3), dtype=np.uint8)
            # 画网格线模拟工厂通道
            for cy in range(100, frame_h, 100):
                cv2.line(canvas, (0, cy), (frame_w, cy), (40, 40, 40), 1)
            cv2.line(canvas, (0, 350), (frame_w, 350), (100, 100, 100), 2)
            cv2.line(canvas, (0, 600), (frame_w, 600), (100, 100, 100), 2)
            cv2.putText(canvas, "Corridor Wall A", (20, 340), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1)
            cv2.putText(canvas, "Corridor Wall B (Ground Floor)", (20, 620), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1)
            
            current_time = time.time()
            time_sim += frame_interval
            
            # 绘制抓拍门 ROI
            pts = np.array(self.identity_binder.capture_zone.exterior.coords, dtype=np.int32)
            cv2.polylines(canvas, [pts], True, (0, 255, 0), 2)
            cv2.putText(canvas, "Capture Zone (OCR Entrance)", (int(pts[0][0])+10, int(pts[0][1])-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            active_sim_ids = []
            
            # --- 模拟工人 A 行走及静止状态机 ---
            if not is_fall_simulated:
                # 工人 A 正常走入
                if a_x < 600:
                    a_x += 4  # 走路中
                    a_velocity = 1.2
                else:
                    a_x = 600  # 到达中央站立不动
                    a_velocity = 0.0
            else:
                a_velocity = 0.0  # 摔倒后静止
                
            # --- 模拟路人 B 相交和遮挡 ---
            if is_occlusion_simulated and not occlusion_done:
                if b_x > 620:
                    b_x -= 6  # B 走向 A
                elif b_x <= 620 and b_x > 500:
                    # B 和 A 在物理位置上发生重叠遮挡
                    b_x -= 2
                else:
                    # B 离开相交区继续往左走
                    b_x -= 6
                    occlusion_done = True
                    # B 离开后，A 重新出现，但由于 ID Switch，A 获得了新的 ID = 3
                    a_id = 3
                    self.tracked_persons[a_id] = TrackedPerson(a_id)
                    print("[Simulator] 遮挡结束！工人重新被检出，但发生 ID Switch。原 ID: 1 => 新 ID: 3")
            
            # --- 构建本帧的检测与姿态结果 ---
            # 1. 模拟工人 A
            a_visible = True
            # 如果处于重合遮挡的区间，A 不可见
            if is_occlusion_simulated and not occlusion_done and (520 < b_x <= 620):
                a_visible = False
                
            if a_visible:
                active_sim_ids.append(a_id)
                person_a = self.tracked_persons[a_id]
                person_a.last_seen_time = current_time
                person_a.is_occluded = False
                person_a.velocity = a_velocity
                
                # 设定 bbox 像素坐标
                if is_fall_simulated:
                    person_a.bbox = [a_x - 110, a_y + 80, a_x + 110, a_y + 220]
                else:
                    person_a.bbox = [a_x - 50, a_y, a_x + 50, a_y + a_h]
                    
                # 模拟骨骼关键点
                kpts_a = [[0, 0, 0.9]] * 17
                if is_fall_simulated:
                    # 摔倒骨骼：鼻子(Y轴向下为正)低于髋部
                    kpts_a[0] = [a_x + 90, a_y + 180, 0.9]
                    kpts_a[5] = [a_x + 20, a_y + 160, 0.9]
                    kpts_a[6] = [a_x + 20, a_y + 170, 0.9]
                    kpts_a[11] = [a_x - 40, a_y + 140, 0.9]
                    kpts_a[12] = [a_x - 40, a_y + 150, 0.9]
                elif is_headdown_simulated:
                    # 低头骨骼：鼻子 Y 坐标显著偏下，向前下方倾斜 (低于双肩)
                    kpts_a[0] = [a_x + 25, a_y + 80, 0.9]  # 鼻子
                    kpts_a[5] = [a_x - 20, a_y + 50, 0.9]  # 左肩
                    kpts_a[6] = [a_x + 20, a_y + 50, 0.9]  # 右肩
                    kpts_a[11] = [a_x - 15, a_y + 110, 0.9]
                    kpts_a[12] = [a_x + 15, a_y + 110, 0.9]
                else:
                    # 正常平视
                    kpts_a[0] = [a_x, a_y + 20, 0.9]
                    kpts_a[5] = [a_x - 20, a_y + 50, 0.9]
                    kpts_a[6] = [a_x + 20, a_y + 50, 0.9]
                    kpts_a[11] = [a_x - 15, a_y + 110, 0.9]
                    kpts_a[12] = [a_x + 15, a_y + 110, 0.9]
                
                person_a.world_pos = self.ipm.to_world((person_a.bbox[0]+person_a.bbox[2])/2.0, person_a.bbox[3])
                
                # 抓拍区 OCR 号码绑定模拟
                self.identity_binder.process_ocr(canvas, person_a)
                
                # 渲染 A 的画面
                self._draw_skeleton(canvas, person_a.bbox, kpts_a, person_a, is_fall_simulated)
                
                # 模拟低头判定运行
                if is_headdown_simulated:
                    is_hd = BehaviorRules.check_head_down(kpts_a)
                    if is_hd and person_a.velocity < 0.2:
                        if not person_a.headdown_timer.is_running:
                            person_a.headdown_timer.start_time = current_time
                            person_a.headdown_timer.is_running = True
                        else:
                            hd_time = current_time - person_a.headdown_timer.start_time
                            if hd_time > 3.0:  # 仿真下持续低头 3 秒即告警
                                cv2.rectangle(canvas, (0, 0), (frame_w, 70), (0, 140, 255), -1)
                                cv2.putText(canvas, f"PLAY PHONE WARNING: Worker #{person_a.identity or a_id} is playing phone!", 
                                            (180, 48), cv2.FONT_HERSHEY_DUPLEX, 0.9, (255, 255, 255), 2)
                else:
                    person_a.headdown_timer.is_running = False
            else:
                if 1 in self.tracked_persons:
                    self.tracked_persons[1].is_occluded = True
                    self.tracked_persons[1].velocity = 0.0
            
            # 2. 模拟路人 B
            if is_occlusion_simulated and b_x > 50:
                active_sim_ids.append(b_id)
                if b_id not in self.tracked_persons:
                    self.tracked_persons[b_id] = TrackedPerson(b_id)
                person_b = self.tracked_persons[b_id]
                person_b.last_seen_time = current_time
                person_b.is_occluded = False
                person_b.velocity = 1.5
                person_b.bbox = [b_x - 47, b_y - 10, b_x + 47, b_y + b_h - 10]
                person_b.world_pos = self.ipm.to_world(b_x, b_y + b_h - 10)
                
                kpts_b = [[0, 0, 0.9]] * 17
                kpts_b[0] = [b_x, b_y + 10, 0.9]
                kpts_b[5] = [b_x - 20, b_y + 40, 0.9]
                kpts_b[6] = [b_x + 20, b_y + 40, 0.9]
                kpts_b[11] = [b_x - 15, b_y + 100, 0.9]
                kpts_b[12] = [b_x + 15, b_y + 100, 0.9]
                
                self._draw_skeleton(canvas, person_b.bbox, kpts_b, person_b)
                
            # 3. 模拟多人聚集人员 (ID=10, 11, 12)
            if is_crowd_simulated:
                crowd_data = [
                    {"id": 10, "x": 420, "y": 410},
                    {"id": 11, "x": 480, "y": 410},
                    {"id": 12, "x": 540, "y": 410}
                ]
                for data in crowd_data:
                    pid = data["id"]
                    px = data["x"]
                    py = data["y"]
                    active_sim_ids.append(pid)
                    if pid not in self.tracked_persons:
                        self.tracked_persons[pid] = TrackedPerson(pid)
                    p = self.tracked_persons[pid]
                    p.last_seen_time = current_time
                    p.is_occluded = False
                    p.velocity = 0.0
                    p.bbox = [px - 45, py, px + 45, py + a_h]
                    p.world_pos = self.ipm.to_world(px, py + a_h)
                    
                    kpts_p = [[0, 0, 0.9]] * 17
                    kpts_p[0] = [px, py + 20, 0.9]
                    kpts_p[5] = [px - 20, py + 50, 0.9]
                    kpts_p[6] = [px + 20, py + 50, 0.9]
                    kpts_p[11] = [px - 15, py + 110, 0.9]
                    kpts_p[12] = [px + 15, py + 110, 0.9]
                    
                    self._draw_skeleton(canvas, p.bbox, kpts_p, p)
            
            # 4. 模拟摔倒判定运行 (摔倒优先置顶红色报警栏)
            if a_visible and is_fall_simulated:
                is_fall = BehaviorRules.check_fall(person_a.bbox, kpts_a)
                if is_fall:
                    cv2.rectangle(canvas, (0, 0), (frame_w, 70), (0, 0, 255), -1)
                    cv2.putText(canvas, "FALL DETECTED ALERT!!!", (420, 48),
                                cv2.FONT_HERSHEY_DUPLEX, 1.1, (255, 255, 255), 2)
            
            # 5. 驱动空间静态锚点及抗遮挡计时
            active_persons = {}
            for pid, p in self.tracked_persons.items():
                if pid in active_sim_ids or (pid == 1 and p.is_occluded):
                    active_persons[pid] = p
            
            self.occlusion_mgr.update_anchors(active_persons, current_time)
            
            # 6. 运行多人聚集检测 (DBSCAN 空间密度聚类)
            crowd_info = self._run_dbscan_crowd(active_persons)
            if crowd_info:
                cv2.rectangle(canvas, (0, 75), (frame_w, 145), (0, 0, 255), -1)
                cv2.putText(canvas, f"CROWD ALERT: {crowd_info['count']} Workers Gathering at Center ({crowd_info['center'][0]:.1f}m, {crowd_info['center'][1]:.1f}m)!", 
                            (180, 120), cv2.FONT_HERSHEY_DUPLEX, 0.9, (255, 255, 255), 2)
            
            # 7. 静支时间判定与告警绘制 (演示静止告警时间：设定为 12.0 秒)
            for pid, person in active_persons.items():
                if person.still_timer.is_running and not person.is_occluded:
                    still_time = current_time - person.still_timer.start_time + person.still_timer.accumulated_time
                    if still_time > 12.0:
                        bbox = person.bbox
                        cv2.rectangle(canvas, (0, frame_h - 70), (frame_w, frame_h), (0, 0, 255), -1)
                        cv2.putText(canvas, f"STILL ALERT: Worker #{person.identity or pid} static for {still_time:.1f}s!", 
                                    (260, frame_h - 25), cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 255), 2)
            
            # 8. 画静态空间锚点以辅助直观展示
            for aid, anchor in self.occlusion_mgr.anchors.items():
                if anchor.is_active:
                    anchor_pixel_x = 600
                    anchor_pixel_y = 620
                    cv2.circle(canvas, (anchor_pixel_x, anchor_pixel_y), 45, (0, 255, 255), 2)
                    cv2.putText(canvas, f"Anchor #{aid}", (anchor_pixel_x - 45, anchor_pixel_y + 65),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                    cv2.putText(canvas, f"Timer Saved: {anchor.accumulated_time:.1f}s", (anchor_pixel_x - 65, anchor_pixel_y + 85),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            
            # 在右上角画控制面板说明
            cv2.rectangle(canvas, (860, 160), (1260, 420), (30, 30, 30), -1)
            cv2.putText(canvas, "Control Panel (Keyboard):", (880, 190), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            cv2.putText(canvas, "- [F] Simulate Worker A Fall", (880, 225), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            cv2.putText(canvas, "- [H] Simulate Worker A Head Down", (880, 260), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            cv2.putText(canvas, "- [C] Simulate 3-Person Gathering", (880, 295), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            cv2.putText(canvas, "- [O] Simulate Occlusion (ID Switch)", (880, 330), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            cv2.putText(canvas, "- [R] Reset Simulation", (880, 365), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            cv2.putText(canvas, "- [Q] Quit Program", (880, 400), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            
            cv2.imshow("Factory Monitor Simulator", canvas)
            
            key = cv2.waitKey(int(frame_interval * 1000)) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('f'):
                is_fall_simulated = not is_fall_simulated
                print(f"[Simulator] 工人摔倒状态切换为: {is_fall_simulated}")
            elif key == ord('h'):
                is_headdown_simulated = not is_headdown_simulated
                print(f"[Simulator] 工人低头状态切换为: {is_headdown_simulated}")
            elif key == ord('c'):
                is_crowd_simulated = not is_crowd_simulated
                print(f"[Simulator] 多人聚集状态切换为: {is_crowd_simulated}")
            elif key == ord('o'):
                is_occlusion_simulated = True
                occlusion_start_time = current_time
                print("[Simulator] 触发路人经过遮挡仿真模拟！")
            elif key == ord('r'):
                is_fall_simulated = False
                is_headdown_simulated = False
                is_crowd_simulated = False
                is_occlusion_simulated = False
                occlusion_done = False
                a_id = 1
                a_x = 50
                b_x = 1200
                self.tracked_persons.clear()
                self.tracked_persons[a_id] = TrackedPerson(a_id)
                self.occlusion_mgr.anchors.clear()
                self.occlusion_mgr.anchor_counter = 0
                self.identity_binder.ocr_cache.clear()
                print("[Simulator] 仿真模拟已重置。")
                
        cv2.destroyAllWindows()

    def _run_dbscan_crowd(self, active_persons):
        """运行 DBSCAN 聚类算法检测多人聚集"""
        from sklearn.cluster import DBSCAN
        coords = []
        pids = []
        for pid, p in active_persons.items():
            if p.world_pos is not None and not p.is_occluded:
                coords.append(p.world_pos)
                pids.append(pid)
                
        if len(coords) < 3:
            return None
            
        coords = np.array(coords)
        db = DBSCAN(eps=1.5, min_samples=3).fit(coords)
        labels = db.labels_
        
        for label in set(labels):
            if label != -1:
                cluster_members = [pids[i] for i, l in enumerate(labels) if l == label]
                cluster_coords = coords[labels == label]
                center = np.mean(cluster_coords, axis=0)
                return {
                    "center": center,
                    "members": cluster_members,
                    "count": len(cluster_members)
                }
        return None

    def _draw_skeleton(self, frame, bbox, keypoints, person, is_fall=False):
        """在画面中绘制骨架、边界框和绑定标签"""
        xmin, ymin, xmax, ymax = map(int, bbox)
        
        # 绘制边界框颜色
        color = (0, 255, 0)  # 默认绿色
        if is_fall:
            color = (0, 0, 255)  # 红色
        elif person.still_timer.is_running:
            color = (0, 255, 255)  # 黄色
            
        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 2)
        
        # 绘制核心骨骼点
        core_kpts = [0, 5, 6, 11, 12]
        for idx in core_kpts:
            if idx < len(keypoints):
                x, y, score = keypoints[idx]
                if score > 0.4:
                    cv2.circle(frame, (int(x), int(y)), 5, (0, 165, 255), -1)
                    
        # 画双肩连线
        if keypoints[5][2] > 0.4 and keypoints[6][2] > 0.4:
            cv2.line(frame, (int(keypoints[5][0]), int(keypoints[5][1])),
                     (int(keypoints[6][0]), int(keypoints[6][1])), (0, 255, 0), 1)

        # 绘制文字标签栏
        label = f"ID: {person.track_id}"
        if person.identity:
            label += f" [Worker #{person.identity}]"
        if person.still_timer.is_running:
            label += f" | Static"
            
        cv2.rectangle(frame, (xmin, ymin - 22), (xmin + 240, ymin), color, -1)
        cv2.putText(frame, label, (xmin + 5, ymin - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Factory People Behavior Monitor System")
    parser.add_argument("--video", type=str, default="simulation", 
                        help="视频文件路径、RTSP流地址或 'simulation' (仿真模式)")
    parser.add_argument("--det_model", type=str, default="models/mot_ppyoloe_s", 
                        help="人体检测推理模型目录")
    parser.add_argument("--keypoint_model", type=str, default="models/tinypose_256x192", 
                        help="姿态估计推理模型目录")
    parser.add_argument("--ocr_det_dir", type=str, default="models/PP-OCRv4_mobile_det", 
                        help="本地 OCR 文字检测模型目录")
    parser.add_argument("--ocr_rec_dir", type=str, default="models/PP-OCRv4_mobile_rec", 
                        help="本地 OCR 文字识别模型目录")
    parser.add_argument("--device", type=str, default="GPU", help="推理运行设备: GPU 或 CPU")
    
    args = parser.parse_args()
    
    system = FactoryMonitorSystem(args)
    system.run()
