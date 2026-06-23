# -*- coding: utf-8 -*-
import threading
import time
import cv2

class RTSPStreamReader:
    """
    多线程防丢帧视频/RTSP流读取器。
    通过独立线程循环读取视频帧，推理线程调用 read() 时始终获取最新图像，
    从而解决由于推理延时导致的画面帧积压与告警延迟问题。
    """
    def __init__(self, rtsp_url):
        self.rtsp_url = rtsp_url
        self.cap = cv2.VideoCapture(rtsp_url)
        self.latest_frame = None
        self.stopped = False
        self.lock = threading.Lock()
        self.is_opened = self.cap.isOpened()
        
    def start(self):
        t = threading.Thread(target=self._update, args=())
        t.daemon = True
        t.start()
        return self

    def _update(self):
        while not self.stopped:
            ret, frame = self.cap.read()
            if not ret:
                # 针对 RTSP 流断开，执行自动重连逻辑
                print(f"[RTSP] 视频流连接断开，正在尝试重连: {self.rtsp_url}")
                self.cap.release()
                time.sleep(2)
                self.cap = cv2.VideoCapture(self.rtsp_url)
                continue
            with self.lock:
                self.latest_frame = frame
                self.is_opened = True

    def read(self):
        with self.lock:
            return self.latest_frame

    def isOpened(self):
        with self.lock:
            return self.is_opened

    def stop(self):
        self.stopped = True
        with self.lock:
            self.is_opened = False
        self.cap.release()
        print("[RTSP] 视频拉流线程已安全退出")
