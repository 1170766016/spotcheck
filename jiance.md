

---

### **一、 环境准备：如何从 GitHub “强攻”安装**

因为内网可能屏蔽了 `pip` 官方源，但你能上 GitHub，这是关键。

1.  **首选尝试（国内镜像）：**
    在命令行输入（这能解决 90% 的内网安装问题）：
    ```bash
    pip install ultralytics -i https://pypi.tuna.tsinghua.edu.cn/simple
    ```
2.  **如果上面失败（GitHub 源码安装）：**
    *   在命令行找个空目录，克隆代码：
        `git clone https://github.com/ultralytics/ultralytics.git`
    *   进入文件夹：`cd ultralytics`
    *   直接安装：`pip install -e .`
3.  **必备辅助工具（本地标注）：**
    由于 Roboflow 网页可能打不开，你需要一个**本地标注软件**。
    *   直接下载：[LabelImg (GitHub 官方)](https://github.com/HumanSignal/labelImg/releases)
    *   下载 `labelImg.exe` 即可，绿色免安装。

---

### **二、 核心方案：基于区域触发（ROI）的“电子围栏”**

不要训练复杂的“工人动作”，要训练“物体位置”。

*   **逻辑图解：**
    *   **Zone A（物料区）：** 识别到材料在这里消失。
    *   **Zone B（烤箱区）：** 材料必须在这里出现，且停留时间超过 X 分钟。
    *   **Zone C（成品区）：** 判定为合格。
*   **违规判定：** 识别到 `material`（材料）在 A 区消失后，直接出现在 C 区，跳过了 B 区。

---

### **三、 14天实操路线图**

#### **第 1-3 天：素材采集与本地标注**
1.  **录像：** 摄像头安装在能看到烤箱全景的角度。录制 1 小时工人正常作业和 10 组“偷懒”动作。
2.  **抽帧：** 从视频里每 10 秒截一张图，凑够 300 张。
3.  **标注：** 打开 `labelImg.exe`。
    *   把 `Save Format`（保存格式）切换为 **YOLO**。
    *   圈出 3 类目标：`material`（材料）、`oven`（烤箱口）、`hand`（手）。

#### **第 4-7 天：本地训练模型**
在你的 Python 环境里新建 `train_model.py`：
```python
from ultralytics import YOLO

# 加载官方预训练模型（会自动从 GitHub 链接下载）
model = YOLO('yolov8n.pt') 

# 开始训练
# data.yaml 需要你自己写，标明你的图片路径和 3 个标签名
model.train(data='my_data.yaml', epochs=100, imgsz=640, device='cpu') 
# 如果你公司电脑没显卡，device 就填 'cpu'，慢点但能跑通
```

#### **第 8-12 天：编写逻辑判断代码（SOP 监测）**
这是你给老板展示的核心 demo。逻辑如下：
```python
import cv2
from ultralytics import YOLO

model = YOLO("runs/detect/train/weights/best.pt") # 加载你自己训练好的

# 定义三个区域的坐标（根据画面自己调）
ZONE_A = [100, 100, 300, 300] # [x1, y1, x2, y2]
ZONE_B = [500, 100, 700, 300] 

# 核心逻辑变量
has_entered_oven = False

def check_sop(frame):
    global has_entered_oven
    results = model(frame)
    
    for box in results[0].boxes:
        cls = int(box.cls[0])
        conf = float(box.conf[0])
        # 获取物体中心点
        x_c, y_c = int(box.xywh[0][0]), int(box.xywh[0][1])

        # 如果材料进入了烤箱区
        if cls == 0 (material) and is_in_zone(x_c, y_c, ZONE_B):
            has_entered_oven = True
            
        # 如果材料出现在成品区（Zone C），但 has_entered_oven 为 False
        # 则触发【报警逻辑】
```

#### **第 13-14 天：现场调试与 Demo 演示**
*   **消除干扰：** 如果光线太乱，就调高 `conf`（置信度）阈值。
*   **界面展示：** 在画面上用 `cv2.rectangle` 画出红绿框。绿色代表材料进过烤箱，红色代表违规。

---

### **四、 应对内网限制的“杀手锏”**

如果 `pip` 完全无法连接外网，你的最后手段是：
1.  **在外网环境下（家里电脑）：**
    *   下载所有需要的 `.whl` 安装包。
    *   命令：`pip download ultralytics`
2.  **内网离线安装：**
    *   用 U 盘（如果公司允许）把这些文件拷进去。
    *   安装：`pip install --no-index --find-links=./download_folder ultralytics`



