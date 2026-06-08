# 图像预处理参数调优与文字识别效果提升方案

为了提升对重影、反色、低对比度等劣质图片的识别准确率，我们将为系统添加可在前端页面实时调优的图像预处理控制面板，并在后台实现全套的图像处理管道。此外，按照要求，我们将清理页面中所有的 "AI" 字样。

## 用户评审要求

> [!IMPORTANT]
> **预处理参数传递机制**：
> 前端会将用户在页面上调整的各项图像处理参数序列化为 JSON 字符串，在上传/重新解析时作为 `FormData` 的 `params` 字段发送给 `/api/recognize` 接口。
> 后端会动态解析此参数，并覆盖默认的 `IMAGE_CONFIG`。
> 
> **重影与抖动去模糊算法**：
> 针对晃动重影，我们设计了**定向反遮罩高斯模糊（Directional Unsharp Mask）**，用户可以选择水平、垂直或双向的抖动方向，并调节其校正强度，通过从原图中扣除特定方向的模糊分量来锐化边缘，极大地缓解了重影带来的重影错字问题。
> 
> **局部黑底反相算法**：
> 针对图片中的黑底区域进行白底黑字反相处理。使用大核盒滤波器（Box Filter）提取局部区域背景均值。当局部亮度低于用户设定的阈值时，自动将该区域的 BGR 颜色进行反相处理，非黑底区域保持不变。

## 待讨论与确认问题

> [!NOTE]
> 我们将以下列默认值作为调优的起始参数。用户可在界面上随时调整这些参数并进行“重新解析”。
> - 灰度化：默认关闭
> - 黑底反相：默认关闭（可选：自动全局、强制全局、局部反相）
> - 去模糊（去重影）：默认关闭（可设置方向与强度）
> - 锐化：默认关闭（拉普拉斯锐化、反遮罩、经典卷积内核）
> - 对比度增强：默认 CLAHE
> - 去噪：默认关闭（双边滤波以保留边缘）
> - 伽马校正：默认关闭
> - 形态学：默认关闭（膨胀、腐蚀、开闭运算）

---

## 拟作出的更改

### 1. 后端修改

#### [MODIFY] [preprocessor.py](file:///e:/mycode/companyworkspace/spotcheck/ai_engine/preprocessor.py)
- 重构 `preprocess_image` 函数以接收可选的 `params` 字典。
- 引入 `_get_merged_config` 动态合并机制，不破坏现有静态配置。
- 实现以下预处理管道：
  - **灰度化** (`grayscale_enabled`)：转换为灰度，然后为了 PaddleOCR 兼容性复制为 3 通道 BGR。
  - **反相处理** (`invert_mode`, `invert_black_bg_thresh`)：
    - `none`: 不反相
    - `invert_black_bg`: 自动判定（原逻辑）整个屏幕反相
    - `always`: 强制全局反相
    - `local_black_bg`: 局部黑底区域反相（使用局部背景均值计算 mask）
  - **去重影去模糊** (`deblur_enabled`, `deblur_direction`, `deblur_strength`)：利用水平/垂直的高斯模糊进行差分补偿。
  - **锐化增强** (`sharpen_enabled`, `sharpen_mode`, `sharpen_strength`)：
    - `laplacian` 锐化：原图 - 强度 * 拉普拉斯算子
    - `unsharp_mask` 锐化：利用高斯模糊与原图的差值进行增强
    - `kernel` 锐化：支持强度的 3x3 经典卷积算子
  - **对比度控制** (`contrast_mode`, `clahe_clip_limit`, `clahe_grid_size`, `linear_alpha`, `linear_beta`)：CLAHE 及线性对比度调整。
  - **自适应去噪** (`denoise_enabled`, `denoise_strength`)：改用**双边滤波（Bilateral Filter）**，在去除高频噪声的同时完美保留文字边缘。
  - **伽马校正** (`gamma_enabled`, `gamma_value`)：调整过亮或过暗屏幕。
  - **形态学处理** (`morphology_enabled`, `morphology_type`, `morphology_size`)：加粗或连接文字断笔。

#### [MODIFY] [main.py](file:///e:/mycode/companyworkspace/spotcheck/main.py)
- 更新 `/api/recognize` 路由以接收 `params: str = Form(None)` 参数。
- 解析 `params` 为 JSON 格式，并输出详细的后台日志，显示收到的调优参数与应用的管道步骤。
- 确保将经过预处理调优后的图片直接传入 `ocr_engine.recognize(processed_img)` 以看真实调优效果。

---

### 2. 前端修改

#### [MODIFY] [index.html](file:///e:/mycode/companyworkspace/spotcheck/frontend/index.html)
- 清理所有可见页面文本和标题中的 "AI" 字样（例如：`SpotCheck AI` -> `SpotCheck`，`AI就绪` -> `系统就绪`，`AI标注` -> `系统标注` 等）。
- 在结果页面（或主界面下方）添加一个精致的、可折叠的“图像预处理调优面板”。
- 面板中将包含所有预处理维度的控制项（下拉菜单、滑动条、复选框等）以及**各参数的清晰解释作用**。
- 添加一个“重新解析”按钮，可以在不用重新选择文件的前提下，使用当前选择的图片及修改后的参数再次运行 OCR。

#### [MODIFY] [index.css](file:///e:/mycode/companyworkspace/spotcheck/frontend/index.css)
- 配合整体极简黑白的设计风格，为预处理控制面板、滑动条、复选框、提示信息等设计精美的样式。
- 使用双栏布局（在宽屏下左侧为控制面板，右侧为图片对比与参数列表；窄屏下自动堆叠），提升操作效率。

#### [MODIFY] [app.js](file:///e:/mycode/companyworkspace/spotcheck/frontend/app.js)
- 清理 "AI 正在识别..." 等 "AI" 提示文案。
- 保存当前正在操作的 `currentFile` 对象。
- 实现读取面板中所有参数并序列化为 JSON 的方法。
- 支持点击“重新解析”或修改参数后发送最新的配置到后端。
- 渲染并展示当前应用了哪些增强方案。

---

## 验证计划

### 自动化与手动测试
1. 启动 FastAPI 后端服务，通过命令行或运行测试脚本确保没有语法错误。
2. 在浏览器打开系统，上传一张带重影/曝光不足/黑底的设备屏幕图。
3. 在页面上调整“黑底反相 -> 局部反相”及“重影校正 -> 水平抖动”等参数，点击重新解析，观察标注图片中原本错字漏字的地方是否被正确识别，对比度是否显著上升。
4. 检查后端控制台日志，确认已输出详细的参数合并与管道执行日志。
5. 检查页面中是否已经完全不存在 "AI" 字符。
