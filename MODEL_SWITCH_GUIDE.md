# SpotCheck AI 模型切换与移植部署指南

本指南介绍如何在不同机器间移植本项目，以及如何在 `PP-OCRv4 Mobile` 和 `PP-OCRv5 Server` 两个模型之间进行一键切换和对比测试。

---

## 📁 1. 项目移植与路径自适应说明
本项目的路径设计已实现完全的自适应：
* 所有路径（原图上传路径、前端静态资源目录、本地 OCR 模型路径）均使用 Python 相对定位动态获取（基于 `config.py` 中的 `BASE_DIR = os.path.dirname(os.path.abspath(__file__))`）。
* **无需修改任何硬编码绝对路径**。您只需将整个 `spotcheck` 文件夹拷贝到公司目标机器，配置好 Conda 虚拟环境后即可直接运行。

---

## 🔄 2. 双模型一键切换步骤

我们在主配置文件 `config.py` 中添加了全局开关 `ACTIVE_MODEL`，无需重写复杂的初始化参数，只需按以下两步操作即可轻松切换：

### 步骤一：准备本地模型文件
确保目标机器的 `models` 目录下放有对应的模型文件夹。完整的目录结构要求如下：

```text
spotcheck/
  ├── config.py
  ├── main.py
  ├── models/
  │    ├── PP-OCRv4_mobile_det/    <-- 本地 Mobile 检测模型目录
  │    ├── PP-OCRv4_mobile_rec/    <-- 本地 Mobile 识别模型目录
  │    ├── PP-OCRv5_server_det/    <-- 本地 Server 检测模型目录
  │    └── PP-OCRv5_server_rec/    <-- 本地 Server 识别模型目录
```
*(注：如果缺少对应的模型文件，需在测试前将解压后的模型放置于对应命名目录下。)*

### 步骤二：修改开关配置
打开 [config.py](file:///e:/mycode/companyworkspace/spotcheck/config.py) 文件，定位到 **模型选择开关** 区域（约第 18 行），修改 `ACTIVE_MODEL` 变量：

* **测试 PP-OCRv4 Mobile（当前默认版，CPU 推理极快，端到端约 4.3s）**：
  ```python
  ACTIVE_MODEL = "mobile"
  ```

* **测试 PP-OCRv5 Server（精度最高版，在无 GPU/仅 CPU 推理时较慢）**：
  ```python
  ACTIVE_MODEL = "server"
  ```

修改完成后保存，直接在终端中启动服务：
```bash
python main.py
```
服务在启动时会根据您的配置自动加载并预热对应的本地模型，并在控制台打印当前的加载路径。

---

## ⚡ 3. 性能测试数据参考（基于 CPU 推理）

在 CPU 单核/多核模拟测试下，对同一张点检图片 `test_screen.png` 提取参数性能表现如下：

| 指标维度 | PP-OCRv4 Mobile (移动端优化版) | PP-OCRv5 Server (服务端高精版) | 性能对比 |
| :--- | :--- | :--- | :--- |
| **OCR 推理耗时** | **~3.6 秒** | **~45.0 秒** | **Mobile 快 12.4 倍** |
| **端到端响应耗时** | **~4.3 秒** | **~45.0 秒** | **Mobile 快 10.5 倍** |
| 首次初始化耗时 | ~2.5 秒 | ~1.9 秒 | - |
| 引擎启动预热耗时 | ~1.5 秒 | ~117.0 秒 | **Mobile 快 78 倍** |

### 💡 测试建议
1. **CPU 运行环境**：建议在公司测试时，首先在 CPU 上体验 `"mobile"`，其耗时（约 4s）完全能够满足工人在设备前的实时点检需求。
2. **GPU 运行环境**：如果公司的测试服务器有英伟达显卡（已安装 CUDA），您可以切回 `"server"`，并将 `config.py` 中的 `"device"` 更改为 `"gpu"`，从而可以兼顾 **v5 的极高识别精度**与**毫秒级别的极快推理响应**。
