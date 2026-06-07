"""
综合测试脚本 - 验证所有改进效果

测试模块：
1. 质量诊断 - 亮度、对比度、清晰度检测
2. 屏显检测 - 黑色背景检测
3. 自适应预处理 - 根据诊断结果动态调整
4. 参数提取 - 位置感知的参数配对
5. 集成测试 - 完整流程

使用方法：
    python test_improvements.py [options]

示例：
    python test_improvements.py --mode all                    # 运行全部测试
    python test_improvements.py --mode diagnostic --verbose   # 诊断模式 + 详细日志
    python test_improvements.py --mode extraction             # 仅测试参数提取
"""

import sys
import argparse
import os

# 设置 OpenCV 后端（在导入前）
os.environ["QT_QPA_PLATFORM"] = "offscreen"
os.environ["DISPLAY"] = ""

# 尝试导入 OpenCV，如果失败则退出
try:
    import cv2
    import numpy as np
except ImportError as e:
    print(f"❌ 依赖项缺失: {e}", file=sys.stderr)
    print("💡 请运行: pip install opencv-python numpy", file=sys.stderr)
    sys.exit(1)

from pathlib import Path

# 导入模块
from config import IMAGE_CONFIG, PARSER_CONFIG
from ai_engine.preprocessor import (
    preprocess_image,
    _diagnose_image_quality,
    _detect_screen_display,
)
from ai_engine.parser import parse_ocr_results, get_raw_texts


def create_test_image(
    width: int = 640,
    height: int = 480,
    image_type: str = "normal"
) -> np.ndarray:
    """
    创建测试图像。
    
    Types:
        - "normal": 正常对比度图像
        - "dark": 暗图像
        - "overexposed": 曝光过度
        - "low_contrast": 低对比度
        - "screen_display": 屏显（黑色背景+浅色字体）
        - "blurry": 模糊图像
    """
    image = np.ones((height, width, 3), dtype=np.uint8)

    if image_type == "normal":
        # 浅灰色背景 + 深灰色文字
        image[:, :] = (200, 200, 200)  # 浅灰背景
        cv2.putText(
            image, "Temperature: 180C",
            (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (50, 50, 50), 2
        )

    elif image_type == "dark":
        # 暗图像（平均亮度 < 80）
        image[:, :] = (40, 40, 40)  # 暗灰背景
        cv2.putText(
            image, "Pressure: 50MPa",
            (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (100, 100, 100), 2
        )

    elif image_type == "overexposed":
        # 曝光过度（平均亮度 > 200）
        image[:, :] = (240, 240, 240)  # 很亮的背景
        cv2.putText(
            image, "Speed: 120rpm",
            (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (220, 220, 220), 2
        )

    elif image_type == "low_contrast":
        # 低对比度
        image[:, :] = (150, 150, 150)  # 中等灰
        cv2.putText(
            image, "Voltage: 380V",
            (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (140, 140, 140), 2
        )

    elif image_type == "screen_display":
        # 屏显：黑色背景 + 浅色字体（高对比）
        image[:, :] = (0, 0, 0)  # 纯黑背景（> 40% 黑色像素）
        # 模拟多个参数行
        cv2.putText(image, "Temperature: 180C", (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (220, 220, 220), 2)
        cv2.putText(image, "Pressure: 50MPa", (30, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (220, 220, 220), 2)
        cv2.putText(image, "Speed: 120rpm", (30, 220), cv2.FONT_HERSHEY_SIMPLEX, 1, (220, 220, 220), 2)

    elif image_type == "blurry":
        # 模糊图像（Laplacian 方差低）
        image[:, :] = (150, 150, 150)
        cv2.putText(
            image, "Power: 25kW",
            (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (100, 100, 100), 2
        )
        # 高斯模糊
        image = cv2.GaussianBlur(image, (15, 15), 0)

    return image


def test_quality_diagnosis(verbose: bool = False):
    """测试质量诊断功能。"""
    print("\n" + "=" * 60)
    print("🔍 质量诊断测试")
    print("=" * 60)

    test_cases = [
        ("normal", "正常图像"),
        ("dark", "暗图像"),
        ("overexposed", "曝光过度"),
        ("low_contrast", "低对比度"),
        ("blurry", "模糊图像"),
    ]

    for image_type, description in test_cases:
        image = create_test_image(image_type=image_type)
        brightness_level, contrast_level, clarity_level = _diagnose_image_quality(image)

        print(f"\n{description}:")
        print(f"  ├─ 亮度: {brightness_level}")
        print(f"  ├─ 对比度: {contrast_level}")
        print(f"  └─ 清晰度: {clarity_level}")

        if verbose:
            # 计算诊断统计
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            mean_brightness = cv2.mean(gray)[0]
            std_brightness = np.std(gray)
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            print(f"     [详细] 平均亮度={mean_brightness:.1f}, 标准差={std_brightness:.1f}, Laplacian方差={laplacian_var:.1f}")


def test_screen_display_detection(verbose: bool = False):
    """测试屏显检测功能。"""
    print("\n" + "=" * 60)
    print("📺 屏显检测测试")
    print("=" * 60)

    test_cases = [
        ("normal", "正常图像", False),
        ("screen_display", "屏显图像", True),
        ("dark", "暗图像", False),
    ]

    for image_type, description, expected in test_cases:
        image = create_test_image(image_type=image_type)
        is_screen = _detect_screen_display(image)

        status = "✅" if is_screen == expected else "❌"
        print(f"\n{status} {description}: is_screen_display={is_screen} (预期={expected})")

        if verbose:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            black_pixels = np.sum(gray < 50)
            total_pixels = gray.shape[0] * gray.shape[1]
            black_ratio = black_pixels / total_pixels
            print(f"     [详细] 黑色占比={black_ratio:.2%}, 阈值={IMAGE_CONFIG['screen_black_threshold']:.2%}")


def test_adaptive_preprocessing(verbose: bool = False):
    """测试自适应预处理效果。"""
    print("\n" + "=" * 60)
    print("🎨 自适应预处理测试")
    print("=" * 60)

    test_cases = [
        ("normal", "正常图像"),
        ("dark", "暗图像（需CLAHE）"),
        ("overexposed", "曝光过度（需伽马校正）"),
        ("blurry", "模糊图像（需锐化）"),
        ("screen_display", "屏显（需自适应阈值）"),
    ]

    for image_type, description in test_cases:
        image = create_test_image(image_type=image_type)
        # 转为字节（模拟上传）
        _, buffer = cv2.imencode(".jpg", image)
        image_bytes = buffer.tobytes()

        # 预处理
        processed, info = preprocess_image(image_bytes)

        print(f"\n{description}:")
        print(f"  ├─ 原始尺寸: {info['original_size']}")
        print(f"  ├─ 处理后尺寸: {info['processed_size']}")
        print(f"  ├─ 亮度等级: {info['brightness_level']}")
        print(f"  ├─ 屏显: {info['is_screen_display']}")
        print(f"  ├─ 清晰度: {info['clarity_level']}")
        print(f"  ├─ 应用增强: {', '.join(info['applied_enhancements']) if info['applied_enhancements'] else '无'}")
        print(f"  └─ 处理耗时: {info['preprocess_time_ms']}ms")


def test_parameter_extraction():
    """测试参数提取（模拟 OCR 结果）。"""
    print("\n" + "=" * 60)
    print("🔗 参数提取测试")
    print("=" * 60)

    # 模拟 OCR 结果（PaddleOCR 格式）
    mock_ocr_results = [[
        # 格式: [box, (text, confidence)]
        ([10, 10], ("Temperature", 0.95), 0.95),   # 参数名
        ([10, 50], ("180.5", 0.98), 0.98),        # 参数值
        ([10, 100], ("℃", 0.99), 0.99),           # 单位
        ([200, 10], ("Pressure", 0.92), 0.92),    # 参数名
        ([200, 50], ("50", 0.97), 0.97),          # 参数值
        ([200, 100], ("MPa", 0.96), 0.96),        # 单位
    ]]

    # 解析参数
    params = parse_ocr_results(mock_ocr_results)

    print(f"\n识别到 {len(params)} 个参数：")
    for i, param in enumerate(params, 1):
        print(f"\n  参数 {i}:")
        print(f"    ├─ 名称: {param['name']}")
        print(f"    ├─ 值: {param['value']}")
        print(f"    ├─ 单位: {param['unit']}")
        print(f"    ├─ 置信度: {param['confidence']:.2%}")
        print(f"    └─ 来源: {param['source']}")


def test_config_info():
    """显示当前配置。"""
    print("\n" + "=" * 60)
    print("⚙️  当前配置")
    print("=" * 60)

    print("\n📸 图像处理配置:")
    print(f"  ├─ 最大尺寸: {IMAGE_CONFIG['max_size']}px")
    print(f"  ├─ 启用质量诊断: {IMAGE_CONFIG['enable_quality_diagnosis']}")
    print(f"  ├─ 启用屏显检测: {IMAGE_CONFIG['enable_screen_detection']}")
    print(f"  ├─ 启用对比度增强: {IMAGE_CONFIG['enhance_contrast']}")
    print(f"  ├─ CLAHE 网格大小: {IMAGE_CONFIG['clahe_grid_size']}")
    print(f"  └─ 屏显阈值: {IMAGE_CONFIG['screen_black_threshold']:.2%}")

    print("\n🔗 参数提取配置:")
    print(f"  ├─ 最低置信度: {PARSER_CONFIG['min_confidence']}")
    print(f"  ├─ 同行容差: {PARSER_CONFIG['row_tolerance_ratio']}")
    print(f"  └─ 最大配对距离: {PARSER_CONFIG['max_pair_distance']}px")


def main():
    parser = argparse.ArgumentParser(
        description="OCR 改进效果验证脚本",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例：
  python test_improvements.py --mode all              运行全部测试
  python test_improvements.py --mode diagnostic -v    诊断模式 + 详细信息
  python test_improvements.py --mode extraction       仅参数提取测试
  python test_improvements.py --mode config           显示配置信息
        """
    )

    parser.add_argument(
        "--mode",
        choices=["all", "diagnostic", "screen", "preprocessing", "extraction", "config"],
        default="all",
        help="测试模式（default: all）"
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="显示详细诊断信息"
    )

    args = parser.parse_args()

    print("\n" + "🚀 " * 20)
    print("SpotCheck AI 改进效果验证")
    print("🚀 " * 20)

    try:
        if args.mode in ["all", "diagnostic"]:
            test_config_info()
            test_quality_diagnosis(args.verbose)

        if args.mode in ["all", "screen"]:
            test_screen_display_detection(args.verbose)

        if args.mode in ["all", "preprocessing"]:
            test_adaptive_preprocessing(args.verbose)

        if args.mode in ["all", "extraction"]:
            test_parameter_extraction()

        if args.mode == "config":
            test_config_info()

        print("\n" + "=" * 60)
        print("✅ 所有测试完成！")
        print("=" * 60 + "\n")

    except Exception as e:
        print(f"\n❌ 测试失败: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
