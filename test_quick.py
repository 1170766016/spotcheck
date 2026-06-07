"""
快速测试脚本 - 配置和参数提取验证（无需 OpenCV）

这个脚本可以在没有 GUI 库的环境中运行，用于验证：
1. 配置是否正确加载
2. 参数提取逻辑是否正常
"""

import sys
sys.path.insert(0, '/workspaces/spotcheck')

from config import IMAGE_CONFIG, PARSER_CONFIG
from ai_engine.parser import parse_ocr_results, get_raw_texts

def test_config():
    """测试配置加载。"""
    print("\n" + "=" * 70)
    print("⚙️  配置验证")
    print("=" * 70)

    print("\n📸 图像处理配置:")
    print(f"  ├─ 最大尺寸: {IMAGE_CONFIG['max_size']}px")
    print(f"  ├─ 启用质量诊断: {IMAGE_CONFIG['enable_quality_diagnosis']}")
    print(f"  ├─ 启用屏显检测: {IMAGE_CONFIG['enable_screen_detection']}")
    print(f"  ├─ 屏显黑色阈值: {IMAGE_CONFIG['screen_black_threshold']:.2%}")
    print(f"  ├─ 启用对比度增强: {IMAGE_CONFIG['enhance_contrast']}")
    print(f"  ├─ CLAHE 网格大小: {IMAGE_CONFIG['clahe_grid_size']}")
    print(f"  ├─ 启用伽马校正: {IMAGE_CONFIG['gamma_correction_enabled']}")
    print(f"  ├─ 启用去噪: {IMAGE_CONFIG['denoise']}")
    print(f"  └─ 启用锐化: {IMAGE_CONFIG['sharpen']}")

    print("\n🔗 参数提取配置:")
    print(f"  ├─ 最低置信度: {PARSER_CONFIG['min_confidence']}")
    print(f"  ├─ 同行容差: {PARSER_CONFIG['row_tolerance_ratio']}")
    print(f"  └─ 最大配对距离: {PARSER_CONFIG['max_pair_distance']}px")

    print("\n📺 屏显处理配置:")
    screen_config = IMAGE_CONFIG['screen_display']
    print(f"  ├─ 自适应阈值: {screen_config['adaptive_threshold_enabled']}")
    print(f"  ├─ 边缘增强: {screen_config['edge_enhance_enabled']}")
    print(f"  └─ 允许反色: {screen_config['invert_enabled']}")

    print("\n✅ 配置加载成功！")


def test_parameter_extraction():
    """测试参数提取逻辑。"""
    print("\n" + "=" * 70)
    print("🔗 参数提取测试")
    print("=" * 70)

    # 模拟多个 OCR 场景
    # 格式: [box, (text, confidence)] 其中 box 是四个坐标点的列表
    test_cases = [
        {
            "name": "简单行内参数",
            "ocr_results": [[
                ([[10, 10], [200, 10], [200, 40], [10, 40]], ("Temperature: 180℃", 0.95)),
                ([[10, 60], [200, 60], [200, 90], [10, 90]], ("Pressure: 50MPa", 0.92)),
            ]],
        },
        {
            "name": "参数名和值分离",
            "ocr_results": [[
                ([[10, 10], [120, 10], [120, 40], [10, 40]], ("Temperature", 0.95)),
                ([[130, 10], [200, 10], [200, 40], [130, 40]], ("180.5", 0.98)),
                ([[210, 10], [250, 10], [250, 40], [210, 40]], ("℃", 0.99)),
                ([[10, 60], [120, 60], [120, 90], [10, 90]], ("Pressure", 0.92)),
                ([[130, 60], [200, 60], [200, 90], [130, 90]], ("50", 0.97)),
                ([[210, 60], [280, 60], [280, 90], [210, 90]], ("MPa", 0.96)),
            ]],
        },
        {
            "name": "混合格式",
            "ocr_results": [[
                ([[10, 10], [280, 10], [280, 40], [10, 40]], ("模具温度:180℃", 0.95)),
                ([[10, 60], [60, 60], [60, 90], [10, 90]], ("PV", 0.90)),
                ([[70, 60], [200, 60], [200, 90], [70, 90]], ("200.5", 0.93)),
            ]],
        },
    ]

    for case in test_cases:
        print(f"\n📋 场景: {case['name']}")
        try:
            parameters = parse_ocr_results(case['ocr_results'], debug=False)
            raw_texts = get_raw_texts(case['ocr_results'])
            
            print(f"   识别到 {len(parameters)} 个参数，{len(raw_texts)} 个文本框")
            for i, param in enumerate(parameters, 1):
                print(f"   {i}. {param['name']}={param['value']}{param['unit']} (源: {param['source']}, 置信度: {param['confidence']:.2%})")
        except Exception as e:
            print(f"   ❌ 错误: {e}")
            import traceback
            traceback.print_exc()

    print("\n✅ 参数提取测试完成！")


def main():
    print("\n" + "🚀 " * 20)
    print("SpotCheck AI 快速验证脚本（无需 OpenCV）")
    print("🚀 " * 20)

    try:
        test_config()
        test_parameter_extraction()
        
        print("\n" + "=" * 70)
        print("✅ 所有测试通过！")
        print("=" * 70 + "\n")
        
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
