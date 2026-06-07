"""
参数验证和后处理模块

功能：
1. 数值类型检查和范围验证
2. 单位转换和标准化
3. 异常值过滤
4. 参数质量评分
"""

import re
from typing import Any, Dict, List, Tuple, Optional
from config import PARSER_CONFIG


# 常见参数的预期范围（用于异常检测）
PARAMETER_RANGES = {
    # 温度
    "temperature": {"min": -100, "max": 500, "unit": "℃"},
    "temp": {"min": -100, "max": 500, "unit": "℃"},
    "模具温度": {"min": 0, "max": 300, "unit": "℃"},
    "料筒温度": {"min": 0, "max": 400, "unit": "℃"},
    
    # 压力
    "pressure": {"min": 0, "max": 500, "unit": "MPa"},
    "压力": {"min": 0, "max": 500, "unit": "MPa"},
    "注射压力": {"min": 0, "max": 300, "unit": "MPa"},
    
    # 速度/转速
    "speed": {"min": 0, "max": 5000, "unit": "rpm"},
    "rpm": {"min": 0, "max": 10000, "unit": "rpm"},
    "转速": {"min": 0, "max": 10000, "unit": "rpm"},
    
    # 功率
    "power": {"min": 0, "max": 10000, "unit": "kW"},
    "功率": {"min": 0, "max": 10000, "unit": "kW"},
    
    # 电流
    "current": {"min": 0, "max": 1000, "unit": "A"},
    "电流": {"min": 0, "max": 1000, "unit": "A"},
    
    # 电压
    "voltage": {"min": 0, "max": 1000, "unit": "V"},
    "电压": {"min": 0, "max": 1000, "unit": "V"},
    
    # 时间
    "time": {"min": 0, "max": 3600, "unit": "s"},
    "时间": {"min": 0, "max": 3600, "unit": "s"},
}

# 单位转换表
UNIT_CONVERSIONS = {
    # 温度（转换到 ℃）
    "°F": lambda x: (x - 32) * 5 / 9,
    "K": lambda x: x - 273.15,
    
    # 压力（转换到 MPa）
    "kPa": lambda x: x / 1000,
    "Pa": lambda x: x / 1e6,
    "bar": lambda x: x / 10,
    "psi": lambda x: x / 145.038,
    
    # 功率（转换到 kW）
    "W": lambda x: x / 1000,
    "MW": lambda x: x * 1000,
    
    # 电流（转换到 A）
    "mA": lambda x: x / 1000,
    
    # 长度（转换到 mm）
    "cm": lambda x: x * 10,
    "m": lambda x: x * 1000,
    "μm": lambda x: x / 1000,
}


class ParameterValidator:
    """参数验证器。"""
    
    @staticmethod
    def validate_parameter(param: Dict[str, Any], debug: bool = False) -> Dict[str, Any]:
        """
        验证和后处理单个参数。
        
        Returns:
            增强后的参数字典，包含：
            - value_validated: 验证后的数值
            - value_normalized: 标准化后的值
            - unit_standard: 标准单位
            - quality_score: 参数质量评分 [0-1]
            - warnings: 警告列表
            - is_valid: 是否通过验证
        """
        result = {
            **param,
            "warnings": [],
            "is_valid": True,
            "quality_score": 0.95,  # 默认高质量
        }
        
        try:
            # Step 1: 清理和转换数值
            value = param.get("value", "")
            if isinstance(value, str):
                # 移除空格和逗号
                value = value.replace(" ", "").replace(",", "")
                # 提取数值
                match = re.search(r"[+-]?\d+\.?\d*", value)
                if match:
                    value = float(match.group())
                else:
                    result["is_valid"] = False
                    result["warnings"].append(f"无法解析数值: {param.get('value')}")
                    return result
            else:
                value = float(value)
            
            result["value_validated"] = value
            
            # Step 2: 单位标准化和转换
            unit = param.get("unit", "")
            # 从参数值中提取单位
            unit_match = re.search(r"([℃°FCKMPakPabarpsirpmshmsVAWKw]+)$", str(param.get("value", "")))
            if unit_match and not unit:
                unit = unit_match.group(1)
            
            standard_unit = unit
            converted_value = value
            
            if unit in UNIT_CONVERSIONS:
                conversion_fn = UNIT_CONVERSIONS[unit]
                try:
                    converted_value = conversion_fn(value)
                    # 根据转换后的单位自动判断标准单位
                    if unit in ["°F", "K"]:
                        standard_unit = "℃"
                    elif unit in ["kPa", "Pa", "bar", "psi"]:
                        standard_unit = "MPa"
                    elif unit in ["W", "MW"]:
                        standard_unit = "kW"
                    elif unit == "mA":
                        standard_unit = "A"
                    elif unit in ["cm", "m", "μm"]:
                        standard_unit = "mm"
                    
                    if debug:
                        print(f"[VALIDATE] 单位转换: {value}{unit} → {converted_value:.2f}{standard_unit}")
                except Exception as e:
                    result["warnings"].append(f"单位转换失败: {str(e)}")
                    result["quality_score"] *= 0.8
            
            result["value_normalized"] = converted_value
            result["unit_standard"] = standard_unit
            
            # Step 3: 范围验证
            param_name = param.get("name", "").lower()
            if param_name in PARAMETER_RANGES:
                param_range = PARAMETER_RANGES[param_name]
                min_val = param_range["min"]
                max_val = param_range["max"]
                
                if converted_value < min_val or converted_value > max_val:
                    result["warnings"].append(
                        f"数值超出预期范围 [{min_val}, {max_val}]: {converted_value}"
                    )
                    result["quality_score"] *= 0.7
            
            # Step 4: 精度评分
            confidence = param.get("confidence", 1.0)
            result["quality_score"] *= confidence
            
            # Step 5: 最终验证
            if result["quality_score"] < 0.5:
                result["is_valid"] = False
                if debug:
                    print(f"[VALIDATE] 质量评分过低: {result['quality_score']:.2f}")
            
            if debug and result["warnings"]:
                print(f"[VALIDATE] 警告: {result['warnings']}")
            
        except Exception as e:
            result["is_valid"] = False
            result["warnings"].append(f"验证异常: {str(e)}")
        
        return result
    
    @staticmethod
    def validate_parameters(parameters: List[Dict[str, Any]], debug: bool = False) -> Dict[str, Any]:
        """
        批量验证参数。
        
        Returns:
            {
                "parameters": [验证后的参数列表],
                "valid_count": 有效参数数量,
                "invalid_count": 无效参数数量,
                "warnings": [全局警告列表],
                "quality_score": 整体质量评分
            }
        """
        validated = []
        valid_count = 0
        invalid_count = 0
        all_warnings = []
        
        for param in parameters:
            result = ParameterValidator.validate_parameter(param, debug=debug)
            validated.append(result)
            
            if result["is_valid"]:
                valid_count += 1
            else:
                invalid_count += 1
            
            all_warnings.extend(result.get("warnings", []))
        
        # 计算整体质量评分
        if validated:
            avg_quality = sum(p.get("quality_score", 0) for p in validated) / len(validated)
        else:
            avg_quality = 0.0
        
        return {
            "parameters": validated,
            "valid_count": valid_count,
            "invalid_count": invalid_count,
            "total_count": len(parameters),
            "validity_ratio": valid_count / len(parameters) if parameters else 0,
            "warnings": all_warnings,
            "quality_score": avg_quality,
        }
