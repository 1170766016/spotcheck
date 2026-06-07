"""
参数异常检测模块

功能：
1. 异常值检测
2. 异常告警
3. 偏差分析
"""

from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import statistics


class AnomalyDetector:
    """异常检测器。"""
    
    # 异常类型
    ANOMALY_TYPES = {
        "outlier": "离群值",
        "sudden_change": "突变",
        "trend_change": "趋势反转",
        "threshold_exceeded": "超过阈值",
        "recurring_anomaly": "重复异常",
    }
    
    # 预设阈值规则
    ANOMALY_RULES = {
        # 温度
        "temperature": {
            "hard_limits": (-100, 500),  # 硬限
            "soft_limits": (0, 400),     # 软限
            "acceptable_deviation": 10,   # 可接受偏差
            "max_jump": 50,               # 最大跳跃
        },
        "温度": {
            "hard_limits": (-100, 500),
            "soft_limits": (0, 400),
            "acceptable_deviation": 10,
            "max_jump": 50,
        },
        # 压力
        "pressure": {
            "hard_limits": (0, 600),
            "soft_limits": (10, 500),
            "acceptable_deviation": 20,
            "max_jump": 100,
        },
        "压力": {
            "hard_limits": (0, 600),
            "soft_limits": (10, 500),
            "acceptable_deviation": 20,
            "max_jump": 100,
        },
        # 转速/速度
        "speed": {
            "hard_limits": (0, 10000),
            "soft_limits": (0, 8000),
            "acceptable_deviation": 200,
            "max_jump": 1000,
        },
        "转速": {
            "hard_limits": (0, 10000),
            "soft_limits": (0, 8000),
            "acceptable_deviation": 200,
            "max_jump": 1000,
        },
        "rpm": {
            "hard_limits": (0, 10000),
            "soft_limits": (0, 8000),
            "acceptable_deviation": 200,
            "max_jump": 1000,
        },
        # 电流
        "current": {
            "hard_limits": (0, 2000),
            "soft_limits": (0, 1500),
            "acceptable_deviation": 50,
            "max_jump": 200,
        },
        "电流": {
            "hard_limits": (0, 2000),
            "soft_limits": (0, 1500),
            "acceptable_deviation": 50,
            "max_jump": 200,
        },
    }
    
    def __init__(self):
        self.anomalies = []  # 记录检测到的异常
        self.baselines = {}  # 基准值
    
    def set_baseline(self, parameter_name: str, value: float) -> None:
        """
        设置参数的基准值（正常值）。
        
        用于偏差检测。
        """
        self.baselines[parameter_name] = value
    
    def detect_anomalies(
        self,
        parameter_name: str,
        values: List[float],
        unit: str = "",
        debug: bool = False
    ) -> Dict[str, Any]:
        """
        检测参数序列中的异常。
        
        Returns:
            {
                "parameter": 参数名,
                "anomalies": [异常列表],
                "anomaly_count": 异常数量,
                "anomaly_score": 异常指数 [0-1],
            }
        """
        anomalies = []
        
        if len(values) < 2:
            return {
                "parameter": parameter_name,
                "anomalies": [],
                "anomaly_count": 0,
                "anomaly_score": 0.0,
            }
        
        # 获取该参数的规则
        rules = self._get_rules_for_parameter(parameter_name)
        
        # 方法 1: 离群值检测 (IQR 方法)
        outliers = self._detect_outliers(values)
        if outliers:
            for idx in outliers:
                anomalies.append({
                    "type": "outlier",
                    "index": idx,
                    "value": values[idx],
                    "reason": "离群值（IQR 方法）",
                })
        
        # 方法 2: 硬限检测
        if rules:
            hard_limits = rules.get("hard_limits")
            if hard_limits:
                for idx, value in enumerate(values):
                    if value < hard_limits[0] or value > hard_limits[1]:
                        anomalies.append({
                            "type": "threshold_exceeded",
                            "index": idx,
                            "value": value,
                            "reason": f"超出硬限 {hard_limits}",
                        })
        
        # 方法 3: 突变检测
        sudden_changes = self._detect_sudden_changes(values, rules, debug=debug)
        anomalies.extend(sudden_changes)
        
        # 方法 4: 基准偏差检测
        if parameter_name in self.baselines:
            baseline = self.baselines[parameter_name]
            tolerance = rules.get("acceptable_deviation", 10) if rules else 10
            
            for idx, value in enumerate(values):
                deviation = abs(value - baseline)
                deviation_pct = deviation / abs(baseline) if baseline != 0 else 0
                
                if deviation > tolerance and deviation_pct > 0.1:  # 10% 偏差
                    anomalies.append({
                        "type": "deviation",
                        "index": idx,
                        "value": value,
                        "baseline": baseline,
                        "deviation": deviation,
                        "deviation_pct": deviation_pct,
                        "reason": f"偏离基准值 {baseline} ，偏差 {deviation_pct:.1%}",
                    })
        
        # 去重异常（同一位置的多个异常合并）
        anomalies = self._deduplicate_anomalies(anomalies)
        
        # 计算异常指数
        anomaly_score = min(len(anomalies) / max(len(values), 1) * 0.5, 1.0)
        
        if debug and anomalies:
            print(f"🚨 异常检测: {parameter_name}")
            for anom in anomalies:
                print(f"  - [{anom['index']}] {anom['reason']}: {anom['value']}")
        
        return {
            "parameter": parameter_name,
            "anomalies": anomalies,
            "anomaly_count": len(anomalies),
            "anomaly_score": anomaly_score,
            "detection_timestamp": datetime.now().isoformat(),
        }
    
    def detect_trend_change(
        self,
        parameter_name: str,
        values: List[float],
        window_size: int = 3,
        debug: bool = False
    ) -> Dict[str, Any]:
        """
        检测参数趋势的突然变化。
        
        Returns:
            {
                "parameter": 参数名,
                "trend_changes": [变化点列表],
                "trend_change_count": 数量,
            }
        """
        if len(values) < window_size * 2:
            return {
                "parameter": parameter_name,
                "trend_changes": [],
                "trend_change_count": 0,
            }
        
        trend_changes = []
        
        # 计算每个窗口的趋势
        for i in range(window_size, len(values) - window_size):
            # 前向趋势
            prev_trend = sum(values[i - j] - values[i - j - 1] for j in range(1, window_size))
            # 后向趋势
            next_trend = sum(values[i + j + 1] - values[i + j] for j in range(window_size))
            
            # 如果趋势反向且变化幅度大
            if prev_trend * next_trend < 0:  # 符号相反
                change_magnitude = abs(prev_trend) + abs(next_trend)
                if change_magnitude > 5:  # 阈值
                    trend_changes.append({
                        "index": i,
                        "value": values[i],
                        "prev_trend": prev_trend,
                        "next_trend": next_trend,
                        "magnitude": change_magnitude,
                    })
        
        if debug and trend_changes:
            print(f"📊 趋势变化: {parameter_name}")
            for change in trend_changes:
                print(f"  - [位置 {change['index']}] 幅度 {change['magnitude']:.1f}")
        
        return {
            "parameter": parameter_name,
            "trend_changes": trend_changes,
            "trend_change_count": len(trend_changes),
        }
    
    def _detect_outliers(
        self,
        values: List[float],
        threshold: float = 1.5
    ) -> List[int]:
        """
        使用 IQR（四分位距）方法检测离群值。
        
        Returns:
            离群值的索引列表
        """
        if len(values) < 4:
            return []
        
        sorted_vals = sorted(values)
        n = len(sorted_vals)
        
        # 计算 Q1, Q3
        q1_idx = n // 4
        q3_idx = 3 * n // 4
        
        q1 = sorted_vals[q1_idx]
        q3 = sorted_vals[q3_idx]
        iqr = q3 - q1
        
        # 离群值范围
        lower_bound = q1 - threshold * iqr
        upper_bound = q3 + threshold * iqr
        
        # 找出离群值
        outliers = []
        for idx, val in enumerate(values):
            if val < lower_bound or val > upper_bound:
                outliers.append(idx)
        
        return outliers
    
    def _detect_sudden_changes(
        self,
        values: List[float],
        rules: Optional[Dict[str, Any]] = None,
        debug: bool = False
    ) -> List[Dict[str, Any]]:
        """检测突变。"""
        sudden_changes = []
        max_jump = rules.get("max_jump", 100) if rules else 100
        
        for i in range(1, len(values)):
            jump = abs(values[i] - values[i - 1])
            if jump > max_jump:
                sudden_changes.append({
                    "type": "sudden_change",
                    "index": i,
                    "value": values[i],
                    "previous_value": values[i - 1],
                    "jump": jump,
                    "reason": f"突变 (从 {values[i-1]:.1f} 跳至 {values[i]:.1f})",
                })
        
        return sudden_changes
    
    def _deduplicate_anomalies(
        self,
        anomalies: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """去除同一位置的重复异常，保留最严重的。"""
        by_index = {}
        
        for anom in anomalies:
            idx = anom["index"]
            if idx not in by_index:
                by_index[idx] = anom
            else:
                # 保留更严重的异常（根据类型优先级）
                severity = {
                    "threshold_exceeded": 3,
                    "outlier": 2,
                    "sudden_change": 2,
                    "deviation": 1,
                }
                if severity.get(anom["type"], 0) > severity.get(by_index[idx]["type"], 0):
                    by_index[idx] = anom
        
        return list(by_index.values())
    
    def _get_rules_for_parameter(self, param_name: str) -> Optional[Dict[str, Any]]:
        """获取参数的异常检测规则。"""
        param_lower = param_name.lower()
        
        # 精确匹配
        if param_lower in self.ANOMALY_RULES:
            return self.ANOMALY_RULES[param_lower]
        
        # 模糊匹配
        for key, rules in self.ANOMALY_RULES.items():
            if key in param_lower:
                return rules
        
        return None
    
    def get_alert_summary(self) -> str:
        """获取告警摘要。"""
        if not self.anomalies:
            return "✅ 无异常"
        
        summary = f"⚠️ 检测到 {len(self.anomalies)} 个异常\n"
        
        # 按类型汇总
        by_type = {}
        for anom in self.anomalies:
            anom_type = anom.get("type", "unknown")
            by_type[anom_type] = by_type.get(anom_type, 0) + 1
        
        for anom_type, count in by_type.items():
            type_name = self.ANOMALY_TYPES.get(anom_type, anom_type)
            summary += f"  - {type_name}: {count} 个\n"
        
        return summary
