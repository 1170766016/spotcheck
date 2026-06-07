"""
参数时间序列分析模块

功能：
1. 参数变化历史记录
2. 趋势分析
3. 时间段统计
"""

from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict
import json


class TimeSeriesAnalyzer:
    """时间序列分析器。"""
    
    def __init__(self):
        """初始化时间序列分析器。"""
        self.history = defaultdict(list)  # {参数名: [(时间, 值, 单位), ...]}
        self.metadata = defaultdict(dict)  # {参数名: {描述, 单位等}}
    
    def record(
        self,
        parameter_name: str,
        value: float,
        unit: str = "",
        timestamp: Optional[datetime] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        记录参数值。
        
        Args:
            parameter_name: 参数名
            value: 参数值
            unit: 单位
            timestamp: 时间戳（默认为当前时间）
            metadata: 额外元数据
        """
        if timestamp is None:
            timestamp = datetime.now()
        
        self.history[parameter_name].append({
            "timestamp": timestamp,
            "value": value,
            "unit": unit,
            "metadata": metadata or {},
        })
        
        # 更新元数据
        if unit:
            self.metadata[parameter_name]["unit"] = unit
        if metadata:
            self.metadata[parameter_name].update(metadata)
    
    def get_history(
        self,
        parameter_name: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 1000
    ) -> List[Dict[str, Any]]:
        """
        获取参数历史。
        
        Args:
            parameter_name: 参数名
            start_time: 开始时间
            end_time: 结束时间
            limit: 最大记录数
        
        Returns:
            历史记录列表
        """
        history = self.history.get(parameter_name, [])
        
        # 时间范围过滤
        if start_time or end_time:
            filtered = []
            for record in history:
                ts = record["timestamp"]
                if start_time and ts < start_time:
                    continue
                if end_time and ts > end_time:
                    continue
                filtered.append(record)
            history = filtered
        
        # 返回最后 limit 条记录
        return history[-limit:]
    
    def calculate_trend(
        self,
        parameter_name: str,
        duration_minutes: int = 60,
        debug: bool = False
    ) -> Dict[str, Any]:
        """
        计算参数趋势。
        
        Returns:
            {
                "parameter": 参数名,
                "current_value": 当前值,
                "previous_value": 前一个值,
                "trend": "rising" | "falling" | "stable",
                "trend_strength": 趋势强度 [0-1],
                "change_rate": 变化率,
                "avg_value": 平均值,
                "min_value": 最小值,
                "max_value": 最大值,
                "volatility": 波动性,
            }
        """
        # 获取所有历史记录，然后按时间范围过滤
        all_history = self.history.get(parameter_name, [])
        
        if not all_history:
            return {
                "parameter": parameter_name,
                "trend": "no_data",
                "message": "没有历史数据",
            }
        
        # 使用最后一条记录的时间作为 end_time
        end_time = all_history[-1]["timestamp"]
        start_time = end_time - timedelta(minutes=duration_minutes)
        
        history = self.get_history(parameter_name, start_time, end_time)
        
        if not history:
            return {
                "parameter": parameter_name,
                "trend": "no_data",
                "message": "没有历史数据",
            }
        
        values = [r["value"] for r in history]
        
        # 计算统计量
        current = values[-1]
        previous = values[-2] if len(values) > 1 else current
        avg_val = sum(values) / len(values)
        min_val = min(values)
        max_val = max(values)
        
        # 计算波动性（标准差）
        if len(values) > 1:
            variance = sum((v - avg_val) ** 2 for v in values) / len(values)
            volatility = variance ** 0.5
        else:
            volatility = 0.0
        
        # 判断趋势
        if len(values) > 1:
            # 简单线性回归趋势
            n = len(values)
            x_avg = n / 2
            y_avg = avg_val
            
            numerator = sum((i - x_avg) * (values[i] - y_avg) for i in range(n))
            denominator = sum((i - x_avg) ** 2 for i in range(n))
            
            if denominator > 0:
                slope = numerator / denominator
                # 趋势强度基于斜率和波动性
                trend_strength = min(abs(slope) / (max(volatility, 0.01)), 1.0)
            else:
                slope = 0
                trend_strength = 0
            
            # 根据斜率判断趋势
            # 使用相对阈值，如果范围足够大的话
            if max_val - min_val > 0:
                relative_slope = abs(slope) / (max_val - min_val)
                threshold = 0.01
            else:
                relative_slope = abs(slope)
                threshold = 0.1
            
            if relative_slope > threshold:
                if slope > 0:
                    trend = "rising"
                else:
                    trend = "falling"
            else:
                trend = "stable"
        else:
            trend = "stable"
            trend_strength = 0.0
        
        change_rate = (current - previous) / abs(previous) if previous != 0 else 0
        
        result = {
            "parameter": parameter_name,
            "current_value": current,
            "previous_value": previous,
            "change_rate": change_rate,
            "trend": trend,
            "trend_strength": trend_strength,
            "avg_value": avg_val,
            "min_value": min_val,
            "max_value": max_val,
            "volatility": volatility,
            "record_count": len(values),
            "duration_minutes": duration_minutes,
        }
        
        if debug:
            print(f"📈 趋势分析: {parameter_name}")
            print(f"  ├─ 当前值: {current}")
            print(f"  ├─ 趋势: {trend} (强度: {trend_strength:.2f})")
            print(f"  ├─ 平均值: {avg_val:.2f}")
            print(f"  ├─ 波动性: {volatility:.2f}")
            print(f"  └─ 变化率: {change_rate:.1%}")
        
        return result
    
    def get_time_statistics(
        self,
        parameter_name: str,
        time_bucket: str = "hour",
        debug: bool = False
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        按时间段获取统计（如按小时、按天）。
        
        Args:
            parameter_name: 参数名
            time_bucket: 时间段 ("minute" | "hour" | "day")
        
        Returns:
            {
                "hour_1": {avg, min, max, count},
                "hour_2": {...},
                ...
            }
        """
        history = self.history.get(parameter_name, [])
        if not history:
            return {}
        
        # 按时间段分组
        buckets = defaultdict(list)
        
        for record in history:
            ts = record["timestamp"]
            value = record["value"]
            
            if time_bucket == "minute":
                bucket_key = ts.strftime("%Y-%m-%d %H:%M")
            elif time_bucket == "hour":
                bucket_key = ts.strftime("%Y-%m-%d %H:00")
            elif time_bucket == "day":
                bucket_key = ts.strftime("%Y-%m-%d")
            else:
                bucket_key = "unknown"
            
            buckets[bucket_key].append(value)
        
        # 计算每个时间段的统计
        statistics = {}
        for bucket_key, values in sorted(buckets.items()):
            statistics[bucket_key] = {
                "count": len(values),
                "avg": sum(values) / len(values),
                "min": min(values),
                "max": max(values),
                "sum": sum(values),
            }
        
        if debug:
            print(f"⏰ 时间统计: {parameter_name} (按{time_bucket})")
            for bucket, stats in statistics.items():
                print(f"  {bucket}: avg={stats['avg']:.2f}, "
                      f"min={stats['min']:.2f}, max={stats['max']:.2f}, "
                      f"count={stats['count']}")
        
        return statistics
    
    def export_history(self, parameter_name: str) -> str:
        """导出参数历史为 JSON。"""
        history = self.history.get(parameter_name, [])
        
        # 转换 datetime 为字符串
        exportable = []
        for record in history:
            exportable.append({
                "timestamp": record["timestamp"].isoformat(),
                "value": record["value"],
                "unit": record["unit"],
                "metadata": record["metadata"],
            })
        
        return json.dumps(exportable, indent=2)
    
    def get_summary_report(self, debug: bool = False) -> str:
        """获取全部参数的摘要报告。"""
        report = "📊 时间序列分析摘要\n" + "=" * 50 + "\n"
        
        for param_name in sorted(self.history.keys()):
            history = self.history[param_name]
            if history:
                values = [r["value"] for r in history]
                avg_val = sum(values) / len(values)
                
                trend_info = self.calculate_trend(param_name, debug=False)
                trend = trend_info.get("trend", "unknown")
                
                unit = self.metadata.get(param_name, {}).get("unit", "")
                
                report += (
                    f"{param_name}:\n"
                    f"  ├─ 记录数: {len(history)}\n"
                    f"  ├─ 当前值: {values[-1]:.2f}{unit}\n"
                    f"  ├─ 平均值: {avg_val:.2f}{unit}\n"
                    f"  ├─ 范围: [{min(values):.2f}, {max(values):.2f}]{unit}\n"
                    f"  └─ 趋势: {trend}\n"
                )
        
        return report


# 全局时间序列分析器实例
_time_series_analyzer = TimeSeriesAnalyzer()


def get_time_series_analyzer() -> TimeSeriesAnalyzer:
    """获取全局时间序列分析器。"""
    global _time_series_analyzer
    if _time_series_analyzer is None:
        _time_series_analyzer = TimeSeriesAnalyzer()
    return _time_series_analyzer
