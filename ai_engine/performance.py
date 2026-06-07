"""
性能优化模块

功能：
1. OCR 模型热启动和预热
2. 质量诊断结果缓存
3. 异步处理管道
4. 性能监控和统计
"""

import time
import threading
from collections import OrderedDict
from typing import Dict, Any, Optional, Callable
from functools import wraps
import hashlib


class SimpleCache:
    """简单的 LRU 缓存实现。"""
    
    def __init__(self, max_size: int = 100):
        self.max_size = max_size
        self.cache = OrderedDict()
        self.hits = 0
        self.misses = 0
        self.lock = threading.Lock()
    
    def get(self, key: str) -> Optional[Any]:
        """获取缓存值。"""
        with self.lock:
            if key in self.cache:
                self.hits += 1
                # 移到末尾（最近使用）
                self.cache.move_to_end(key)
                return self.cache[key]
            self.misses += 1
            return None
    
    def put(self, key: str, value: Any) -> None:
        """存放缓存值。"""
        with self.lock:
            if key in self.cache:
                self.cache.move_to_end(key)
            self.cache[key] = value
            
            # 超过大小限制，移除最旧的项
            if len(self.cache) > self.max_size:
                self.cache.popitem(last=False)
    
    def clear(self) -> None:
        """清空缓存。"""
        with self.lock:
            self.cache.clear()
            self.hits = 0
            self.misses = 0
    
    def stats(self) -> Dict[str, Any]:
        """获取缓存统计。"""
        with self.lock:
            total = self.hits + self.misses
            hit_rate = self.hits / total if total > 0 else 0
            return {
                "size": len(self.cache),
                "max_size": self.max_size,
                "hits": self.hits,
                "misses": self.misses,
                "hit_rate": hit_rate,
            }


class PerformanceMonitor:
    """性能监控器。"""
    
    def __init__(self):
        self.metrics = {}
        self.lock = threading.Lock()
    
    def record(self, metric_name: str, value: float) -> None:
        """记录性能指标。"""
        with self.lock:
            if metric_name not in self.metrics:
                self.metrics[metric_name] = []
            self.metrics[metric_name].append(value)
    
    def get_stats(self, metric_name: str) -> Dict[str, float]:
        """获取指标统计。"""
        with self.lock:
            if metric_name not in self.metrics or not self.metrics[metric_name]:
                return {}
            
            values = self.metrics[metric_name]
            return {
                "count": len(values),
                "mean": sum(values) / len(values),
                "min": min(values),
                "max": max(values),
                "total": sum(values),
            }
    
    def report(self) -> str:
        """生成性能报告。"""
        report = "📊 性能监控报告\n" + "=" * 50 + "\n"
        with self.lock:
            for metric_name, values in self.metrics.items():
                if values:
                    stats = self.get_stats(metric_name)
                    report += (
                        f"{metric_name}:\n"
                        f"  ├─ 平均: {stats.get('mean', 0):.1f}ms\n"
                        f"  ├─ 最小: {stats.get('min', 0):.1f}ms\n"
                        f"  ├─ 最大: {stats.get('max', 0):.1f}ms\n"
                        f"  └─ 总计: {stats.get('total', 0):.0f}ms ({stats.get('count', 0)} 次)\n"
                    )
        return report


class PerformanceOptimizer:
    """性能优化管理器。"""
    
    # 全局单例
    _instance = None
    _lock = threading.Lock()
    
    # 诊断缓存
    _diagnosis_cache = SimpleCache(max_size=200)
    
    # 性能监控
    _monitor = PerformanceMonitor()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance
    
    @classmethod
    def cache_diagnosis_result(
        cls,
        image_bytes: bytes,
        diagnosis_result: Dict[str, Any]
    ) -> None:
        """
        缓存图像质量诊断结果。
        
        用哈希值作为缓存键。
        """
        # 计算图像的哈希值
        image_hash = hashlib.md5(image_bytes).hexdigest()[:16]
        cls._diagnosis_cache.put(image_hash, diagnosis_result)
    
    @classmethod
    def get_cached_diagnosis(cls, image_bytes: bytes) -> Optional[Dict[str, Any]]:
        """获取缓存的诊断结果。"""
        image_hash = hashlib.md5(image_bytes).hexdigest()[:16]
        return cls._diagnosis_cache.get(image_hash)
    
    @classmethod
    def get_cache_stats(cls) -> Dict[str, Any]:
        """获取缓存统计。"""
        return cls._diagnosis_cache.stats()
    
    @classmethod
    def clear_cache(cls) -> None:
        """清空缓存。"""
        cls._diagnosis_cache.clear()
    
    @classmethod
    def record_time(cls, operation: str, duration_ms: float) -> None:
        """记录操作耗时。"""
        cls._monitor.record(operation, duration_ms)
    
    @classmethod
    def get_performance_report(cls) -> str:
        """获取性能报告。"""
        return cls._monitor.report()


def cached_diagnosis(func):
    """
    装饰器：为质量诊断函数添加缓存。
    
    使用方式：
        @cached_diagnosis
        def diagnose(image_bytes):
            ...
    """
    @wraps(func)
    def wrapper(image_bytes: bytes, *args, **kwargs):
        # 尝试获取缓存
        cached = PerformanceOptimizer.get_cached_diagnosis(image_bytes)
        if cached is not None:
            return cached
        
        # 执行函数
        start = time.time()
        result = func(image_bytes, *args, **kwargs)
        duration = (time.time() - start) * 1000
        
        # 记录性能和缓存结果
        PerformanceOptimizer.record_time(f"{func.__name__}", duration)
        PerformanceOptimizer.cache_diagnosis_result(image_bytes, result)
        
        return result
    
    return wrapper


def profile_time(operation_name: str):
    """
    装饰器：为函数计时并记录性能。
    
    使用方式：
        @profile_time("preprocess")
        def preprocess_image(image):
            ...
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start = time.time()
            result = func(*args, **kwargs)
            duration = (time.time() - start) * 1000
            PerformanceOptimizer.record_time(operation_name, duration)
            return result
        
        return wrapper
    
    return decorator


class AsyncProcessor:
    """异步处理器。"""
    
    def __init__(self, max_workers: int = 2):
        self.max_workers = max_workers
        self.queue = []
        self.lock = threading.Lock()
        self.condition = threading.Condition(self.lock)
    
    def submit(self, task_fn: Callable, *args, **kwargs) -> threading.Thread:
        """
        提交异步任务。
        
        Returns:
            任务线程（可选择等待）
        """
        def task_wrapper():
            try:
                task_fn(*args, **kwargs)
            except Exception as e:
                print(f"❌ 异步任务失败: {e}")
        
        thread = threading.Thread(target=task_wrapper, daemon=True)
        thread.start()
        return thread
    
    def wait_all(self, threads: list, timeout: float = 30) -> bool:
        """
        等待所有任务完成。
        
        Returns:
            是否全部完成
        """
        for thread in threads:
            thread.join(timeout=timeout)
        return all(not t.is_alive() for t in threads)


# 全局实例
_performance_optimizer = PerformanceOptimizer()
_async_processor = AsyncProcessor(max_workers=2)
