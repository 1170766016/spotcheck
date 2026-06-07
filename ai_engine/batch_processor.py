"""
批量处理模块

功能：
1. 多图像处理（顺序/并行）
2. 进度跟踪
3. 结果聚合和统计
"""

import threading
import time
from typing import List, Dict, Any, Callable, Optional, Tuple
from queue import Queue
import traceback


class BatchProcessor:
    """批处理器。"""
    
    def __init__(self, max_workers: int = 2):
        """
        初始化批处理器。
        
        Args:
            max_workers: 最大并发工作线程数
        """
        self.max_workers = max_workers
        self.current_job_id = 0
        self.jobs = {}
        self.lock = threading.Lock()
    
    def process_batch(
        self,
        items: List[Any],
        processor_fn: Callable[[Any, Dict[str, Any]], Any],
        mode: str = "sequential",
        debug: bool = False
    ) -> Dict[str, Any]:
        """
        批量处理多个项目。
        
        Args:
            items: 待处理的项目列表
            processor_fn: 处理函数，签名: processor_fn(item, context) -> result
            mode: 处理模式 ("sequential" | "parallel")
            debug: 是否输出调试信息
        
        Returns:
            {
                "job_id": 任务 ID,
                "total": 总项数,
                "successful": 成功数,
                "failed": 失败数,
                "results": [结果列表],
                "errors": [错误列表],
                "duration_ms": 总耗时,
                "avg_item_time_ms": 平均耗时,
            }
        """
        with self.lock:
            job_id = self.current_job_id
            self.current_job_id += 1
        
        start_time = time.time()
        results = []
        errors = []
        success_count = 0
        failed_count = 0
        
        context = {
            "job_id": job_id,
            "total_items": len(items),
            "current_index": 0,
        }
        
        if mode == "parallel":
            results, errors, success_count, failed_count = self._process_parallel(
                items, processor_fn, context, debug
            )
        else:  # sequential
            results, errors, success_count, failed_count = self._process_sequential(
                items, processor_fn, context, debug
            )
        
        duration = (time.time() - start_time) * 1000
        avg_time = duration / len(items) if items else 0
        
        report = {
            "job_id": job_id,
            "total": len(items),
            "successful": success_count,
            "failed": failed_count,
            "success_rate": success_count / len(items) if items else 0,
            "results": results,
            "errors": errors,
            "duration_ms": duration,
            "avg_item_time_ms": avg_time,
        }
        
        if debug:
            print(f"\n📦 批处理完成 (作业 ID: {job_id})")
            print(f"  ├─ 总计: {len(items)} 项")
            print(f"  ├─ 成功: {success_count} 项")
            print(f"  ├─ 失败: {failed_count} 项")
            print(f"  ├─ 耗时: {duration:.0f}ms")
            print(f"  └─ 平均: {avg_time:.0f}ms/项")
        
        with self.lock:
            self.jobs[job_id] = report
        
        return report
    
    def _process_sequential(
        self,
        items: List[Any],
        processor_fn: Callable,
        context: Dict[str, Any],
        debug: bool = False
    ) -> Tuple[List, List, int, int]:
        """顺序处理。"""
        results = []
        errors = []
        success_count = 0
        failed_count = 0
        
        for index, item in enumerate(items):
            context["current_index"] = index + 1
            context["current_item"] = item
            
            try:
                result = processor_fn(item, context)
                results.append(result)
                success_count += 1
                
                if debug:
                    print(f"  ✅ [{index + 1}/{len(items)}] 处理完成")
            
            except Exception as e:
                error_info = {
                    "index": index,
                    "item": str(item)[:100],
                    "error": str(e),
                    "traceback": traceback.format_exc(),
                }
                errors.append(error_info)
                results.append(None)
                failed_count += 1
                
                if debug:
                    print(f"  ❌ [{index + 1}/{len(items)}] 失败: {e}")
        
        return results, errors, success_count, failed_count
    
    def _process_parallel(
        self,
        items: List[Any],
        processor_fn: Callable,
        context: Dict[str, Any],
        debug: bool = False
    ) -> Tuple[List, List, int, int]:
        """并行处理。"""
        results = [None] * len(items)
        errors = []
        success_count = 0
        failed_count = 0
        result_lock = threading.Lock()
        
        queue = Queue(maxsize=len(items))
        
        # 将项目加入队列
        for index, item in enumerate(items):
            queue.put((index, item))
        
        # 工作线程函数
        def worker():
            nonlocal success_count, failed_count
            while True:
                try:
                    index, item = queue.get(block=False)
                except:
                    break
                
                try:
                    ctx = context.copy()
                    ctx["current_index"] = index + 1
                    ctx["current_item"] = item
                    
                    result = processor_fn(item, ctx)
                    
                    with result_lock:
                        results[index] = result
                        success_count += 1
                    
                    if debug:
                        print(f"  ✅ [{index + 1}/{len(items)}] 处理完成")
                
                except Exception as e:
                    error_info = {
                        "index": index,
                        "item": str(item)[:100],
                        "error": str(e),
                        "traceback": traceback.format_exc(),
                    }
                    with result_lock:
                        errors.append(error_info)
                        results[index] = None
                        failed_count += 1
                    
                    if debug:
                        print(f"  ❌ [{index + 1}/{len(items)}] 失败: {e}")
        
        # 启动工作线程
        threads = []
        for _ in range(min(self.max_workers, len(items))):
            thread = threading.Thread(target=worker, daemon=True)
            thread.start()
            threads.append(thread)
        
        # 等待所有线程完成
        for thread in threads:
            thread.join(timeout=300)
        
        return results, errors, success_count, failed_count
    
    def get_job_report(self, job_id: int) -> Optional[Dict[str, Any]]:
        """获取指定作业的报告。"""
        with self.lock:
            return self.jobs.get(job_id)
    
    def list_jobs(self) -> List[Dict[str, Any]]:
        """列出所有已完成的作业。"""
        with self.lock:
            return list(self.jobs.values())


class ProgressTracker:
    """进度跟踪器。"""
    
    def __init__(self, total: int, description: str = "处理中"):
        self.total = total
        self.description = description
        self.current = 0
        self.start_time = time.time()
        self.lock = threading.Lock()
    
    def update(self, n: int = 1) -> None:
        """更新进度。"""
        with self.lock:
            self.current = min(self.current + n, self.total)
    
    def get_progress(self) -> Dict[str, Any]:
        """获取当前进度。"""
        with self.lock:
            elapsed = time.time() - self.start_time
            if self.current > 0:
                estimated_total = elapsed * self.total / self.current
                estimated_remaining = estimated_total - elapsed
            else:
                estimated_total = 0
                estimated_remaining = 0
            
            return {
                "current": self.current,
                "total": self.total,
                "percentage": (self.current / self.total * 100) if self.total > 0 else 0,
                "elapsed_ms": elapsed * 1000,
                "estimated_total_ms": estimated_total * 1000,
                "estimated_remaining_ms": estimated_remaining * 1000,
            }
    
    def print_progress(self, interval: float = 1.0) -> None:
        """周期性打印进度条。"""
        last_print = 0
        while self.current < self.total:
            now = time.time()
            if now - last_print >= interval:
                progress = self.get_progress()
                percent = progress["percentage"]
                elapsed = progress["elapsed_ms"] / 1000
                remaining = progress["estimated_remaining_ms"] / 1000
                
                bar_length = 30
                filled = int(bar_length * self.current / self.total)
                bar = "█" * filled + "░" * (bar_length - filled)
                
                print(
                    f"\r{self.description} [{bar}] "
                    f"{percent:.0f}% ({self.current}/{self.total}) "
                    f"经过 {elapsed:.0f}s, 剩余 {remaining:.0f}s",
                    end="", flush=True
                )
                last_print = now
            
            time.sleep(0.1)
        
        print()  # 新行


# 全局批处理器实例
_batch_processor = BatchProcessor(max_workers=2)


def get_batch_processor() -> BatchProcessor:
    """获取全局批处理器。"""
    global _batch_processor
    if _batch_processor is None:
        _batch_processor = BatchProcessor(max_workers=2)
    return _batch_processor
