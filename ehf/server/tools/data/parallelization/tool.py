import concurrent.futures
import asyncio
import time
import psutil
from typing import List, Callable, Any, Optional, Union
from functools import wraps
import logging

logger = logging.getLogger(__name__)


class ParallelizationTool:
    """
    A utility tool for parallelizing function calls with various execution strategies.
    Supports both synchronous and asynchronous parallel execution.
    Automatically detects and optimizes for high-performance systems like M3 Ultra Mac Studio.
    """
    
    def __init__(self, max_workers: Optional[int] = None, timeout: Optional[float] = None):
        """
        Initialize the parallelization tool with automatic hardware detection.
        
        Args:
            max_workers: Override automatic worker calculation (None for auto-detection)
            timeout: Default timeout for operations in seconds
        """
        self.timeout = timeout
        
        # Hardware detection and optimization
        self._detect_hardware()
        self._calculate_optimal_settings()
        
        # Use provided max_workers or calculated optimal
        self.max_workers = max_workers or self.optimal_workers
        
    def _detect_hardware(self):
        """Detect system hardware and set optimization flags."""
        self.cpu_count = psutil.cpu_count(logical=True)
        self.memory_gb = psutil.virtual_memory().total / (1024**3)
        
        # Detect high-performance systems
        self.is_m3_ultra = self.cpu_count >= 28 and self.memory_gb >= 90
        self.is_high_end = self.cpu_count >= 16 and self.memory_gb >= 32
        
        if self.is_m3_ultra:
            print(f"🔥 M3 Ultra Mac Studio detected! ({self.cpu_count} cores, {self.memory_gb:.1f}GB RAM)")
        elif self.is_high_end:
            print(f"🚀 High-end system detected: {self.cpu_count} cores, {self.memory_gb:.1f}GB RAM")
        else:
            print(f"💻 Standard system: {self.cpu_count} cores, {self.memory_gb:.1f}GB RAM")
    
    def _calculate_optimal_settings(self):
        """Calculate optimal parallelization settings based on detected hardware."""
        if self.is_m3_ultra:
            # M3 Ultra Mac Studio - maximum performance
            self.optimal_workers = min(self.cpu_count * 6, 300)  # Up to 300 workers
            self.rate_limit_threshold = 100  # No rate limiting for tasks up to 100 items
            self.small_task_threshold = 20
            self.medium_task_threshold = 200
        elif self.is_high_end:
            # High-end systems (16+ cores, 32+ GB RAM)
            self.optimal_workers = min(self.cpu_count * 4, 200)  # Up to 200 workers
            self.rate_limit_threshold = 75
            self.small_task_threshold = 15
            self.medium_task_threshold = 150
        else:
            # Standard systems
            self.optimal_workers = min(self.cpu_count * 2, 100)  # Up to 100 workers
            self.rate_limit_threshold = 50
            self.small_task_threshold = 10
            self.medium_task_threshold = 100
        
        print(f"📊 Optimal settings: {self.optimal_workers} workers")
    
    def get_dynamic_workers_for_task(self, item_count: int) -> int:
        """Calculate optimal worker count based on task size and hardware."""
        if item_count <= self.small_task_threshold:
            return min(self.optimal_workers // 4, item_count)  # Small tasks
        elif item_count <= self.medium_task_threshold:
            return min(self.optimal_workers // 2, item_count)  # Medium tasks
        else:
            return min(self.optimal_workers, item_count)  # Large tasks
    
    def parallel_map(
        self, 
        func: Callable, 
        items: List[Any], 
        max_workers: Optional[int] = None,
        timeout: Optional[float] = None,
        return_exceptions: bool = False
    ) -> List[Any]:
        """
        Execute a function in parallel for each item in the list.
        
        Args:
            func: Function to execute for each item
            items: List of items to process
            max_workers: Override default max_workers
            timeout: Override default timeout
            return_exceptions: Whether to return exceptions instead of raising them
            
        Returns:
            List of results in the same order as input items
        """
        workers = max_workers or self.max_workers
        timeout_val = timeout or self.timeout
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
            future_to_item = {executor.submit(func, item): item for item in items}
            
            results = []
            for future in concurrent.futures.as_completed(future_to_item, timeout=timeout_val):
                try:
                    result = future.result()
                    results.append(result)
                except Exception as exc:
                    if return_exceptions:
                        results.append(exc)
                    else:
                        logger.error(f"Function call generated an exception: {exc}")
                        raise exc
            
            return results
    
    def parallel_map_with_args(
        self,
        func: Callable,
        items: List[Any],
        args: tuple = (),
        kwargs: dict = None,
        max_workers: Optional[int] = None,
        timeout: Optional[float] = None,
        return_exceptions: bool = False
    ) -> List[Any]:
        """
        Execute a function in parallel for each item with additional arguments.
        
        Args:
            func: Function to execute for each item
            items: List of items to process
            args: Additional positional arguments to pass to func
            kwargs: Additional keyword arguments to pass to func
            max_workers: Override default max_workers
            timeout: Override default timeout
            return_exceptions: Whether to return exceptions instead of raising them
            
        Returns:
            List of results in the same order as input items
        """
        kwargs = kwargs or {}
        workers = max_workers or self.max_workers
        timeout_val = timeout or self.timeout
        
        def wrapper(item):
            return func(item, *args, **kwargs)
        
        return self.parallel_map(wrapper, items, workers, timeout_val, return_exceptions)
    
    def parallel_batch(
        self,
        func: Callable,
        items: List[Any],
        batch_size: int = 10,
        max_workers: Optional[int] = None,
        timeout: Optional[float] = None,
        delay_between_batches: float = 0.0
    ) -> List[Any]:
        """
        Execute function in parallel batches to avoid overwhelming APIs or resources.
        
        Args:
            func: Function to execute for each item
            items: List of items to process
            batch_size: Number of items to process in each batch
            max_workers: Override default max_workers
            timeout: Override default timeout
            delay_between_batches: Delay in seconds between batches
            
        Returns:
            List of results in the same order as input items
        """
        workers = max_workers or self.max_workers
        timeout_val = timeout or self.timeout
        all_results = []
        
        for i in range(0, len(items), batch_size):
            batch = items[i:i + batch_size]
            logger.info(f"Processing batch {i//batch_size + 1}/{(len(items) + batch_size - 1)//batch_size}")
            
            batch_results = self.parallel_map(func, batch, workers, timeout_val)
            all_results.extend(batch_results)
            
            if delay_between_batches > 0 and i + batch_size < len(items):
                time.sleep(delay_between_batches)
        
        return all_results
    
    async def async_parallel_map(
        self,
        async_func: Callable,
        items: List[Any],
        max_workers: Optional[int] = None,
        timeout: Optional[float] = None
    ) -> List[Any]:
        """
        Execute an async function in parallel for each item.
        
        Args:
            async_func: Async function to execute for each item
            items: List of items to process
            max_workers: Override default max_workers
            timeout: Override default timeout
            
        Returns:
            List of results in the same order as input items
        """
        workers = max_workers or self.max_workers
        semaphore = asyncio.Semaphore(workers)
        
        async def bounded_func(item):
            async with semaphore:
                return await async_func(item)
        
        tasks = [bounded_func(item) for item in items]
        return await asyncio.gather(*tasks, return_exceptions=True)
    
    def rate_limited_parallel_map(
        self,
        func: Callable,
        items: List[Any],
        calls_per_second: float = 10.0,
        max_workers: Optional[int] = None,
        timeout: Optional[float] = None
    ) -> List[Any]:
        """
        Execute function in parallel with rate limiting to respect API limits.
        
        Args:
            func: Function to execute for each item
            items: List of items to process
            calls_per_second: Maximum calls per second
            max_workers: Override default max_workers
            timeout: Override default timeout
            
        Returns:
            List of results in the same order as input items
        """
        workers = max_workers or self.max_workers
        timeout_val = timeout or self.timeout
        delay_between_calls = 1.0 / calls_per_second
        
        def rate_limited_func(item):
            time.sleep(delay_between_calls)
            return func(item)
        
        return self.parallel_map(rate_limited_func, items, workers, timeout_val)
    
    def parallel_map_with_retry(
        self,
        func: Callable,
        items: List[Any],
        max_retries: int = 3,
        retry_delay: float = 1.0,
        max_workers: Optional[int] = None,
        timeout: Optional[float] = None
    ) -> List[Any]:
        """
        Execute function in parallel with automatic retry on failure.
        
        Args:
            func: Function to execute for each item
            items: List of items to process
            max_retries: Maximum number of retry attempts
            retry_delay: Delay between retries in seconds
            max_workers: Override default max_workers
            timeout: Override default timeout
            
        Returns:
            List of results in the same order as input items
        """
        workers = max_workers or self.max_workers
        timeout_val = timeout or self.timeout
        
        def retry_func(item):
            for attempt in range(max_retries + 1):
                try:
                    return func(item)
                except Exception as e:
                    if attempt == max_retries:
                        logger.error(f"Failed after {max_retries} retries for item {item}: {e}")
                        raise e
                    logger.warning(f"Attempt {attempt + 1} failed for item {item}: {e}. Retrying...")
                    time.sleep(retry_delay)
        
        return self.parallel_map(retry_func, items, workers, timeout_val)
    
    def smart_parallel_map(
        self,
        func: Callable,
        items: List[Any],
        force_workers: Optional[int] = None
    ) -> List[Any]:
        """
        Smart parallelization that automatically chooses optimal workers based on hardware and task size.
        
        Args:
            func: Function to execute for each item
            items: List of items to process
            force_workers: Override automatic worker calculation
            
        Returns:
            List of results in the same order as input items
        """
        item_count = len(items)
        start_time = time.time()
        
        # Dynamic configuration based on hardware and task size
        workers = force_workers or self.get_dynamic_workers_for_task(item_count)
        
        print(f"📈 Processing {item_count} items with {workers} workers")
        
        # Use parallel execution with retry logic
        results = self.parallel_map_with_retry(
            func, 
            items, 
            max_workers=workers,
            max_retries=2,
            retry_delay=0.5
        )
        
        # Performance metrics
        duration = time.time() - start_time
        items_per_second = item_count / duration if duration > 0 else 0
        
        print(f"✅ Completed in {duration:.2f}s ({items_per_second:.1f} items/sec)")
        
        return results
    
    def ultra_mode_parallel_map(self, func: Callable, items: List[Any]) -> List[Any]:
        """
        Ultra mode for maximum performance on high-end systems.
        Only available on M3 Ultra or high-end systems.
        """
        if not (self.is_m3_ultra or self.is_high_end):
            print("⚠️  Ultra mode only available on high-end systems")
            return self.smart_parallel_map(func, items)
        
        if self.is_m3_ultra:
            print("🔥 M3 ULTRA MODE ACTIVATED - UNLEASHING THE BEAST! 🔥")
        else:
            print("🚀 HIGH-END MODE ACTIVATED - MAXIMUM PERFORMANCE! 🚀")
        
        # Ultra settings - use maximum workers
        ultra_workers = min(self.optimal_workers, len(items))
        
        return self.smart_parallel_map(
            func, 
            items,
            force_workers=ultra_workers
        )


def parallelize(max_workers: int = 50, timeout: Optional[float] = None):
    """
    Decorator to automatically parallelize a function that takes a list of items.
    
    Args:
        max_workers: Maximum number of worker threads
        timeout: Timeout for the operation
        
    Usage:
        @parallelize(max_workers=10)
        def process_items(items):
            return [expensive_operation(item) for item in items]
    """
    def decorator(func):
        @wraps(func)
        def wrapper(items: List[Any], *args, **kwargs):
            tool = ParallelizationTool(max_workers=max_workers, timeout=timeout)
            return tool.parallel_map_with_args(func, items, args, kwargs)
        return wrapper
    return decorator


# Convenience functions for common use cases
def parallel_api_calls(
    api_func: Callable,
    items: List[Any],
    max_workers: int = 50,
    calls_per_second: Optional[float] = None,
    batch_size: Optional[int] = None
) -> List[Any]:
    """
    Convenience function for parallel API calls with common configurations.
    
    Args:
        api_func: API function to call
        items: List of items to process
        max_workers: Maximum number of workers
        calls_per_second: Rate limit (if None, no rate limiting)
        batch_size: Batch size (if None, no batching)
        
    Returns:
        List of API responses
    """
    tool = ParallelizationTool(max_workers=max_workers)
    
    if calls_per_second:
        return tool.rate_limited_parallel_map(api_func, items, calls_per_second)
    elif batch_size:
        return tool.parallel_batch(api_func, items, batch_size)
    else:
        return tool.parallel_map(api_func, items)
