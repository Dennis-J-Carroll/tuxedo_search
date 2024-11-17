"""Performance tracking utilities."""
import time
from typing import Dict, Any, Callable
from functools import wraps

def track_performance(func: Callable) -> Callable:
    """Decorator to track function performance."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        
        stats = getattr(wrapper, 'stats', {
            'calls': 0,
            'total_time': 0.0
        })
        stats['calls'] += 1
        stats['total_time'] += (end_time - start_time)
        wrapper.stats = stats
        
        return result
    return wrapper 