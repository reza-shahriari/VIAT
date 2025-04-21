"""Performance optimization utilities for the Video Annotation Tool."""

import os
import psutil
import time
from functools import wraps


def measure_time(func):
    """Decorator to measure function execution time."""

    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(
            f"Function {func.__name__} took {end_time - start_time:.4f} seconds to execute"
        )
        return result

    return wrapper


class PerformanceMonitor:
    """Monitor and optimize application performance."""

    @staticmethod
    def get_memory_usage():
        """Get current memory usage of the application."""
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()
        return memory_info.rss / 1024 / 1024  # Convert to MB

    @staticmethod
    def optimize_image(image, max_size=1920):
        """Optimize image size for better performance."""
        import cv2
        import numpy as np

        # If image is too large, resize it
        h, w = image.shape[:2]
        if max(h, w) > max_size:
            scale = max_size / max(h, w)
            new_size = (int(w * scale), int(h * scale))
            image = cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)

        return image

    @staticmethod
    def lazy_load(func):
        """Decorator for lazy loading of resources."""

        @wraps(func)
        def wrapper(*args, **kwargs):
            if not hasattr(wrapper, "result"):
                wrapper.result = func(*args, **kwargs)
            return wrapper.result

        return wrapper
