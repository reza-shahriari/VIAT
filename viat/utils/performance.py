"""Performance optimization utilities for the Video Annotation Tool."""

import os
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


class PerfomanceManger:
    """Monitor and optimize application performance."""

    @staticmethod
    def get_memory_usage():
        import psutil
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
    
    
    def optimize_frame_hashes(self,frame_hashes, duplicate_frames_cache):
        """
        Optimize frame hashes by converting hash strings to numeric IDs.
        This reduces memory usage while maintaining duplicate frame detection functionality.
        
        Args:
            frame_hashes (dict): Dictionary mapping frame numbers to hash values
            duplicate_frames_cache (dict): Dictionary mapping hash values to lists of frame numbers
            
        Returns:
            tuple: (optimized_frame_hashes, optimized_duplicate_frames_cache, hash_to_id, id_to_hash)
        """
        if not frame_hashes:
            return {}, {}, {}, {}
        
        # Create a mapping of unique hashes to sequential IDs
        unique_hashes = set(frame_hashes.values())
        hash_to_id = {}
        id_to_hash = {}
        
        for i, hash_value in enumerate(unique_hashes, 1):
            hash_to_id[hash_value] = i
            id_to_hash[i] = hash_value
        
        # Convert frame hashes to use IDs instead of full hash strings
        optimized_frame_hashes = {}
        for frame_num, hash_value in frame_hashes.items():
            optimized_frame_hashes[frame_num] = hash_to_id[hash_value]
        
        # Convert duplicate frames cache to use hash IDs
        optimized_duplicate_frames_cache = {}
        for hash_value, frame_list in duplicate_frames_cache.items():
            if hash_value in hash_to_id:
                hash_id = hash_to_id[hash_value]
                optimized_duplicate_frames_cache[hash_id] = frame_list
        
        return optimized_frame_hashes, optimized_duplicate_frames_cache