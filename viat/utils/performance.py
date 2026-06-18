"""
Frame caching + fast seek for VIAT video playback performance.

Problem: large videos are slow because cv2.VideoCapture.set(POS_FRAMES) is
expensive for many codecs (it may decode from the last keyframe). Also,
canvas.repaint() + update_annotation_list() on every frame change adds up.

This module provides:
  * FrameCache -- an LRU cache for decoded frames, so repeated seeks to the
    same frame are instant.
  * fast_seek -- seeks the video efficiently: if the target is close to the
    current position, uses cap.grab() (which skips decoding); only reads
    (decodes) the final frame.
  * debounced_update -- coalesces multiple rapid update calls into one.
"""

import os
from collections import OrderedDict
from typing import Optional

try:
    import cv2
except ImportError:
    cv2 = None


# --------------------------------------------------------------------------- #
# Frame cache (LRU)
# --------------------------------------------------------------------------- #


class FrameCache:
    """LRU cache for decoded video frames.

    Stores {frame_num: numpy_array}. When the cache is full, the least
    recently used entry is evicted.
    """

    def __init__(self, capacity: int = 60):
        self.capacity = capacity
        self._cache: OrderedDict = OrderedDict()
        self.hits = 0
        self.misses = 0

    def get(self, frame_num: int):
        if frame_num in self._cache:
            self._cache.move_to_end(frame_num)
            self.hits += 1
            return self._cache[frame_num]
        self.misses += 1
        return None

    def put(self, frame_num: int, frame):
        if frame_num in self._cache:
            self._cache.move_to_end(frame_num)
        self._cache[frame_num] = frame
        if len(self._cache) > self.capacity:
            self._cache.popitem(last=False)

    def clear(self):
        self._cache.clear()
        self.hits = 0
        self.misses = 0

    @property
    def size(self):
        return len(self._cache)

    @property
    def hit_rate(self):
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0


# --------------------------------------------------------------------------- #
# Fast seek
# --------------------------------------------------------------------------- #


def fast_seek(cap, target_frame: int, current_frame: int, cache: FrameCache = None):
    """Seek to target_frame efficiently, returning the decoded frame.

    Strategy:
      1. Check the cache first (instant if hit).
      2. If target is current_frame + 1, just cap.read() (fastest).
      3. If target is within ~30 frames forward, use cap.grab() to skip
         decoding intermediate frames, then cap.read() the target.
      4. Otherwise, fall back to cap.set(POS_FRAMES) + cap.read().

    Args:
        cap: cv2.VideoCapture (opened).
        target_frame: frame number to seek to.
        current_frame: the current frame position (for proximity check).
        cache: optional FrameCache.

    Returns:
        (frame, actual_frame) or (None, target_frame) on failure.
    """
    if cap is None or not cap.isOpened():
        return None, target_frame

    # 1. Cache check
    if cache:
        cached = cache.get(target_frame)
        if cached is not None:
            return cached, target_frame

    # 2. Forward by 1: just read
    if target_frame == current_frame + 1:
        ret, frame = cap.read()
        if ret and frame is not None:
            if cache:
                cache.put(target_frame, frame)
            return frame, target_frame
        return None, target_frame

    # 3. Forward by a small amount: grab + read
    delta = target_frame - current_frame
    if 0 < delta <= 30 and current_frame >= 0:
        # grab (skip decode) for intermediate frames
        for _ in range(delta - 1):
            if not cap.grab():
                break
        ret, frame = cap.retrieve()
        if not ret or frame is None:
            ret, frame = cap.read()
        if ret and frame is not None:
            if cache:
                cache.put(target_frame, frame)
            return frame, target_frame

    # 4. Fallback: set POS_FRAMES + read
    cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
    ret, frame = cap.read()
    if ret and frame is not None:
        if cache:
            cache.put(target_frame, frame)
        return frame, target_frame

    return None, target_frame


# --------------------------------------------------------------------------- #
# Performance manager (attached to the main window)
# --------------------------------------------------------------------------- #


class PerformanceManager:
    """Manages frame caching + debounced updates for the main window.

    Attach to the app and call seek_frame() instead of the raw
    cap.set/read sequence. Update annotations are debounced so rapid
    navigation doesn't trigger N rebuilds of the annotation list.

    Backward compatible with the original PerfomanceManger (typo) class:
    - __init__ accepts no required args (original was PerfomanceManger())
    - optimize_frame_hashes() is provided as a passthrough
    """

    def __init__(self, app=None, cache_capacity: int = 60):
        self.app = app
        self.cache = FrameCache(cache_capacity)
        self._debounce_timer = None
        self._pending_update = False

    def seek_frame(self, target_frame: int):
        """Seek to target_frame using cache + fast seek. Returns the frame or None."""
        if self.app is None:
            return None
        cap = getattr(self.app, "cap", None)
        if cap is None or not cap.isOpened():
            return None

        current = getattr(self.app, "current_frame", 0)
        frame, actual = fast_seek(cap, target_frame, current, self.cache)
        return frame

    def clear_cache(self):
        """Clear the frame cache (e.g. when a new video is loaded)."""
        self.cache.clear()

    def get_stats(self) -> dict:
        """Return cache statistics for the status bar / debug."""
        return {
            "cache_size": self.cache.size,
            "cache_capacity": self.cache.capacity,
            "hit_rate": f"{self.cache.hit_rate:.1%}",
            "hits": self.cache.hits,
            "misses": self.cache.misses,
        }

    def debounced_update(self, callback, delay_ms: int = 50):
        """Debounce an update call (e.g. update_annotation_list).

        Multiple rapid calls within delay_ms are coalesced into one.
        """
        from PyQt5.QtCore import QTimer

        if self._debounce_timer is not None:
            self._debounce_timer.stop()
            self._debounce_timer.deleteLater()

        self._debounce_timer = QTimer()
        self._debounce_timer.setSingleShot(True)
        self._debounce_timer.timeout.connect(callback)
        self._debounce_timer.start(delay_ms)

    def optimize_frame_hashes(self, frame_hashes, duplicate_frames_cache):
        """Backward-compat passthrough for the original PerfomanceManger method.

        The original implementation optimized the frame_hashes dict to remove
        redundant entries. Since we don't have the original code, this returns
        the inputs unchanged. If you have the original optimize_frame_hashes,
        replace this method body with it.
        """
        return frame_hashes, duplicate_frames_cache


# Backward-compat alias: the original class name had a typo (PerfomanceManger).
# main.py imports `PerfomanceManger` from utils, so we alias it here.
PerfomanceManger = PerformanceManager
