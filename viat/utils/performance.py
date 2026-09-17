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


def fast_seek(cap, target_frame: int, current_frame: int, cache: FrameCache = None, backward_prefetch: int = 15):
    """Seek to target_frame efficiently, returning the decoded frame.

    Strategy:
      1. Check the cache first (instant if hit).
      2. If target is current_frame + 1, just cap.read() (fastest).
      3. If target is within ~30 frames forward, use cap.grab() to skip
         decoding intermediate frames, then cap.read() the target.
      4. If target < current_frame (backward seek) and not in cache,
         pre-fetch a small contiguous range of frames [target_frame - prefetch .. target_frame]
         starting from an earlier frame so that subsequent backward steps hit cache.
      5. Otherwise, fall back to cap.set(POS_FRAMES) + cap.read().

    Args:
        cap: cv2.VideoCapture (opened).
        target_frame: frame number to seek to.
        current_frame: the current frame position (for proximity check).
        cache: optional FrameCache.
        backward_prefetch: number of preceding frames to pre-fetch on backward seek cache miss.

    Returns:
        (frame, actual_frame) or (None, target_frame) on failure.
    """
    import time
    t_fs_start = time.perf_counter()

    def _log_slow(strategy, extra=""):
        elapsed = time.perf_counter() - t_fs_start
        if elapsed > 0.1:
            from ..logger import logger
            logger.warning(
                f"fast_seek({target_frame}): strategy={strategy} took {elapsed:.3f}s{extra}"
            )

    if cap is None or not cap.isOpened():
        return None, target_frame

    # 1. Cache check
    if cache:
        cached = cache.get(target_frame)
        if cached is not None:
            _log_slow("cache_hit", f" (cache size={cache.size})")
            return cached, target_frame

    # The actual position of the cv2.VideoCapture object
    cap_pos = int(cap.get(cv2.CAP_PROP_POS_FRAMES))

    # 2. Exact next frame (most common for playback): just read
    if target_frame == cap_pos:
        ret, frame = cap.read()
        if ret and frame is not None:
            if cache:
                cache.put(target_frame, frame)
            _log_slow("sequential_read", f" (cap_pos={cap_pos})")
            return frame, target_frame
        _log_slow("sequential_read_failed", f" (cap_pos={cap_pos})")
        return None, target_frame

    # 3. Forward by a small amount: grab + read
    delta = target_frame - cap_pos
    if 0 < delta <= 30 and cap_pos >= 0:
        # grab (skip decode) for intermediate frames
        for _ in range(delta):
            if not cap.grab():
                break
        ret, frame = cap.read()
        if ret and frame is not None:
            if cache:
                cache.put(target_frame, frame)
            _log_slow("forward_grab", f" (delta={delta}, cap_pos={cap_pos})")
            return frame, target_frame

    # 4. Backward seek: pre-fetch range if target_frame < cap_pos
    if target_frame < cap_pos and backward_prefetch > 0:
        start_frame = max(0, target_frame - backward_prefetch)
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        target_img = None
        for f in range(start_frame, target_frame + 1):
            ret, frame = cap.read()
            if not ret or frame is None:
                break
            if cache:
                cache.put(f, frame)
            if f == target_frame:
                target_img = frame
        if target_img is not None:
            _log_slow("backward_prefetch", f" (start_frame={start_frame}, cap_pos={cap_pos})")
            return target_img, target_frame

    # 5. Fallback: set POS_FRAMES + read
    cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
    ret, frame = cap.read()
    if ret and frame is not None:
        if cache:
            cache.put(target_frame, frame)
        _log_slow("fallback_set_read", f" (delta={target_frame - cap_pos}, cap_pos={cap_pos})")
        return frame, target_frame

    _log_slow("fallback_failed", f" (cap_pos={cap_pos})")
    return None, target_frame


# --------------------------------------------------------------------------- #
# Background seek worker
# --------------------------------------------------------------------------- #
#
# cv2.VideoCapture.set(POS_FRAMES) can require decoding forward from the
# last keyframe, which measured up to ~1.5s on some videos. Doing that
# directly on the UI thread freezes the whole app for that long (a native
# blocking call can't interleave with Qt's event loop). VideoSeekWorker runs
# the decode on a dedicated background thread instead; the caller
# (VideoAnnotationTool.seek_to_frame) waits for the result via a QEventLoop,
# which keeps pumping Qt's event loop while it waits, so the app stays
# responsive (repaints, doesn't get flagged "Not Responding") even though
# that one seek is still just as slow.
#
# cv2.VideoCapture is not safe for concurrent access from multiple threads,
# so `cap_lock` must be the SAME lock every other direct app.cap consumer
# in the codebase acquires before touching the capture (playback, undo/redo,
# batch scans, etc.) -- otherwise this worker can race with them.

from PyQt5.QtCore import QObject, pyqtSignal


class VideoSeekDispatcher(QObject):
    """Lives on the main thread; emitting seekRequested queues a job onto
    VideoSeekWorker (which lives on a different thread), via Qt's automatic
    cross-thread queued connection."""

    seekRequested = pyqtSignal(int, int, int)  # target_frame, current_frame, request_id


class VideoSeekWorker(QObject):
    """Runs on a dedicated background QThread. Do not call methods on this
    directly from the main thread -- trigger it via VideoSeekDispatcher's
    signal so PyQt queues the call onto the worker thread."""

    resultReady = pyqtSignal(int, bool, object, int)  # request_id, ok, frame, actual_frame

    def __init__(self, app, cap_lock):
        super().__init__()
        self.app = app
        self.cap_lock = cap_lock

    def do_seek(self, target_frame, current_frame, request_id):
        import time
        ok, frame, actual = False, None, target_frame
        t0 = time.perf_counter()
        with self.cap_lock:
            t_lock_acquired = time.perf_counter()
            cap = getattr(self.app, "cap", None)
            if cap is not None and cap.isOpened():
                perf_mgr = getattr(self.app, "performance_manager", None)
                if perf_mgr is not None:
                    frame = perf_mgr.seek_frame(target_frame)
                    ok = frame is not None
                    actual = target_frame
                elif target_frame == current_frame + 1:
                    ok, frame = cap.read()
                    actual = target_frame
                else:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
                    ok, frame = cap.read()
                    actual = target_frame
            t_decode_done = time.perf_counter()
        lock_wait = t_lock_acquired - t0
        decode_time = t_decode_done - t_lock_acquired
        if lock_wait > 0.05 or decode_time > 0.1:
            from ..logger import logger
            sys_info = _sample_system_state()
            logger.warning(
                f"VideoSeekWorker.do_seek({target_frame}): lock_wait={lock_wait:.3f}s, "
                f"decode/seek={decode_time:.3f}s -- {sys_info}"
            )
        self.resultReady.emit(request_id, ok, frame, actual)


def _sample_system_state():
    """Best-effort snapshot of what else the system/process was doing,
    captured only on an already-slow path (so its own cost doesn't
    contaminate the timing being reported). Never raises."""
    parts = []
    try:
        import psutil
        proc = psutil.Process()
        parts.append(f"proc_cpu%={proc.cpu_percent(interval=0.05):.0f}")
        parts.append(f"sys_cpu%={psutil.cpu_percent(interval=None):.0f}")
        parts.append(f"threads={proc.num_threads()}")
        mem = proc.memory_info()
        parts.append(f"proc_rss_mb={mem.rss / (1024 * 1024):.0f}")
    except Exception as e:
        parts.append(f"psutil_error={e}")
    try:
        import subprocess
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,utilization.gpu,memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=1
        )
        if out.returncode == 0:
            gpu_lines = "; ".join(l.strip() for l in out.stdout.strip().splitlines())
            parts.append(f"gpu=[{gpu_lines}]")
    except Exception:
        pass
    return ", ".join(parts)


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

    def __init__(self, app=None, cache_capacity: int = 200, backward_prefetch: int = 15):
        self.app = app
        self.cache = FrameCache(cache_capacity)
        self.backward_prefetch = backward_prefetch
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
        frame, actual = fast_seek(cap, target_frame, current, self.cache, self.backward_prefetch)
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
# main.py imports `PerfomanceManger` from viat.utils, so we alias it here.
PerfomanceManger = PerformanceManager
