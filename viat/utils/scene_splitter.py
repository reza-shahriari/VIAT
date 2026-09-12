"""
Scene-cut detection and splitting for VIAT.

Detects hard cuts in a video with PySceneDetect and splits it into one clip per
detected scene. Compared to the previous implementation, this module:

  - Reports real per-frame progress for both the detection and splitting
    phases via callbacks, instead of an indeterminate progress bar / a tqdm
    bar that only ever printed to the console.
  - Names clips with their final ``video_<n>.mp4`` name from the moment they
    are written, instead of writing them under PySceneDetect's own
    ``$VIDEO_NAME-Scene-$SCENE_NUMBER`` scheme and renaming them afterwards.
    That rename-after-the-fact is what made a name "change" mid-job; naming
    correctly up front is also what makes it safe to open the output folder
    as a video dataset (or add a clip to one already open) while splitting is
    still running.
  - Cuts each scene using an explicit output frame count (``-frames:v``)
    anchored to PySceneDetect's own frame numbers, rather than a
    ``-t <seconds>`` duration. A duration in seconds has to be rounded to the
    stream's time base, which could make a cut land a frame short/long -- the
    visible symptom being that the first frame of a clip actually still
    belonged to the previous scene.
  - Emits a callback right after each clip finishes writing, so a caller can
    react in real time (e.g. let the user start annotating the first couple
    of clips while the rest are still being cut) instead of blocking until
    the whole video has been processed.
  - Prefers a hardware H.264 encoder (NVENC/VAAPI/QSV) when ffmpeg reports one
    is available, falling back to libx264, since re-encoding every scene with
    libx264 on the CPU is what was pegging the CPU during a split.
"""
import os
import shutil
import subprocess

_HW_ENCODER_CACHE = {}


def _ffmpeg_path():
    return shutil.which("ffmpeg")


def _pick_video_encoder(ffmpeg_bin):
    """Pick the least CPU-hungry H.264 encoder ffmpeg has available.

    Hardware encoders (NVENC/VAAPI/QSV) offload encoding to the GPU, which is
    both much faster and avoids the near-100% CPU usage that re-encoding many
    scenes back-to-back with libx264 causes. Falls back to libx264 if none of
    them are usable. The result is cached per ffmpeg binary for the process.
    """
    if ffmpeg_bin in _HW_ENCODER_CACHE:
        return _HW_ENCODER_CACHE[ffmpeg_bin]

    args = ["-c:v", "libx264", "-preset", "veryfast", "-crf", "22"]
    try:
        out = subprocess.run(
            [ffmpeg_bin, "-hide_banner", "-encoders"],
            capture_output=True, text=True, timeout=10, check=False,
        ).stdout
        if "h264_nvenc" in out:
            args = ["-c:v", "h264_nvenc", "-preset", "p4", "-cq", "22"]
        elif "h264_qsv" in out:
            args = ["-c:v", "h264_qsv", "-preset", "veryfast", "-global_quality", "22"]
        elif "h264_vaapi" in out and os.path.exists("/dev/dri/renderD128"):
            args = [
                "-vaapi_device", "/dev/dri/renderD128",
                "-vf", "format=nv12,hwupload",
                "-c:v", "h264_vaapi", "-qp", "22",
            ]
    except Exception:
        pass  # Fall back to the libx264 default set above.

    _HW_ENCODER_CACHE[ffmpeg_bin] = args
    return args


def detect_scenes(video_path, threshold=3.0, min_scene_len=15, progress_cb=None, is_cancelled=None):
    """Run PySceneDetect's AdaptiveDetector over `video_path`.

    `progress_cb(current_frame, total_frames)`, if given, is called as frames
    are scanned so a caller can show a real percentage. `is_cancelled`, if
    given, is polled on every frame and stops the scan early when it returns
    True.

    Returns `(scene_list, frame_rate)`, where `scene_list` is the list of
    `(start, end)` `FrameTimecode` pairs PySceneDetect detected (empty if
    cancelled before finishing).
    """
    from scenedetect import open_video, SceneManager, AdaptiveDetector
    from scenedetect import scene_manager as _scene_manager_module

    video = open_video(video_path)
    scene_manager = SceneManager()
    scene_manager.add_detector(
        AdaptiveDetector(adaptive_threshold=threshold, min_scene_len=min_scene_len)
    )

    class _ProgressShim:
        """Minimal tqdm-compatible stand-in.

        PySceneDetect only reports scan progress through an internal `tqdm`
        reference (a no-op shim if `tqdm` isn't installed at all). Swapping
        that reference for this class for the duration of the scan lets us
        forward real frame counts to `progress_cb` instead of only ever being
        able to print a bar to the console. On cancellation it calls
        `scene_manager.stop()` -- the same thread-safe API `SceneManager`
        exposes for this purpose -- rather than tearing the scan down with an
        exception, so the background frame-decoding thread it started gets to
        shut down cleanly instead of being abandoned.
        """

        def __init__(self, total=0, **kwargs):
            self.total = total or 0
            self.n = 0

        def update(self, n=1):
            self.n += n
            if progress_cb:
                progress_cb(self.n, self.total)
            if is_cancelled and is_cancelled():
                scene_manager.stop()

        def set_description(self, *a, **kw):
            pass

        def close(self):
            pass

    original_tqdm = _scene_manager_module.tqdm
    _scene_manager_module.tqdm = _ProgressShim
    try:
        scene_manager.detect_scenes(video=video, show_progress=True)
    finally:
        _scene_manager_module.tqdm = original_tqdm

    if is_cancelled and is_cancelled():
        return [], video.frame_rate

    return scene_manager.get_scene_list(), video.frame_rate


def _clip_path(session_dir, index):
    return os.path.join(session_dir, f"video_{index}.mp4")


def split_scene_ffmpeg(video_path, start_frame, end_frame, frame_rate, output_path):
    """Cut a single `[start_frame, end_frame)` scene out of `video_path`.

    Uses an explicit output frame count (`-frames:v`) rather than a
    `-t <seconds>` duration for the clip length: a duration in seconds has to
    be rounded to the stream's time base, which can make a cut land a frame
    short/long. Anchoring both endpoints directly to PySceneDetect's own frame
    numbers keeps every clip frame-exact and gap-free with its neighbours, so
    the frame right after a cut always belongs to the new clip, never the
    previous one.
    """
    ffmpeg_bin = _ffmpeg_path()
    if not ffmpeg_bin:
        raise RuntimeError("ffmpeg was not found on PATH; it is required to split videos.")

    fps = float(frame_rate)
    start_seconds = start_frame / fps
    num_frames = end_frame - start_frame
    if num_frames <= 0:
        raise ValueError(f"Empty scene range [{start_frame}, {end_frame})")

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    call = [
        ffmpeg_bin, "-nostdin", "-y",
        "-ss", f"{start_seconds:.6f}",
        "-i", video_path,
        "-frames:v", str(num_frames),
        "-map", "0:v:0", "-map", "0:a?",
    ]
    call += _pick_video_encoder(ffmpeg_bin)
    call += ["-c:a", "aac", "-avoid_negative_ts", "make_zero", "-v", "error", output_path]

    result = subprocess.run(call, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise RuntimeError(
            f"ffmpeg failed splitting frames {start_frame}-{end_frame}:\n{result.stderr}"
        )


def split_video_by_scenes(
    video_path, session_dir, threshold=3.0, min_scene_len=15,
    detect_progress_cb=None, split_progress_cb=None, clip_ready_cb=None,
    is_cancelled=None,
):
    """Detect scene cuts in `video_path` and split it into one clip per scene.

    Clips are written directly under their final `video_<n>.mp4` name inside
    `session_dir` -- there is no rename pass afterwards -- so a caller can
    safely open `session_dir` as a video dataset (or add a freshly written
    clip to one already open) as soon as `clip_ready_cb` fires, without a
    filename changing underneath it mid-job.

    `detect_progress_cb(current_frame, total_frames)` and
    `split_progress_cb(clips_done, total_clips)` report progress for each
    phase; `clip_ready_cb(path, index)` fires right after each clip finishes
    writing -- this is the hook a caller uses to let the user start working
    with clips before the whole job is done. `is_cancelled()`, if given, is
    polled between frames/scenes; when it returns True the job stops early and
    the original video is left untouched (any clips already written stay on
    disk).

    The original video is only removed once every clip has been written
    successfully.
    """
    os.makedirs(session_dir, exist_ok=True)

    scene_list, frame_rate = detect_scenes(
        video_path, threshold=threshold, min_scene_len=min_scene_len,
        progress_cb=detect_progress_cb, is_cancelled=is_cancelled,
    )

    if is_cancelled and is_cancelled():
        return session_dir, 0

    num_clips = len(scene_list)

    if num_clips <= 1:
        # No cuts detected -- just move the whole video into place under the
        # same naming scheme used for multi-clip output, so downstream code
        # never has to special-case a single-clip "dataset".
        dest = _clip_path(session_dir, 0)
        if os.path.exists(dest):
            os.remove(dest)
        shutil.move(video_path, dest)
        if clip_ready_cb:
            clip_ready_cb(dest, 0)
        if split_progress_cb:
            split_progress_cb(1, 1)
        return session_dir, 1

    clips_done = 0
    for i, (start_tc, end_tc) in enumerate(scene_list):
        if is_cancelled and is_cancelled():
            break
        out_path = _clip_path(session_dir, i)
        split_scene_ffmpeg(video_path, start_tc.frame_num, end_tc.frame_num, frame_rate, out_path)
        clips_done = i + 1
        if clip_ready_cb:
            clip_ready_cb(out_path, i)
        if split_progress_cb:
            split_progress_cb(clips_done, num_clips)

    if is_cancelled and is_cancelled():
        return session_dir, clips_done

    # Only remove the source once every clip has been written successfully.
    if os.path.exists(video_path):
        os.remove(video_path)

    return session_dir, num_clips


try:
    from PyQt5.QtCore import QThread, pyqtSignal

    class SceneSplitWorker(QThread):
        """Runs `split_video_by_scenes` in the background and forwards its
        callbacks as Qt signals, so the GUI thread never blocks on it and can
        react to clips as they complete."""

        detect_progress = pyqtSignal(int, int)   # current_frame, total_frames
        split_progress = pyqtSignal(int, int)    # clips_done, total_clips
        clip_ready = pyqtSignal(str, int)        # path, index
        finished_all = pyqtSignal(str, int)      # session_dir, num_clips
        error = pyqtSignal(str)

        def __init__(self, video_path, session_dir, threshold=3.0, min_scene_len=15, parent=None):
            super().__init__(parent)
            self.video_path = video_path
            self.session_dir = session_dir
            self.threshold = threshold
            self.min_scene_len = min_scene_len
            self._cancelled = False

        def cancel(self):
            self._cancelled = True

        def _is_cancelled(self):
            return self._cancelled

        def run(self):
            try:
                res_dir, num = split_video_by_scenes(
                    self.video_path, self.session_dir,
                    threshold=self.threshold, min_scene_len=self.min_scene_len,
                    detect_progress_cb=lambda c, t: self.detect_progress.emit(c, t),
                    split_progress_cb=lambda c, t: self.split_progress.emit(c, t),
                    clip_ready_cb=lambda p, i: self.clip_ready.emit(p, i),
                    is_cancelled=self._is_cancelled,
                )
                self.finished_all.emit(res_dir, num)
            except Exception as e:
                import traceback
                self.error.emit(f"{e}\n{traceback.format_exc()}")

except ImportError:
    SceneSplitWorker = None
