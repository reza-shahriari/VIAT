import cv2


def open_video_writer(path, fps, size):
    """Open a VideoWriter preferring H.264 ('avc1'), falling back to 'mp4v'
    on builds/machines whose FFmpeg lacks libx264 (e.g. resolves to an
    unavailable hardware encoder like h264_v4l2m2m and fails to open)."""
    out = None
    for fourcc_name in ('avc1', 'mp4v'):
        out = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*fourcc_name), fps, size)
        if out.isOpened():
            return out
        out.release()
    return out
