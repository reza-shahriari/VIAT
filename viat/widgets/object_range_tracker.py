"""Background worker for "Edit Object over Frame Range".

Finds the reference object in each frame of a range -- by track_id when the
object has one, otherwise by a visual tracker (MIL) + IoU against the
annotations on that frame -- and reports each match back to the UI thread,
which applies the actual edit. The worker never touches frame_annotations or
any widget directly; it only works from a snapshot taken on the UI thread.

Tracking stops at the first frame in the range where the object can't be
found, so a gap (object left the scene, occlusion, missing labels) doesn't
cause the edit to latch onto a different object further on.

The worker opens its own cv2.VideoCapture on the video file instead of using
the main window's shared `cap`, so it never needs `_cap_lock` and the user
can keep navigating while it runs.
"""

import cv2
from PyQt5.QtCore import QThread, pyqtSignal, QRect


def _rect_iou(a, b):
    inter = a.intersected(b)
    if inter.isEmpty():
        return 0.0
    inter_area = inter.width() * inter.height()
    union = a.width() * a.height() + b.width() * b.height() - inter_area
    return inter_area / union if union > 0 else 0.0


class ObjectRangeTrackWorker(QThread):
    """Emits frameMatched(frame_num, [annotation, ...]) for each frame where
    the object is found, progressChanged(frame_num) as it advances, and
    finishedTracking(last_frame_processed, stopped_frame) when done
    (stopped_frame is -1 if the whole range was covered)."""

    frameMatched = pyqtSignal(int, list)
    progressChanged = pyqtSignal(int)
    finishedTracking = pyqtSignal(int, int)

    IOU_THRESHOLD = 0.2

    def __init__(self, video_filename, start_frame, end_frame, reference_rect,
                 class_name, ref_track_id, snapshot, parent=None):
        """snapshot: {frame_num: [(annotation, QRect copy, class_name, track_id), ...]}
        taken on the UI thread for every frame in the range."""
        super().__init__(parent)
        self.video_filename = video_filename
        self.start_frame = start_frame
        self.end_frame = end_frame
        self.reference_rect = QRect(reference_rect)
        self.class_name = class_name
        self.ref_track_id = ref_track_id
        self.snapshot = snapshot

    def run(self):
        last_frame = self.start_frame - 1
        stopped_frame = -1
        cap = None
        tracker = None
        last_matched_rect = QRect(self.reference_rect)

        try:
            if self.ref_track_id is None and self.video_filename:
                cap = cv2.VideoCapture(self.video_filename, cv2.CAP_ANY)
                if cap.isOpened():
                    cap.set(cv2.CAP_PROP_POS_FRAMES, self.start_frame)
                    ret, frame = cap.read()
                    if ret:
                        try:
                            tracker = cv2.TrackerMIL_create()
                            r = last_matched_rect
                            tracker.init(frame, (r.x(), r.y(), r.width(), r.height()))
                        except Exception:
                            tracker = None

            for frame_num in range(self.start_frame, self.end_frame + 1):
                if self.isInterruptionRequested():
                    break

                predicted_rect = last_matched_rect
                # Frames are consumed strictly sequentially (we stop at the
                # first gap), so a plain read() keeps the tracker in step
                # without a per-frame seek.
                if tracker is not None and frame_num > self.start_frame:
                    ret, frame = cap.read()
                    if ret:
                        success, box = tracker.update(frame)
                        if success:
                            predicted_rect = QRect(int(box[0]), int(box[1]), int(box[2]), int(box[3]))

                entries = self.snapshot.get(frame_num, [])
                matched = []
                if self.ref_track_id is not None:
                    matched = [ann for ann, _, _, tid in entries if tid == self.ref_track_id]
                else:
                    best_ann, best_rect, best_iou = None, None, 0.0
                    for ann, rect, cls, _ in entries:
                        if cls != self.class_name:
                            continue
                        iou = _rect_iou(predicted_rect, rect)
                        if iou > best_iou:
                            best_ann, best_rect, best_iou = ann, rect, iou
                    if best_ann is not None and best_iou >= self.IOU_THRESHOLD:
                        matched = [best_ann]
                        last_matched_rect = best_rect

                if not matched:
                    stopped_frame = frame_num
                    break

                last_frame = frame_num
                self.frameMatched.emit(frame_num, matched)
                self.progressChanged.emit(frame_num)
        finally:
            if cap is not None:
                cap.release()
            self.finishedTracking.emit(last_frame, stopped_frame)
