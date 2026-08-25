import traceback
from viat.logger import logger
from viat.tracking.manager import TrackerManager
import cv2

class FastTrackerManager:
    """
    Adapter class to bridge viat.tracking.manager.TrackerManager with the API expected 
    by main.py for SAM tracking (track_video_from_prompt generator).
    """
    def __init__(self):
        self.tracker_manager = TrackerManager()
        
    def load_model(self, model_name):
        info = self.tracker_manager.get_tracker_info()
        if model_name in info:
            if info[model_name]["available"]:
                return True, "Loaded"
            else:
                return False, info[model_name].get("message", "Tracker unavailable")
        return False, f"Unknown tracker {model_name}"

    def _get_frames_stream(self, source_input, start_f=0):
        if isinstance(source_input, str):
            cap = cv2.VideoCapture(source_input)
            try:
                if start_f > 0:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, start_f)
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    yield cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            finally:
                cap.release()
        elif hasattr(source_input, '__iter__'):
            for frame in source_input:
                yield frame
        else:
            yield source_input

    def track_video_from_prompt(self, frame_generator, points=None, labels=None, box=None, text_prompt=None, model_type="E.T.Track", initial_polygon=None, start_f=0, end_f=None):
        if box is None:
            yield False, "Fast trackers require an initial bounding box. Please draw a box, click points, or generate a mask preview."
            return

        try:
            tracker = self.tracker_manager.create_tracker(model_type)
        except Exception as e:
            yield False, f"Failed to create tracker {model_type}: {str(e)}"
            return

        is_first_frame = True
        for i, frame_rgb in enumerate(self._get_frames_stream(frame_generator, start_f=start_f)):
            if end_f is not None and (start_f + i) > end_f:
                break
            # OpenCV trackers and typical trackers expect BGR format
            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

            if is_first_frame:
                x1, y1, x2, y2 = box
                w, h = x2 - x1, y2 - y1
                init_bbox = (x1, y1, w, h)
                
                try:
                    success = tracker.init(frame_bgr, init_bbox)
                except Exception as e:
                    yield False, f"Tracker init error: {str(e)}"
                    return
                
                if not success:
                    yield False, "Failed to initialize tracker on the first frame."
                    return
                
                yield True, {
                    'boxes': [[x1, y1, x2, y2]],
                    'polygons': [initial_polygon] if initial_polygon else []
                }
                is_first_frame = False
            else:
                try:
                    success, bbox = tracker.update(frame_bgr)
                except Exception as e:
                    yield False, f"Tracker update error: {str(e)}"
                    break
                    
                if success:
                    x, y, w, h = bbox
                    x2 = x + w
                    y2 = y + h
                    yield True, {
                        'boxes': [[int(x), int(y), int(x2), int(y2)]],
                        'polygons': []
                    }
                else:
                    # If tracking is lost, we still want to yield the failure to stop the process gracefully
                    yield False, "Tracking lost."
                    break

    def clear_session(self):
        """Clears any internal states if necessary. Matches SAM manager API."""
        pass
