import os
import cv2
import traceback
from viat.logger import logger

class BaseTracker:
    def __init__(self, **kwargs):
        pass

    def init(self, image, bbox):
        """
        image: numpy array (H, W, 3) BGR image
        bbox: tuple (x, y, w, h)
        Returns: success (bool)
        """
        raise NotImplementedError

    def update(self, image):
        """
        image: numpy array
        Returns: (success (bool), bbox (tuple (x,y,w,h)))
        """
        raise NotImplementedError

class OpenCVTracker(BaseTracker):
    def __init__(self, tracker_type="CSRT", **kwargs):
        super().__init__(**kwargs)
        self.tracker_type = tracker_type
        self.tracker = None
        self._create_tracker()

    def _create_tracker(self):
        if self.tracker_type == "CSRT":
            self.tracker = cv2.TrackerCSRT_create()
        elif self.tracker_type == "KCF":
            self.tracker = cv2.TrackerKCF_create()
        elif self.tracker_type == "MIL":
            self.tracker = cv2.TrackerMIL_create()

    def init(self, image, bbox):
        if self.tracker is None:
            return False
        # Ensure bbox is a tuple of floats/ints
        try:
            return self.tracker.init(image, tuple(bbox))
        except Exception as e:
            logger.error(f"OpenCVTracker init error: {e}")
            return False

    def update(self, image):
        if self.tracker is None:
            return False, (0,0,0,0)
        try:
            success, bbox = self.tracker.update(image)
            return success, bbox
        except Exception as e:
            logger.error(f"OpenCVTracker update error: {e}")
            return False, (0,0,0,0)

class TrackerManager:
    def __init__(self):
        self.available_trackers = {
            "CSRT (Accurate)": {"class": OpenCVTracker, "kwargs": {"tracker_type": "CSRT"}, "available": True, "message": ""},
            "KCF (Fast)": {"class": OpenCVTracker, "kwargs": {"tracker_type": "KCF"}, "available": True, "message": ""},
            "MIL": {"class": OpenCVTracker, "kwargs": {"tracker_type": "MIL"}, "available": True, "message": ""},
        }
        self._load_ettrack()
        self._load_ostrack()

    def _load_ostrack(self):
        try:
            import sys
            # Clear lib to avoid conflict with E.T.Track
            for k in list(sys.modules.keys()):
                if k == 'lib' or k.startswith('lib.'):
                    del sys.modules[k]
            from viat.tracking.ostrack_tracker import OSTrackWrapper
            is_available, msg = OSTrackWrapper.check_availability()
            self.available_trackers["OSTrack"] = {
                "class": OSTrackWrapper,
                "kwargs": {"use_fp16": False},
                "available": is_available,
                "message": msg
            }
            self.available_trackers["OSTrack TRT"] = {
                "class": OSTrackWrapper,
                "kwargs": {"use_fp16": True},
                "available": is_available,
                "message": msg
            }
            
            current_dir = os.path.dirname(os.path.abspath(__file__))
            engine_path = os.path.abspath(os.path.join(current_dir, '..', 'checkpoints', 'ostrack.engine'))
            engine_avail = os.path.exists(engine_path)
            self.available_trackers["OSTrack Native TRT"] = {
                "class": OSTrackWrapper,
                "kwargs": {"use_engine": True},
                "available": engine_avail,
                "message": "Engine file found" if engine_avail else f"Engine file missing at '{engine_path}'. Run setup_ostrack_trt.sh first."
            }
        except ImportError as e:
            logger.warning(f"OSTrack dependencies missing: {e}")
            for t_name in ["OSTrack", "OSTrack TRT", "OSTrack Native TRT"]:
                self.available_trackers[t_name] = {
                    "class": None,
                    "kwargs": {},
                    "available": False,
                    "message": f"Missing library: {e}"
                }
        except Exception as e:
            logger.error(f"Failed to load OSTrack module: {e}\n{traceback.format_exc()}")
            for t_name in ["OSTrack", "OSTrack TRT", "OSTrack Native TRT"]:
                self.available_trackers[t_name] = {
                    "class": None,
                    "kwargs": {},
                    "available": False,
                    "message": f"Error: {e}"
                }

    def _load_ettrack(self):
        try:
            import sys
            # Clear lib to avoid conflict with OSTrack
            for k in list(sys.modules.keys()):
                if k == 'lib' or k.startswith('lib.'):
                    del sys.modules[k]
            from viat.tracking.ettrack import ETTracker
            is_available, msg = ETTracker.check_availability()
            self.available_trackers["E.T.Track"] = {
                "class": ETTracker,
                "kwargs": {},
                "available": is_available,
                "message": msg
            }
        except ImportError as e:
            logger.warning(f"ETTrack dependencies missing: {e}")
            self.available_trackers["E.T.Track"] = {
                "class": None,
                "kwargs": {},
                "available": False,
                "message": f"Missing library: {e}"
            }
        except Exception as e:
            logger.error(f"Failed to load E.T.Track module: {e}\n{traceback.format_exc()}")
            self.available_trackers["E.T.Track"] = {
                "class": None,
                "kwargs": {},
                "available": False,
                "message": f"Error: {e}"
            }

    def get_tracker_info(self):
        """Returns a dict of tracker names and their availability info"""
        info = {}
        for k, v in self.available_trackers.items():
            info[k] = {
                "available": v["available"],
                "message": v.get("message", "")
            }
        return info

    def create_tracker(self, name):
        info = self.available_trackers.get(name)
        if not info:
            raise ValueError(f"Unknown tracker {name}")
        if not info["available"] or info["class"] is None:
            raise ValueError(f"Tracker {name} is not available: {info.get('message', 'Unknown reason')}")
        return info["class"](**info["kwargs"])
