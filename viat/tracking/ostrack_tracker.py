import os
import sys
import torch
import cv2
import numpy as np
import traceback

from viat.logger import logger
from viat.tracking.manager import BaseTracker

# Add ostrack_repo to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
ostrack_repo_path = os.path.join(current_dir, 'ostrack_repo')
if ostrack_repo_path not in sys.path:
    sys.path.insert(0, ostrack_repo_path)

_OSTRACK_AVAILABLE = False
try:
    from lib.config.ostrack.config import cfg, update_config_from_file
    from lib.test.utils import TrackerParams
    from lib.test.tracker.ostrack import OSTrack as OSTrackNative
    _OSTRACK_AVAILABLE = True
except ImportError as e:
    logger.error(f"Failed to import OSTrack dependencies: {e}")

class OSTrackWrapper(BaseTracker):
    """
    Standalone OSTrack inference wrapper using the submodule.
    """
    YAML_FILE = os.path.abspath(os.path.join(current_dir, 'ostrack_repo', 'experiments', 'ostrack', 'vitb_256_mae_ce_32x4_ep300.yaml'))

    @classmethod
    def get_checkpoint_path(cls):
        candidates = [
            os.path.abspath(os.path.join(current_dir, 'checkpoints', 'OSTrack_ep0300.pth.tar')),
            os.path.abspath(os.path.join(current_dir, '..', 'checkpoints', 'OSTrack_ep0300.pth.tar')),
            os.path.abspath(os.path.join("checkpoints", "OSTrack_ep0300.pth.tar"))
        ]
        for c in candidates:
            if os.path.exists(c):
                return c
        return candidates[0]

    def __init__(self, use_fp16=False, **kwargs):
        super().__init__(**kwargs)
        self.tracker = None
        self.use_fp16 = use_fp16
        self._load_model()

    @classmethod
    def check_availability(cls):
        if not _OSTRACK_AVAILABLE:
            return False, "OSTrack modules not found or dependencies missing (check timm, jpeg4py)."
        try:
            import torch
        except ImportError:
            return False, "PyTorch is not installed."
        
        ckpt = cls.get_checkpoint_path()
        if not os.path.exists(ckpt):
            return False, f"Model weights not found at '{ckpt}'"
        return True, "Available"

    def _load_model(self):
        is_available, msg = self.check_availability()
        if not is_available:
            raise RuntimeError(f"Cannot initialize OSTrack: {msg}")

        ckpt_path = self.get_checkpoint_path()

        # Update config
        update_config_from_file(self.YAML_FILE)
        
        params = TrackerParams()
        params.cfg = cfg
        params.checkpoint = ckpt_path
        params.template_factor = cfg.TEST.TEMPLATE_FACTOR
        params.template_size = cfg.TEST.TEMPLATE_SIZE
        params.search_factor = cfg.TEST.SEARCH_FACTOR
        params.search_size = cfg.TEST.SEARCH_SIZE
        params.save_all_boxes = False
        params.debug = 0

        logger.info(f"Loading OSTrack (use_fp16={self.use_fp16}) from {ckpt_path}")
        self.tracker = OSTrackNative(params, 'dummy')
        logger.info("OSTrack loaded successfully")

    def init(self, image, bbox):
        """
        image: BGR numpy array
        bbox: (x, y, w, h)
        """
        if self.tracker is None: return False
        
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # OSTrack's initialize expects info dict
        info = {'init_bbox': list(bbox)}
        import torch
        if self.use_fp16 and torch.cuda.is_available():
            with torch.amp.autocast('cuda', dtype=torch.float16):
                self.tracker.initialize(image_rgb, info)
        else:
            self.tracker.initialize(image_rgb, info)
        return True

    def update(self, image):
        if self.tracker is None: return False, (0,0,0,0)
        
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        info = {}
        
        import torch
        if self.use_fp16 and torch.cuda.is_available():
            with torch.amp.autocast('cuda', dtype=torch.float16):
                out = self.tracker.track(image_rgb, info)
        else:
            out = self.tracker.track(image_rgb, info)
            
        target_bbox = out['target_bbox'] # [x, y, w, h]
        
        return True, tuple(target_bbox)
