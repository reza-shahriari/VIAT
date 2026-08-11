import os
import sys
import torch
import numpy as np
import cv2
from viat.logger import logger
from viat.tracking.manager import BaseTracker

# Add ettrack_repo (or ettrack_lib) to path so it can import lib and tracking
current_dir = os.path.dirname(os.path.abspath(__file__))
ettrack_repo_path = os.path.join(current_dir, 'ettrack_repo')
ettrack_lib_path = os.path.join(current_dir, 'ettrack_lib')
if os.path.exists(ettrack_repo_path) and ettrack_repo_path not in sys.path:
    sys.path.insert(0, ettrack_repo_path)
elif ettrack_lib_path not in sys.path:
    sys.path.insert(0, ettrack_lib_path)

_ETTRACK_AVAILABLE = False
try:
    from tracking.basic_model.et_tracker import ET_Tracker
    from lib.utils.utils import get_subwindow_tracking, python2round
    from lib.utils.utils import cxy_wh_2_rect, get_axis_aligned_bbox
    _ETTRACK_AVAILABLE = True
except ImportError as e:
    logger.error(f"Failed to import E.T.Track dependencies from ettrack_repo/ettrack_lib: {e}")


class Config(object):
    """Matches the original E.T.Track config exactly."""
    def __init__(self, stride=8, even=1):
        self.penalty_k = 0.007
        self.window_influence = 0.225
        self.lr = 0.616
        self.windowing = 'cosine'
        if even:
            self.exemplar_size = 128
            self.instance_size = 288
        else:
            self.exemplar_size = 127
            self.instance_size = 255
        self.total_stride = stride
        self.score_size = int(round(self.instance_size / self.total_stride))
        self.context_amount = 0.5
        self.ratio = 1

    def renew(self):
        self.score_size = int(round(self.instance_size / self.total_stride))


class ETTracker(BaseTracker):
    """
    Standalone E.T.Track inference wrapper.
    Matches original pytracking inference loop exactly.
    """
    CHECKPOINT_PATH = os.path.join("checkpoints", "ettrack.pth")

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.tracker = None
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # Keep mean/std on CPU (original repo doesn't move them to device in normalize)
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

        # Hyperparameters that match the original repo for the provided checkpoint (stride=16)
        self.big_sz = 288
        self.small_sz = 288
        self.stride = 16
        self.even = 1

        self._load_model()

    @classmethod
    def check_availability(cls):
        if not _ETTRACK_AVAILABLE:
            return False, "ET_Tracker module not found in ettrack_lib."
        try:
            import torch
        except ImportError:
            return False, "PyTorch is not installed."
        if not os.path.exists(cls.CHECKPOINT_PATH):
            return False, f"Model weights not found at '{cls.CHECKPOINT_PATH}'"
        return True, "Available"

    def _load_model(self):
        is_available, msg = self.check_availability()
        if not is_available:
            raise RuntimeError(f"Cannot initialize E.T.Track: {msg}")

        self.tracker = ET_Tracker(search_size=288,
                                  template_size=128,
                                  stride=16,
                                  e_exemplars=4,
                                  sm_normalization=True,
                                  temperature=2,
                                  dropout=False)

        # Load weights
        checkpoint = torch.load(self.CHECKPOINT_PATH, map_location='cpu', weights_only=False)
        if 'net' in checkpoint:
            state_dict = checkpoint['net']
        else:
            state_dict = checkpoint

        self.tracker.load_state_dict(state_dict, strict=False)
        self.tracker.eval()
        self.tracker.to(self.device)
        logger.info(f"Loaded E.T.Track from {self.CHECKPOINT_PATH}")

    def normalize(self, x):
        """Normalize a (C,H,W) float tensor. Matches original repo (CPU ops then move)."""
        x = x.float()
        x /= 255.0
        x -= self.mean
        x /= self.std
        return x.to(self.device)

    def grids(self, p):
        sz = p.score_size
        sz_x = sz // 2
        sz_y = sz // 2
        x, y = np.meshgrid(np.arange(0, sz) - np.floor(float(sz_x)),
                            np.arange(0, sz) - np.floor(float(sz_y)))
        self.grid_to_search_x = x * p.total_stride + p.instance_size // 2
        self.grid_to_search_y = y * p.total_stride + p.instance_size // 2

    def init(self, image, bbox):
        """
        Initialize tracker.
        image: BGR numpy array (H, W, 3)
        bbox: (x, y, w, h) in pixel coordinates
        """
        if self.tracker is None:
            return False

        # Original pytracking converts BGR→RGB before running tracker
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        self.im_h, self.im_w = image.shape[0], image.shape[1]

        # Convert (x, y, w, h) → center (cx, cy, w, h)
        x, y, w, h = bbox
        cx = x + w / 2.0
        cy = y + h / 2.0
        self.target_pos = np.array([cx, cy])
        self.target_sz = np.array([w, h])

        p = Config(stride=self.stride, even=self.even)
        # Use big search window for tiny objects
        if ((self.target_sz[0] * self.target_sz[1]) / float(self.im_h * self.im_w)) < 0.004:
            p.instance_size = self.big_sz
        else:
            p.instance_size = self.small_sz
        p.renew()
        self.p = p

        self.grids(p)

        # Compute template crop size (context-padded square)
        wc_z = self.target_sz[0] + p.context_amount * sum(self.target_sz)
        hc_z = self.target_sz[1] + p.context_amount * sum(self.target_sz)
        s_z = round(np.sqrt(wc_z * hc_z))

        self.avg_chans = np.mean(image, axis=(0, 1))
        z_crop, _ = get_subwindow_tracking(image, self.target_pos, p.exemplar_size, s_z, self.avg_chans)

        z_crop = self.normalize(z_crop)
        z = z_crop.unsqueeze(0)

        with torch.no_grad():
            self.tracker.template(z)

        if p.windowing == 'cosine':
            self.window = np.outer(np.hanning(p.score_size), np.hanning(p.score_size))
        else:
            self.window = np.ones((int(p.score_size), int(p.score_size)))

        return True

    def change(self, r):
        return np.maximum(r, 1. / r)

    def sz(self, w, h):
        pad = (w + h) * 0.5
        sz2 = (w + pad) * (h + pad)
        return np.sqrt(sz2)

    def sz_wh(self, wh):
        pad = (wh[0] + wh[1]) * 0.5
        sz2 = (wh[0] + pad) * (wh[1] + pad)
        return np.sqrt(sz2)

    def update(self, image):
        """
        Run one tracking step.
        image: BGR numpy array (H, W, 3)
        Returns: (success, (x, y, w, h))
        """
        if self.tracker is None:
            return False, (0, 0, 0, 0)

        # Original pytracking converts BGR→RGB before running tracker
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        p = self.p

        # Compute search region size in image pixels
        hc_z = self.target_sz[1] + p.context_amount * sum(self.target_sz)
        wc_z = self.target_sz[0] + p.context_amount * sum(self.target_sz)
        s_z = np.sqrt(wc_z * hc_z)

        scale_z = p.exemplar_size / s_z
        d_search = (p.instance_size - p.exemplar_size) / 2
        pad = d_search / scale_z
        s_x = s_z + 2 * pad

        x_crop, _ = get_subwindow_tracking(image, self.target_pos, p.instance_size,
                                            python2round(s_x), self.avg_chans)
        x_crop = self.normalize(x_crop)
        x_crop = x_crop.unsqueeze(0)

        with torch.no_grad():
            cls_score, bbox_pred = self.tracker.track(x_crop)

        cls_score = torch.sigmoid(cls_score).squeeze().cpu().data.numpy()
        bbox_pred = bbox_pred.squeeze().cpu().data.numpy()

        # Decode predicted boxes in search-crop coordinate space
        pred_x1 = self.grid_to_search_x - bbox_pred[0, ...]
        pred_y1 = self.grid_to_search_y - bbox_pred[1, ...]
        pred_x2 = self.grid_to_search_x + bbox_pred[2, ...]
        pred_y2 = self.grid_to_search_y + bbox_pred[3, ...]

        # Size/ratio penalty — target_sz must be in CROP coordinates (× scale_z)
        # This matches the original: self.update(x_crop, target_pos, target_sz * scale_z, ...)
        target_sz_crop = self.target_sz * scale_z
        pred_w = np.maximum(1e-6, pred_x2 - pred_x1)
        pred_h = np.maximum(1e-6, pred_y2 - pred_y1)
        s_c = self.change(self.sz(pred_w, pred_h) / (self.sz_wh(target_sz_crop) + 1e-6))
        r_c = self.change((target_sz_crop[0] / target_sz_crop[1]) / (pred_w / pred_h))

        penalty = np.exp(-(r_c * s_c - 1) * p.penalty_k)
        pscore = penalty * cls_score
        pscore = pscore * (1 - p.window_influence) + self.window * p.window_influence

        r_max, c_max = np.unravel_index(pscore.argmax(), pscore.shape)

        # Best prediction in crop-space
        pred_x1 = pred_x1[r_max, c_max]
        pred_y1 = pred_y1[r_max, c_max]
        pred_x2 = pred_x2[r_max, c_max]
        pred_y2 = pred_y2[r_max, c_max]

        pred_xs = (pred_x1 + pred_x2) / 2
        pred_ys = (pred_y1 + pred_y2) / 2
        pred_w = pred_x2 - pred_x1
        pred_h = pred_y2 - pred_y1

        # Displace from crop center (instance_size/2) and scale back to image coords
        diff_xs = (pred_xs - p.instance_size // 2) / scale_z
        diff_ys = (pred_ys - p.instance_size // 2) / scale_z
        pred_w = pred_w / scale_z
        pred_h = pred_h / scale_z

        # Scale target_sz back to image coords for the smoothing step
        target_sz_img = target_sz_crop / scale_z  # == self.target_sz

        lr = penalty[r_max, c_max] * cls_score[r_max, c_max] * p.lr

        res_xs = self.target_pos[0] + diff_xs
        res_ys = self.target_pos[1] + diff_ys
        res_w = pred_w * lr + (1 - lr) * target_sz_img[0]
        res_h = pred_h * lr + (1 - lr) * target_sz_img[1]

        self.target_pos = np.array([res_xs, res_ys])
        self.target_sz = target_sz_img * (1 - lr) + lr * np.array([res_w, res_h])

        # Clamp to image boundaries
        self.target_pos[0] = np.clip(self.target_pos[0], 0, self.im_w)
        self.target_pos[1] = np.clip(self.target_pos[1], 0, self.im_h)
        self.target_sz[0] = np.clip(self.target_sz[0], 10, self.im_w)
        self.target_sz[1] = np.clip(self.target_sz[1], 10, self.im_h)

        # Convert center+size → top-left x,y,w,h
        x = self.target_pos[0] - self.target_sz[0] / 2
        y = self.target_pos[1] - self.target_sz[1] / 2
        return True, (float(x), float(y), float(self.target_sz[0]), float(self.target_sz[1]))
