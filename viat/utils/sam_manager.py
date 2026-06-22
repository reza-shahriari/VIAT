import os
import cv2
import numpy as np

try:
    import torch
    from ultralytics import SAM
    ULTRALYTICS_AVAILABLE = True
except ImportError:
    ULTRALYTICS_AVAILABLE = False


class SamManager:
    """
    Manager for loading and running Segment Anything Models (SAM) for auto bounding box generation.
    """
    def __init__(self, checkpoints_dir="checkpoints"):
        self.checkpoints_dir = checkpoints_dir
        self.model = None
        self.current_model_type = None

    def is_available(self):
        return ULTRALYTICS_AVAILABLE

    def load_model(self, model_type="sam2_s.pt"):
        """
        Loads the SAM model.
        Returns (success_bool, message_string).
        """
        if not ULTRALYTICS_AVAILABLE:
            return False, "Ultralytics is not installed. Please run: pip install ultralytics"

        if self.current_model_type == model_type and self.model is not None:
            return True, "Model already loaded"

        try:
            import sys
            if getattr(sys, 'frozen', False):
                # Running as compiled PyInstaller executable
                project_root = os.path.dirname(sys.executable)
            else:
                # Running from source
                project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
                
            chkp_dir = os.path.join(project_root, self.checkpoints_dir)
            if not os.path.exists(chkp_dir):
                os.makedirs(chkp_dir, exist_ok=True)
            
            model_path = os.path.join(chkp_dir, model_type)
            
            # If the file doesn't exist locally, Ultralytics downloads it automatically.
            # To ensure it downloads to the checkpoints folder, we temporarily change cwd.
            if not os.path.exists(model_path):
                old_cwd = os.getcwd()
                os.chdir(chkp_dir)
                try:
                    self.model = SAM(model_type)
                finally:
                    os.chdir(old_cwd)
            else:
                self.model = SAM(model_path)
                
            self.current_model_type = model_type
            return True, "Model loaded successfully"
        except Exception as e:
            self.model = None
            self.current_model_type = None
            return False, f"Failed to load model: {str(e)}"

    def predict_bbox_from_point(self, image_array, x, y):
        """
        Given an image and a click coordinate (x, y), uses SAM to generate a mask 
        and extracts the bounding box.
        Returns (x_min, y_min, x_max, y_max) or None if prediction fails.
        """
        if not self.model:
            return None

        try:
            # Run inference. ultralytics accepts numpy arrays.
            # Convert point to format expected by ultralytics [[x, y]]
            # Label 1 means positive click.
            results = self.model(image_array, points=[[x, y]], labels=[1], verbose=False)
            
            if len(results) > 0 and results[0].boxes is not None:
                boxes = results[0].boxes.xyxy.cpu().numpy()
                if len(boxes) > 0:
                    box = boxes[0] # Take the first box
                    return (int(box[0]), int(box[1]), int(box[2]), int(box[3]))
            
            return None
        except Exception as e:
            print(f"Error during SAM inference: {e}")
            return None

    def predict_mask_from_box(self, image_array, box):
        """
        Given an image and a bounding box [x_min, y_min, x_max, y_max], 
        uses SAM to generate a pixel-perfect polygon mask.
        Returns a list of (x, y) tuples representing the polygon, or None.
        """
        if not self.model:
            return None

        try:
            results = self.model(image_array, bboxes=[box], verbose=False)
            
            if len(results) > 0 and results[0].masks is not None and len(results[0].masks.xy) > 0:
                polygon = results[0].masks.xy[0] # Take the first mask
                if len(polygon) > 0:
                    return [(float(pt[0]), float(pt[1])) for pt in polygon]
            
            return None
        except Exception as e:
            print(f"Error during SAM inference with box: {e}")
            return None
