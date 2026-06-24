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
                self.model.predictor.is_cli = False  # restore
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

    def predict_bbox_from_box(self, image_array, box):
        """
        Given an image and a bounding box [x_min, y_min, x_max, y_max], 
        uses SAM to generate a mask and extracts the bounding box.
        Returns (x_min, y_min, x_max, y_max) or None if prediction fails.
        """
        if not self.model:
            return None

        try:
            results = self.model(image_array, bboxes=[box], verbose=False)
            
            if len(results) > 0 and results[0].boxes is not None:
                boxes = results[0].boxes.xyxy.cpu().numpy()
                if len(boxes) > 0:
                    b = boxes[0] # Take the first box
                    return (int(b[0]), int(b[1]), int(b[2]), int(b[3]))
            
            return None
        except Exception as e:
            print(f"Error during SAM inference with box: {e}")
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
    def predict_mask_from_prompt(self, image_array, points=None, labels=None, box=None, text_prompt=None):
        """
        Uses SAM to generate a mask from a combination of points, bounding box, and text prompt.
        Returns a list of (x, y) tuples representing the polygon, or None.
        """
        if not self.model:
            return None

        try:
            kwargs = {'verbose': False, 'imgsz': 1024}
            if points and labels:
                # Ultralytics expects [[[x1,y1], [x2,y2]]] for a single image with multiple points.
                if len(points) > 0 and not isinstance(points[0][0], list):
                    kwargs['points'] = [points]
                    kwargs['labels'] = [labels]
                else:
                    kwargs['points'] = points
                    kwargs['labels'] = labels
            if box:
                kwargs['bboxes'] = [box]
            if text_prompt and self.current_model_type and "fastsam" in self.current_model_type.lower():
                kwargs['texts'] = text_prompt

            results = self.model(image_array, **kwargs)
            
            if len(results) > 0 and results[0].masks is not None and len(results[0].masks.xy) > 0:
                polygon = results[0].masks.xy[0]
                if len(polygon) > 0:
                    return [(float(pt[0]), float(pt[1])) for pt in polygon]
            
            return None
        except Exception as e:
            print(f"Error during SAM inference with prompt: {e}")
            return None


    def track_video_from_boxes(self, frame_generator, bboxes, model_type="sam3_s.pt"):
        """
        frame_generator: a Python generator that yields numpy arrays (frames).
        bboxes: list of bounding boxes [[x1, y1, x2, y2], ...] corresponding to objects in the FIRST frame yielded.
        Returns a generator yielding lists of polygons corresponding to the bboxes for each frame.
        """
        if not ULTRALYTICS_AVAILABLE:
            yield False, "Ultralytics is not installed."
            return

        try:
            from ultralytics.models.sam import SAM3VideoPredictor, SAM2VideoPredictor
            from ultralytics.utils.downloads import attempt_download_asset
            
            import sys
            project_root = os.path.dirname(sys.executable) if getattr(sys, 'frozen', False) else os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            chkp_dir = os.path.join(project_root, self.checkpoints_dir)
            model_path = os.path.join(chkp_dir, model_type)
            
            if not os.path.exists(model_path):
                old_cwd = os.getcwd()
                os.chdir(chkp_dir)
                try:
                    attempt_download_asset(model_type)
                finally:
                    os.chdir(old_cwd)
            
            overrides = dict(conf=0.25, task="segment", mode="predict", model=model_path, half=True, verbose=False, imgsz=1024)
            
            if "sam3" in model_type:
                predictor = SAM3VideoPredictor(overrides=overrides)
            else:
                predictor = SAM2VideoPredictor(overrides=overrides)
                
            # Predictor requires list or path, not generator
            source_frames = list(frame_generator) if hasattr(frame_generator, '__iter__') and not isinstance(frame_generator, (list, str)) else frame_generator
            
            import tempfile
            import uuid
            temp_video_path = None
            
            try:
                if isinstance(source_frames, list) and len(source_frames) > 0 and isinstance(source_frames[0], np.ndarray):
                    temp_video_path = os.path.join(tempfile.gettempdir(), f"sam2_temp_{uuid.uuid4().hex}.mp4")
                    h, w = source_frames[0].shape[:2]
                    out = cv2.VideoWriter(temp_video_path, cv2.VideoWriter_fourcc(*'mp4v'), 30, (w, h))
                    for frame in source_frames:
                        out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                    out.release()
                    source_to_predict = temp_video_path
                else:
                    source_to_predict = source_frames

                results_stream = predictor(source=source_to_predict, bboxes=bboxes, stream=True)
                
                for result in results_stream:
                    frame_polygons = []
                    if result.masks is not None:
                        masks_xy = result.masks.xy
                        for mask in masks_xy:
                            if len(mask) >= 3:
                                polygon = [(int(x), int(y)) for x, y in mask]
                                frame_polygons.append(polygon)
                            else:
                                frame_polygons.append(None)
                    
                    frame_boxes = []
                    if result.boxes is not None:
                        for box in result.boxes.xyxy:
                            b = box.cpu().numpy()
                            frame_boxes.append([int(b[0]), int(b[1]), int(b[2]), int(b[3])])
                            
                    yield True, {"polygons": frame_polygons, "boxes": frame_boxes}
                
            finally:
                if temp_video_path and os.path.exists(temp_video_path):
                    try:
                        os.remove(temp_video_path)
                    except:
                        pass
                
        except Exception as e:
            yield False, f"Tracking failed: {str(e)}"

    def track_video_from_prompt(self, frame_generator, points=None, labels=None, box=None, text_prompt=None, model_type="sam3_l.pt"):
        """
        frame_generator: a Python generator that yields numpy arrays (frames).
        Returns a generator yielding lists of polygons corresponding to the prompts for each frame.
        """
        if not ULTRALYTICS_AVAILABLE:
            yield False, "Ultralytics is not installed."
            return

        try:
            from ultralytics.models.sam import SAM3VideoPredictor, SAM2VideoPredictor
            from ultralytics.utils.downloads import attempt_download_asset
            
            import sys
            project_root = os.path.dirname(sys.executable) if getattr(sys, 'frozen', False) else os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            chkp_dir = os.path.join(project_root, self.checkpoints_dir)
            model_path = os.path.join(chkp_dir, model_type)
            
            if not os.path.exists(model_path):
                old_cwd = os.getcwd()
                os.chdir(chkp_dir)
                try:
                    attempt_download_asset(model_type)
                finally:
                    os.chdir(old_cwd)
            
            overrides = dict(conf=0.25, task="segment", mode="predict", model=model_path, half=True, verbose=False, imgsz=1024)
            
            if "sam3" in model_type:
                predictor = SAM3VideoPredictor(overrides=overrides)
            else:
                predictor = SAM2VideoPredictor(overrides=overrides)
                
            kwargs = {'stream': True}
            if points and labels:
                if len(points) > 0 and not isinstance(points[0][0], list):
                    kwargs['points'] = [points]
                    kwargs['labels'] = [labels]
                else:
                    kwargs['points'] = points
                    kwargs['labels'] = labels
            if box:
                kwargs['bboxes'] = [box]
            if text_prompt and "fastsam" in model_type.lower():
                kwargs['texts'] = text_prompt

            source_frames = list(frame_generator) if hasattr(frame_generator, '__iter__') and not isinstance(frame_generator, (list, str)) else frame_generator
            
            # Ultralytics SAM2VideoPredictor strictly requires a video file path (dataset.mode == "video").
            # If we received a list of numpy arrays, we must write them to a temporary video file.
            import tempfile
            import uuid
            temp_video_path = None
            
            try:
                if isinstance(source_frames, list) and len(source_frames) > 0 and isinstance(source_frames[0], np.ndarray):
                    temp_video_path = os.path.join(tempfile.gettempdir(), f"sam2_temp_{uuid.uuid4().hex}.mp4")
                    h, w = source_frames[0].shape[:2]
                    out = cv2.VideoWriter(temp_video_path, cv2.VideoWriter_fourcc(*'mp4v'), 30, (w, h))
                    for frame in source_frames:
                        # frames from main.py generator are RGB, so convert to BGR for cv2
                        out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                    out.release()
                    source_to_predict = temp_video_path
                else:
                    source_to_predict = source_frames

                results_stream = predictor(source=source_to_predict, **kwargs)
                
                for result in results_stream:
                    frame_polygons = []
                    if result.masks is not None:
                        masks_xy = result.masks.xy
                        for mask in masks_xy:
                            if len(mask) >= 3:
                                polygon = [(int(x), int(y)) for x, y in mask]
                                frame_polygons.append(polygon)
                            else:
                                frame_polygons.append(None)
                    
                    frame_boxes = []
                    if result.boxes is not None:
                        for b_ in result.boxes.xyxy:
                            b = b_.cpu().numpy()
                            frame_boxes.append([int(b[0]), int(b[1]), int(b[2]), int(b[3])])
                            
                    yield True, {"polygons": frame_polygons, "boxes": frame_boxes}
                
            finally:
                if temp_video_path and os.path.exists(temp_video_path):
                    try:
                        os.remove(temp_video_path)
                    except:
                        pass
                
        except Exception as e:
            import traceback
            traceback.print_exc()
            yield False, f"Tracking failed: {str(e)}"
