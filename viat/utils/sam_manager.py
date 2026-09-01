import os
import cv2
import numpy as np

_ULTRALYTICS_AVAILABLE = None

def check_ultralytics():
    global _ULTRALYTICS_AVAILABLE
    if _ULTRALYTICS_AVAILABLE is None:
        try:
            import ultralytics
            _ULTRALYTICS_AVAILABLE = True
        except ImportError:
            _ULTRALYTICS_AVAILABLE = False
    return _ULTRALYTICS_AVAILABLE


class SamManager:
    """
    Manager for loading and running Segment Anything Models (SAM) for auto bounding box generation.
    """
    def __init__(self, checkpoints_dir="checkpoints"):
        self.checkpoints_dir = checkpoints_dir
        self.model = None
        self.current_model_type = None

    def is_available(self):
        return check_ultralytics()

    def load_model(self, model_type="sam2.1_s.pt"):
        """
        Loads the SAM model.
        Returns (success_bool, message_string).
        """
        if not check_ultralytics():
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
                    from ultralytics import SAM
                    self.model = SAM(model_type)
                finally:
                    os.chdir(old_cwd)
            else:
                from ultralytics import SAM
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
        print(f"[DEBUG LOG] predict_bbox_from_point called with x={x}, y={y}")
        if not self.model:
            print("[DEBUG LOG] self.model is None. Returning None.")
            return None

        try:
            # Run inference. ultralytics accepts numpy arrays.
            # Format expected by ultralytics for a single point is [x, y]
            print(f"[DEBUG LOG] Running ultralytics SAM inference with points=[{x}, {y}] and labels=[1]")
            results = self.model(image_array, points=[x, y], labels=[1], verbose=False)
            print(f"[DEBUG LOG] Inference completed. len(results)={len(results)}")
            
            if len(results) > 0:
                result = results[0]
                print(f"[DEBUG LOG] result.boxes={result.boxes is not None}, len(boxes)={len(result.boxes) if result.boxes is not None else 0}")
                print(f"[DEBUG LOG] result.masks={result.masks is not None}, len(masks)={len(result.masks.xy) if result.masks is not None and hasattr(result.masks, 'xy') else 0}")
                
                # Check for boxes first
                if result.boxes is not None and len(result.boxes) > 0:
                    box = result.boxes.xyxy[0].cpu().float().numpy()
                    print(f"[DEBUG LOG] Returning box from result.boxes: {box}")
                    return (int(box[0]), int(box[1]), int(box[2]), int(box[3]))
                
                # If no boxes, derive from the mask
                if result.masks is not None and len(result.masks.xy) > 0:
                    mask = result.masks.xy[0]
                    if len(mask) > 0:
                        x_min = int(np.min(mask[:, 0]))
                        y_min = int(np.min(mask[:, 1]))
                        x_max = int(np.max(mask[:, 0]))
                        y_max = int(np.max(mask[:, 1]))
                        print(f"[DEBUG LOG] Derived box from mask: {(x_min, y_min, x_max, y_max)}")
                        return (x_min, y_min, x_max, y_max)
            
            print("[DEBUG LOG] returning None.")
            return None
        except Exception as e:
            print(f"[DEBUG LOG] Error during SAM inference: {e}")
            import traceback
            traceback.print_exc()
            return None

    def predict_bbox_from_box(self, image_array, box):
        """
        Given an image and a bounding box [x_min, y_min, x_max, y_max], 
        uses SAM to generate a mask and extracts the bounding box.
        Returns (x_min, y_min, x_max, y_max) or None if prediction fails.
        """
        print(f"[DEBUG LOG] predict_bbox_from_box called with box={box}")
        if not self.model:
            print("[DEBUG LOG] self.model is None. Returning None.")
            return None

        try:
            print(f"[DEBUG LOG] Running ultralytics SAM inference with bboxes={box}")
            results = self.model(image_array, bboxes=box, verbose=False)
            print(f"[DEBUG LOG] Inference completed. len(results)={len(results)}")
            
            if len(results) > 0:
                result = results[0]
                print(f"[DEBUG LOG] result.boxes={result.boxes is not None}, len(boxes)={len(result.boxes) if result.boxes is not None else 0}")
                print(f"[DEBUG LOG] result.masks={result.masks is not None}, len(masks)={len(result.masks.xy) if result.masks is not None and hasattr(result.masks, 'xy') else 0}")
                
                if result.boxes is not None and len(result.boxes) > 0:
                    b = result.boxes.xyxy[0].cpu().float().numpy()
                    print(f"[DEBUG LOG] Returning box from result.boxes: {b}")
                    return (int(b[0]), int(b[1]), int(b[2]), int(b[3]))
                
                if result.masks is not None and len(result.masks.xy) > 0:
                    mask = result.masks.xy[0]
                    if len(mask) > 0:
                        x_min = int(np.min(mask[:, 0]))
                        y_min = int(np.min(mask[:, 1]))
                        x_max = int(np.max(mask[:, 0]))
                        y_max = int(np.max(mask[:, 1]))
                        print(f"[DEBUG LOG] Derived box from mask: {(x_min, y_min, x_max, y_max)}")
                        return (x_min, y_min, x_max, y_max)
            
            print("[DEBUG LOG] returning None.")
            return None
        except Exception as e:
            print(f"[DEBUG LOG] Error during SAM inference with box: {e}")
            import traceback
            traceback.print_exc()
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
            results = self.model(image_array, bboxes=box, verbose=False)
            
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
                if isinstance(points[0], (int, float)):
                    pts_fmt = [[[float(points[0]), float(points[1])]]]
                    lbls_fmt = [[int(labels[0]) if isinstance(labels, (list, tuple)) else int(labels)]]
                elif isinstance(points[0], (list, tuple)) and isinstance(points[0][0], (int, float)):
                    pts_fmt = [[[float(p[0]), float(p[1])] for p in points]]
                    lbls_fmt = [[int(l) for l in labels]]
                else:
                    pts_fmt = points
                    lbls_fmt = labels
                kwargs['points'] = pts_fmt
                kwargs['labels'] = lbls_fmt
            if box:
                if isinstance(box[0], (int, float)):
                    kwargs['bboxes'] = [[float(b) for b in box]]
                else:
                    kwargs['bboxes'] = box
            if text_prompt and self.current_model_type and "fastsam" in self.current_model_type.lower():
                kwargs['texts'] = text_prompt

            results = self.model(image_array, **kwargs)
            
            if len(results) > 0 and results[0].masks is not None and len(results[0].masks.xy) > 0:
                valid_polygons = [m for m in results[0].masks.xy if len(m) >= 3]
                if valid_polygons:
                    largest_poly = max(valid_polygons, key=lambda p: cv2.contourArea(np.array(p, dtype=np.int32)))
                    return [(float(pt[0]), float(pt[1])) for pt in largest_poly]
                elif len(results[0].masks.xy[0]) > 0:
                    polygon = results[0].masks.xy[0]
                    return [(float(pt[0]), float(pt[1])) for pt in polygon]
            
            return None
        except Exception as e:
            print(f"Error during SAM inference with prompt: {e}")
            return None


    def track_video_from_boxes(self, frame_generator, bboxes, model_type="sam3.1_s.pt", start_f=0, end_f=None):
        """
        frame_generator: a Python generator yielding numpy arrays (frames), or a string video file path.
        bboxes: list of bounding boxes [[x1, y1, x2, y2], ...] corresponding to objects in the FIRST frame yielded.
        Returns a generator yielding lists of polygons corresponding to the bboxes for each frame.
        """
        if not check_ultralytics():
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
            
            if not getattr(self, 'video_predictor', None) or getattr(self, 'video_predictor_type', None) != model_type:
                overrides = dict(conf=0.25, task="segment", mode="predict", model=model_path, verbose=False, imgsz=1024, save=False)
                if "sam3" in model_type:
                    self.video_predictor = SAM3VideoPredictor(overrides=overrides)
                else:
                    self.video_predictor = SAM2VideoPredictor(overrides=overrides)
                
                # Pre-load and compile the model for massive speedups
                try:
                    import torch
                    print(f"Loading and compiling PyTorch model: {model_type}...")
                    self.video_predictor.setup_model(model=None, verbose=False)
                    if hasattr(self.video_predictor, 'model'):
                        # Using default compilation mode for faster JIT warmup
                        self.video_predictor.model = torch.compile(self.video_predictor.model)
                        print("SAM Video model successfully compiled with torch.compile!")
                except Exception as comp_err:
                    print(f"Warning: torch.compile failed or skipped: {comp_err}")
                    
                self.video_predictor_type = model_type
                
            predictor = self.video_predictor
            # Force reset inference state to prevent prompt accumulation and crossover
            predictor.inference_state = {}
                
            import tempfile
            import uuid
            temp_video_path = None
            
            try:
                if isinstance(frame_generator, str):
                    source_to_predict = frame_generator
                elif isinstance(frame_generator, list) and len(frame_generator) > 0 and isinstance(frame_generator[0], str):
                    source_to_predict = frame_generator
                else:
                    # Stream frame_generator into VideoWriter to avoid list() memory bloat
                    temp_video_path = os.path.join(tempfile.gettempdir(), f"sam2_temp_{uuid.uuid4().hex}.mp4")
                    out = None
                    for frame in frame_generator:
                        if isinstance(frame, np.ndarray):
                            if out is None:
                                h, w = frame.shape[:2]
                                out = cv2.VideoWriter(temp_video_path, cv2.VideoWriter_fourcc(*'mp4v'), 30, (w, h))
                            out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                    if out is not None:
                        out.release()
                    source_to_predict = temp_video_path

                results_stream = predictor(source=source_to_predict, bboxes=bboxes, stream=True)
                
                for idx, result in enumerate(results_stream):
                    if start_f is not None and isinstance(source_to_predict, str) and not temp_video_path:
                        if idx < start_f:
                            continue
                        if end_f is not None and idx > end_f:
                            break

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
                            b = box.cpu().float().numpy()
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

    def track_video_from_prompt(self, frame_generator, points=None, labels=None, box=None, text_prompt=None, model_type="sam3.1_l.pt", start_f=0, end_f=None):
        """
        frame_generator: a Python generator yielding numpy arrays (frames), or a string video file path.
        Returns a generator yielding lists of polygons corresponding to the prompts for each frame.
        """
        if not check_ultralytics():
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
            
            if not getattr(self, 'video_predictor', None) or getattr(self, 'video_predictor_type', None) != model_type:
                overrides = dict(conf=0.25, task="segment", mode="predict", model=model_path, verbose=False, imgsz=1024, save=False)
                if "sam3" in model_type:
                    self.video_predictor = SAM3VideoPredictor(overrides=overrides)
                else:
                    self.video_predictor = SAM2VideoPredictor(overrides=overrides)
                
                # Pre-load and compile the model for massive speedups
                try:
                    import torch
                    print(f"Loading and compiling PyTorch model: {model_type}...")
                    self.video_predictor.setup_model(model=None, verbose=False)
                    if hasattr(self.video_predictor, 'model'):
                        # Using default compilation mode for faster JIT warmup
                        self.video_predictor.model = torch.compile(self.video_predictor.model)
                        print("SAM Video model successfully compiled with torch.compile!")
                except Exception as comp_err:
                    print(f"Warning: torch.compile failed or skipped: {comp_err}")
                    
                self.video_predictor_type = model_type
                
            predictor = self.video_predictor
            # Force reset inference state to prevent prompt accumulation and crossover
            predictor.inference_state = {}
                
            kwargs = {'stream': True}
            if points and labels:
                if isinstance(points[0], (int, float)):
                    pts_fmt = [[[float(points[0]), float(points[1])]]]
                    lbls_fmt = [[int(labels[0]) if isinstance(labels, (list, tuple)) else int(labels)]]
                elif isinstance(points[0], (list, tuple)) and isinstance(points[0][0], (int, float)):
                    pts_fmt = [[[float(p[0]), float(p[1])] for p in points]]
                    lbls_fmt = [[int(l) for l in labels]]
                else:
                    pts_fmt = points
                    lbls_fmt = labels
                kwargs['points'] = pts_fmt
                kwargs['labels'] = lbls_fmt
            if box:
                if isinstance(box[0], (int, float)):
                    kwargs['bboxes'] = [[float(b) for b in box]]
                else:
                    kwargs['bboxes'] = box
            if text_prompt and ("fastsam" in model_type.lower() or "sam3" in model_type.lower()):
                kwargs['texts'] = text_prompt

            import tempfile
            import uuid
            temp_video_path = None
            
            try:
                if isinstance(frame_generator, str):
                    source_to_predict = frame_generator
                elif isinstance(frame_generator, list) and len(frame_generator) > 0 and isinstance(frame_generator[0], str):
                    source_to_predict = frame_generator
                else:
                    # Stream frame_generator into VideoWriter to avoid list() memory bloat
                    temp_video_path = os.path.join(tempfile.gettempdir(), f"sam2_temp_{uuid.uuid4().hex}.mp4")
                    out = None
                    for frame in frame_generator:
                        if isinstance(frame, np.ndarray):
                            if out is None:
                                h, w = frame.shape[:2]
                                out = cv2.VideoWriter(temp_video_path, cv2.VideoWriter_fourcc(*'mp4v'), 30, (w, h))
                            out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                    if out is not None:
                        out.release()
                    source_to_predict = temp_video_path

                results_stream = predictor(source=source_to_predict, **kwargs)
                
                for idx, result in enumerate(results_stream):
                    if start_f is not None and isinstance(source_to_predict, str) and not temp_video_path:
                        if idx < start_f:
                            continue
                        if end_f is not None and idx > end_f:
                            break

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
                            b = b_.cpu().float().numpy()
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

    def clear_session(self):
        """
        Resets the predictor inference state.
        """
        if hasattr(self, 'video_predictor') and self.video_predictor is not None:
            self.video_predictor.inference_state = {}
