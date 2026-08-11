import os
import cv2
import numpy as np
import traceback

class Sam2TrtManager:
    """
    Manager for loading and running the C++ TensorRT implementation of SAM2.
    """
    def __init__(self, engine_dir="engines"):
        self.engine_dir = engine_dir
        self.tracker = None
        self.current_model_type = None

    def _get_sam2_trt_dir(self):
        """Returns the absolute path to the sam2_trt directory."""
        return os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "sam2_trt")

    def _import_cpp_module(self):
        """Import sam2_trt_cpp .so by adding its directory directly to sys.path."""
        import sys
        sam2_trt_dir = self._get_sam2_trt_dir()
        if sam2_trt_dir not in sys.path:
            sys.path.insert(0, sam2_trt_dir)
        import importlib
        return importlib.import_module("sam2_trt_cpp")

    def is_available(self):
        try:
            self._import_cpp_module()
            return True
        except ImportError:
            return False

    def load_model(self, model_type="SAM2 TRT C++"):
        """
        Loads the TensorRT engines via PyBind11 wrapper.
        """
        try:
            sam2_trt_cpp = self._import_cpp_module()

            sam2_trt_dir = self._get_sam2_trt_dir()
            project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

            # Prefer engines/ next to the VIAT project root, fallback to sam2_trt/engines/
            engine_path = os.path.join(project_root, self.engine_dir)
            if not os.path.exists(engine_path):
                engine_path = os.path.join(sam2_trt_dir, "engines")

            self.tracker = sam2_trt_cpp.Sam2TrtTracker(engine_path)
            self.current_model_type = model_type
            return True, "SAM2 TRT C++ Backend loaded successfully"

        except ImportError as e:
            self.tracker = None
            return False, f"Could not import sam2_trt_cpp. Did you compile the C++ extension? Error: {e}"
        except Exception as e:
            self.tracker = None
            return False, f"Failed to load TRT engines: {str(e)}"

    def predict_mask_from_prompt(self, image_array, points=None, labels=None, box=None, text_prompt=None):
        """
        Uses the TRT backend to generate a mask (Magic Wand mode).
        Does NOT update tracking state.
        """
        if not self.tracker:
            return None

        try:
            cpp_points = []
            if points and labels:
                for p, l in zip(points, labels):
                    cpp_points.append(((p[0], p[1]), l))
                    
            cpp_box = []
            if box:
                cpp_box = [float(b) for b in box]
                
            # Process image (returns numpy array of shape (H, W) type uint8)
            mask = self.tracker.process_image(image_array, cpp_points, cpp_box)
            
            # Extract polygon from mask
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                largest_contour = max(contours, key=cv2.contourArea)
                polygon = [(float(pt[0][0]), float(pt[0][1])) for pt in largest_contour]
                return polygon
                
            return None
            
        except Exception as e:
            print(f"Error during SAM TRT inference with prompt: {e}")
            traceback.print_exc()
            return None

    def track_video_from_prompt(self, frame_generator, points=None, labels=None, box=None, text_prompt=None, model_type=None):
        """
        Video tracking using TRT Memory bank.
        """
        if not self.tracker:
            yield False, "TRT Tracker not initialized."
            return

        try:
            # Force reset inference state
            self.tracker.init_state()
            
            cpp_points = []
            if points and labels:
                for p, l in zip(points, labels):
                    cpp_points.append(((p[0], p[1]), l))
                    
            cpp_box = []
            if box:
                cpp_box = [float(b) for b in box]

            source_frames = list(frame_generator) if hasattr(frame_generator, '__iter__') and not isinstance(frame_generator, (list, str)) else frame_generator
            
            if isinstance(source_frames, str):
                # If it's a video file, we need to read it manually
                cap = cv2.VideoCapture(source_frames)
                frames = []
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                cap.release()
                source_frames = frames

            is_first_frame = True
            
            for frame in source_frames:
                if is_first_frame:
                    mask = self.tracker.add_prompt(frame, cpp_points, cpp_box)
                    is_first_frame = False
                else:
                    mask = self.tracker.track_frame(frame)
                    
                # Extract polygon
                frame_polygons = []
                frame_boxes = []
                
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if contours:
                    largest_contour = max(contours, key=cv2.contourArea)
                    polygon = [(float(pt[0][0]), float(pt[0][1])) for pt in largest_contour]
                    frame_polygons.append(polygon)
                    
                    x, y, w, h = cv2.boundingRect(largest_contour)
                    frame_boxes.append([x, y, x+w, y+h])
                else:
                    frame_polygons.append(None)
                
                yield True, {"polygons": frame_polygons, "boxes": frame_boxes}
                
        except Exception as e:
            traceback.print_exc()
            yield False, f"TRT Tracking failed: {str(e)}"

    def clear_session(self):
        """
        Resets the predictor inference state.
        """
        if self.tracker:
            self.tracker.init_state()
