import os
import cv2
import numpy as np
import traceback

class Sam2TrtManager:
    """
    Manager for loading and running the C++ TensorRT implementation of SAM2 using the sam2trt package.
    """
    def __init__(self, engine_dir="engines"):
        self.engine_dir = engine_dir
        self.image_predictor = None
        self.video_predictor = None
        self.current_model_type = None

    def _get_sam2_trt_dir(self):
        """Returns the absolute path to the sam2-trt directory."""
        return os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "sam2-trt")

    def is_available(self):
        try:
            import sam2trt
            return True
        except ImportError:
            return False

    def load_model(self, model_type="SAM2 TRT C++"):
        """
        Loads the TensorRT engines via the sam2trt package.
        """
        try:
            import sam2trt

            sam2_trt_dir = self._get_sam2_trt_dir()
            project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

            # Prefer engines/ next to the VIAT project root, fallback to sam2_trt/engines/
            engine_path = os.path.join(project_root, self.engine_dir)
            if not os.path.exists(engine_path):
                engine_path = os.path.join(sam2_trt_dir, "engines")
                
            model_size = sam2trt.ModelSize.SMALL
            if "tiny" in model_type.lower():
                model_size = sam2trt.ModelSize.TINY

            paths = sam2trt.EnginePaths.discover(engine_path, model_size)
            
            self.image_predictor = sam2trt.ImagePredictor(paths)
            self.video_predictor = sam2trt.VideoPredictor(paths)
            
            self.current_model_type = model_type
            return True, "SAM2 TRT Backend loaded successfully"

        except ImportError as e:
            self.image_predictor = None
            self.video_predictor = None
            return False, f"Could not import sam2trt. Did you compile the C++ extension? Error: {e}"
        except Exception as e:
            self.image_predictor = None
            self.video_predictor = None
            return False, f"Failed to load TRT engines: {str(e)}"

    def predict_mask_from_prompt(self, image_array, points=None, labels=None, box=None, text_prompt=None):
        """
        Uses the TRT backend to generate a mask (Magic Wand mode).
        """
        if not self.image_predictor:
            return None

        try:
            self.image_predictor.set_image(image_array)
            
            pts = []
            if points and labels:
                for p, l in zip(points, labels):
                    pts.append((float(p[0]), float(p[1]), bool(l)))
                    
            b = None
            if box:
                b = (float(box[0]), float(box[1]), float(box[2]), float(box[3]))
                
            if b:
                result = self.image_predictor.click_box(box=b, points=pts)
            elif pts:
                result = self.image_predictor.click_multi(points=pts)
            else:
                return None
            
            mask = result.best().mask
            mask_uint8 = (mask * 255).astype(np.uint8)
            
            contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
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
        if not self.video_predictor:
            yield False, "TRT Tracker not initialized."
            return

        try:
            self.video_predictor.init_state()
            
            pts = []
            if points and labels:
                for p, l in zip(points, labels):
                    pts.append((float(p[0]), float(p[1]), bool(l)))
                    
            b = None
            if box:
                b = (float(box[0]), float(box[1]), float(box[2]), float(box[3]))

            source_frames = list(frame_generator) if hasattr(frame_generator, '__iter__') and not isinstance(frame_generator, (list, str)) else frame_generator
            
            if isinstance(source_frames, str):
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
            obj_id = None
            
            for i, frame in enumerate(source_frames):
                if is_first_frame:
                    pts_arg = pts if pts else None
                    b_arg = b if b else None
                    if pts_arg:
                        res = self.video_predictor.add_prompt(frame_index=0, frame=frame, points=pts_arg)
                        obj_id = res.object_id
                        mask = res.mask
                    elif b_arg:
                        res = self.video_predictor.click_box(frame_index=0, frame=frame, box=b_arg)
                        obj_id = res.object_id
                        mask = res.mask
                    else:
                        mask = None
                    is_first_frame = False
                else:
                    tracks = self.video_predictor.track_frame(i, frame)
                    mask = None
                    for oid, mask_res in zip(tracks.object_ids, tracks.masks):
                        if oid == obj_id:
                            mask = mask_res
                            break
                    
                frame_polygons = []
                frame_boxes = []
                
                if mask is not None:
                    # mask could be a MaskResult object, so extract the numpy array if it has a .mask attribute
                    mask_arr = mask.mask if hasattr(mask, 'mask') else mask
                    mask_uint8 = (mask_arr * 255).astype(np.uint8)
                    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    if contours:
                        largest_contour = max(contours, key=cv2.contourArea)
                        polygon = [(float(pt[0][0]), float(pt[0][1])) for pt in largest_contour]
                        frame_polygons.append(polygon)
                        
                        x, y, w, h = cv2.boundingRect(largest_contour)
                        frame_boxes.append([x, y, x+w, y+h])
                    else:
                        frame_polygons.append(None)
                else:
                    frame_polygons.append(None)
                
                yield True, {"polygons": frame_polygons, "boxes": frame_boxes}
                
        except Exception as e:
            traceback.print_exc()
            yield False, f"TRT Tracking failed: {str(e)}"

    def track_video_from_boxes(self, frame_generator, bboxes, model_type=None):
        if not self.video_predictor:
            yield False, "TRT Tracker not initialized."
            return

        try:
            self.video_predictor.init_state()

            source_frames = list(frame_generator) if hasattr(frame_generator, '__iter__') and not isinstance(frame_generator, (list, str)) else frame_generator
            
            if isinstance(source_frames, str):
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
            obj_ids = []
            
            for i, frame in enumerate(source_frames):
                frame_polygons = []
                frame_boxes = []
                if is_first_frame:
                    for b in bboxes:
                        b_tuple = (float(b[0]), float(b[1]), float(b[2]), float(b[3]))
                        res = self.video_predictor.click_box(frame_index=0, frame=frame, box=b_tuple)
                        obj_ids.append(res.object_id)
                        
                        mask = res.mask
                        if mask is not None:
                            mask_arr = mask.mask if hasattr(mask, 'mask') else mask
                            mask_uint8 = (mask_arr * 255).astype(np.uint8)
                            contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                            if contours:
                                largest_contour = max(contours, key=cv2.contourArea)
                                polygon = [(float(pt[0][0]), float(pt[0][1])) for pt in largest_contour]
                                frame_polygons.append(polygon)
                                x, y, w, h = cv2.boundingRect(largest_contour)
                                frame_boxes.append([x, y, x+w, y+h])
                            else:
                                frame_polygons.append(None)
                        else:
                            frame_polygons.append(None)
                    is_first_frame = False
                else:
                    tracks = self.video_predictor.track_frame(i, frame)
                    mask_map = {oid: m for oid, m in zip(tracks.object_ids, tracks.masks)}
                    
                    for oid in obj_ids:
                        mask = mask_map.get(oid)
                        if mask is not None:
                            mask_arr = mask.mask if hasattr(mask, 'mask') else mask
                            mask_uint8 = (mask_arr * 255).astype(np.uint8)
                            contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                            if contours:
                                largest_contour = max(contours, key=cv2.contourArea)
                                polygon = [(float(pt[0][0]), float(pt[0][1])) for pt in largest_contour]
                                frame_polygons.append(polygon)
                                x, y, w, h = cv2.boundingRect(largest_contour)
                                frame_boxes.append([x, y, x+w, y+h])
                            else:
                                frame_polygons.append(None)
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
        if self.video_predictor:
            self.video_predictor.init_state()
