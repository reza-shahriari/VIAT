import os
import cv2
import numpy as np
import tempfile
from PIL import Image

_SAM3_AVAILABLE = None
def check_sam3():
    global _SAM3_AVAILABLE
    if _SAM3_AVAILABLE is None:
        try:
            import sam3
            _SAM3_AVAILABLE = True
        except ImportError:
            _SAM3_AVAILABLE = False
    return _SAM3_AVAILABLE

def abs_to_rel_coords(coords, IMG_WIDTH, IMG_HEIGHT, coord_type="point"):
    if coord_type == "point":
        return [[x / IMG_WIDTH, y / IMG_HEIGHT] for x, y in coords]
    elif coord_type == "box":
        return [[x1 / IMG_WIDTH, y1 / IMG_HEIGHT, x2 / IMG_WIDTH, y2 / IMG_HEIGHT] for x1, y1, x2, y2 in coords]
    return coords

class Sam3NativeManager:
    """
    Native Manager for SAM3 models from facebookresearch/sam3.
    """
    def __init__(self, checkpoints_dir="checkpoints"):
        self.checkpoints_dir = checkpoints_dir
        self.image_model = None
        self.image_processor = None
        self.video_predictor = None
        self.current_model_type = None
        self.current_session_id = None
        self.current_img_hash = None
        self.current_box = None
        self.current_text_prompt = None

    def is_available(self):
        return check_sam3()

    def load_model(self, model_type="sam3.1_l.pt"):
        """
        Loads the SAM3 model.
        Returns (success_bool, message_string).
        """
        if not check_sam3():
            return False, "SAM3 native package is not installed."

        if self.current_model_type == model_type and self.image_model is not None:
            return True, "Model already loaded"

        try:
            import sys
            project_root = os.path.dirname(sys.executable) if getattr(sys, 'frozen', False) else os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
                
            chkp_dir = os.path.join(project_root, self.checkpoints_dir)
            model_path = os.path.join(chkp_dir, model_type)
            
            if not os.path.exists(model_path):
                return False, f"Model checkpoint not found at {model_path}. Please download it."

            # Set CUDA memory allocator config to reduce fragmentation
            os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
            
            old_cwd = os.getcwd()
            os.chdir(chkp_dir)
            try:
                # We load both image and video models depending on the need
                if "sam3.1" in model_type.lower():
                    from sam3.model_builder import build_sam3_predictor
                    self.video_predictor = build_sam3_predictor(checkpoint_path=model_path, version="sam3.1", use_fa3=False)
                else:
                    from sam3.model_builder import build_sam3_video_predictor
                    self.video_predictor = build_sam3_video_predictor(checkpoint_path=model_path)
                    
                self.current_session_id = None
                self.current_img_hash = None
                self.current_box = None
                self.current_text_prompt = None
                self.temp_dir = os.path.join(tempfile.gettempdir(), "sam3_native_session")
                os.makedirs(self.temp_dir, exist_ok=True)
            finally:
                os.chdir(old_cwd)
                
            self.current_model_type = model_type
            return True, "SAM3 Native Model loaded successfully"
        except Exception as e:
            self.image_model = None
            self.video_predictor = None
            self.current_model_type = None
            import traceback
            traceback.print_exc()
            return False, f"Failed to load SAM3 native model: {str(e)}"

    def predict_mask_from_prompt(self, image_array, points=None, labels=None, box=None, text_prompt=None):
        """
        Uses SAM3 to generate a mask. image_array is RGB numpy array.
        """
        if not self.video_predictor:
            return None

        import hashlib
        import tempfile
        import os
        import cv2
        import torch
        import numpy as np
        
        # Hash image to see if it's new
        img_bytes = image_array.tobytes()
        img_hash = hashlib.md5(img_bytes).hexdigest()

        # Check if the box or text prompt has changed (indicating a new object/prompt query)
        box_tuple = tuple(box) if box is not None else None
        prompt_changed = (box_tuple != self.current_box) or (text_prompt != self.current_text_prompt)

        if self.current_session_id is None or img_hash != self.current_img_hash or prompt_changed:
            # New image or new prompt, start new session to prevent prompt accumulation
            if self.current_session_id is not None:
                try:
                    self.video_predictor.handle_request({"type": "close_session", "session_id": self.current_session_id})
                except Exception as e:
                    print(f"[SAM3 Native] Failed to close session: {e}")
            
            # Save image
            img_path = os.path.join(self.temp_dir, "00000.jpg")
            # BGR to RGB for cv2 write
            cv2.imwrite(img_path, cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR))
            
            res = self.video_predictor.handle_request({"type": "start_session", "resource_path": self.temp_dir, "offload_video_to_cpu": True})
            self.current_session_id = res["session_id"]
            self.current_img_hash = img_hash
            self.current_box = box_tuple
            self.current_text_prompt = text_prompt
        else:
            # Same image and same box/text prompt (adding/refining points), clear points in the session
            self.video_predictor.handle_request({"type": "reset_session", "session_id": self.current_session_id})

        IMG_HEIGHT, IMG_WIDTH = image_array.shape[:2]

        has_points = points is not None and labels is not None
        has_box = box is not None
        has_text = bool(text_prompt)

        if has_points:
            # SAM3 requires points to be sent in a SEPARATE call, exclusive of text/box.
            # First send text/box as a VG prompt (if any), then add points as a SAM2 instance prompt.
            if has_text or has_box:
                vg_req = {
                    "type": "add_prompt",
                    "session_id": self.current_session_id,
                    "frame_index": 0,
                    "obj_id": 1
                }
                if has_text:
                    vg_req["text"] = text_prompt
                if has_box:
                    box_rel = abs_to_rel_coords(np.array([box]), IMG_WIDTH, IMG_HEIGHT, coord_type="box")
                    vg_req["box"] = torch.tensor(box_rel, dtype=torch.float32)
                self.video_predictor.handle_request(vg_req)

            # Points-only prompt
            rel_points = [[p[0] / IMG_WIDTH, p[1] / IMG_HEIGHT] for p in points]
            pt_req = {
                "type": "add_prompt",
                "session_id": self.current_session_id,
                "frame_index": 0,
                "obj_id": 1,
                "points": rel_points,
                "point_labels": labels,
            }
            self.video_predictor.handle_request(pt_req)
        else:
            # Text and/or box only
            # SAM3 handle_request maps:
            #   "bounding_boxes"       → add_prompt(bounding_boxes=...) → model(boxes_xywh=...)
            #   "bounding_box_labels"  → required companion (1=fg per box)
            prompt_req = {
                "type": "add_prompt",
                "session_id": self.current_session_id,
                "frame_index": 0,
                "obj_id": 1,
                # Use a lower threshold for text prompts to detect objects more leniently
                "output_prob_thresh": 0.1,
            }
            if has_text:
                prompt_req["text"] = text_prompt
            if has_box:
                # Convert absolute xyxy → normalized xywh
                x1n = box[0] / IMG_WIDTH
                y1n = box[1] / IMG_HEIGHT
                wn  = (box[2] - box[0]) / IMG_WIDTH
                hn  = (box[3] - box[1]) / IMG_HEIGHT
                x1n = max(0.0, min(1.0, x1n))
                y1n = max(0.0, min(1.0, y1n))
                wn  = max(0.0, min(1.0, wn))
                hn  = max(0.0, min(1.0, hn))
                prompt_req["bounding_boxes"] = torch.tensor([[x1n, y1n, wn, hn]], dtype=torch.float32)
                prompt_req["bounding_box_labels"] = torch.tensor([1], dtype=torch.int32)
                
            print(f"[SAM3 Debug] Sending request with keys: {list(prompt_req.keys())}")
            if has_text:
                print(f"[SAM3 Debug] Text prompt: {text_prompt}")
            if has_box:
                print(f"[SAM3 Debug] Box prompt (xywh): {x1n:.3f}, {y1n:.3f}, {wn:.3f}, {hn:.3f}")

            res = self.video_predictor.handle_request(prompt_req)
            print(f"[SAM3 Native] add_prompt returned dict keys: {res.keys()}")
            if "outputs" in res:
                print(f"[SAM3 Native] add_prompt outputs keys: {res['outputs'].keys() if res['outputs'] else 'None'}")
                if res['outputs'] and "out_binary_masks" in res['outputs']:
                    print(f"[SAM3 Native] add_prompt out_binary_masks shape: {res['outputs']['out_binary_masks'].shape}")
        
        # Propagate to get mask
        outputs = res.get("outputs") if 'res' in locals() and res.get("outputs") is not None else None
        
        # fallback to stream if add_prompt didn't yield outputs
        if outputs is None:
            print("[SAM3 Native] outputs is None, trying propagate_in_video")
            for out in self.video_predictor.handle_stream_request({"type": "propagate_in_video", "session_id": self.current_session_id, "start_frame_index": 0}):
                print(f"[SAM3 Native] stream out frame_index: {out.get('frame_index')}")
                if out.get("frame_index") == 0:
                    outputs = out.get("outputs")
                    break

        if outputs is None:
            print("[SAM3 Native] No outputs from SAM3 predictor")
            return None

        out_obj_ids = outputs.get("out_obj_ids", None)
        out_binary_masks = outputs.get("out_binary_masks", None)

        print(f"[SAM3 Native] out_obj_ids: {out_obj_ids}, out_binary_masks shape: {out_binary_masks.shape if out_binary_masks is not None else 'None'}")

        if out_binary_masks is None or len(out_binary_masks) == 0:
            print("[SAM3 Native] out_binary_masks is empty or None")
            return None

        # Take mask for obj_id=1 if present, otherwise take the first one
        mask = None
        if out_obj_ids is not None:
            for i, oid in enumerate(out_obj_ids):
                if int(oid) == 1:
                    mask = out_binary_masks[i]
                    break
        if mask is None and len(out_binary_masks) > 0:
            mask = out_binary_masks[0]

        if mask is None:
            return None

        mask = mask.astype(bool)
        mask_uint8 = mask.astype(np.uint8) * 255

        print(f"[SAM3 Native] mask max value: {mask_uint8.max()}")

        if mask_uint8.max() == 0:
            print("[SAM3 Native] mask is completely empty (all 0s)")
            return None

        # Find contours
        contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            print("[SAM3 Native] cv2.findContours found no contours.")
            return None

        # Return largest contour
        largest_contour = max(contours, key=cv2.contourArea)
        polygon_pts = largest_contour.reshape(-1, 2).tolist()
        
        # --- DEBUG IMAGE SAVING ---
        try:
            import time
            debug_img = cv2.cvtColor(image_array.copy(), cv2.COLOR_RGB2BGR)
            if has_box:
                cv2.rectangle(debug_img, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 0, 255), 2)
                cv2.putText(debug_img, "Prompt Box", (int(box[0]), int(box[1]-5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)
            if has_text:
                cv2.putText(debug_img, f"Text: {text_prompt}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2)
                
            pts = np.array(polygon_pts, np.int32).reshape((-1, 1, 2))
            cv2.polylines(debug_img, [pts], isClosed=True, color=(0, 255, 0), thickness=2)
            
            dbg_path = os.path.join(tempfile.gettempdir(), f"sam3_debug_{int(time.time()*1000)}.jpg")
            cv2.imwrite(dbg_path, debug_img)
            print(f"[SAM3 Debug] Saved debug image to {dbg_path}")
        except Exception as e:
            print(f"[SAM3 Debug] Failed to save debug image: {e}")
            
        return polygon_pts

    def track_video_from_prompt(self, resource_path, start_f, end_f, points=None, labels=None, box=None, text_prompt=None):
        """
        Tracks an object in video using SAM3.
        Yields (success_bool, {"polygons": [...], "boxes": [...]}) for each frame.
        """
        if not self.video_predictor:
            yield False, "SAM3 Video Predictor not loaded."
            return
            
        # Close any active preview session to free VRAM and clear state variables
        if self.current_session_id is not None:
            try:
                self.video_predictor.handle_request({"type": "close_session", "session_id": self.current_session_id})
            except Exception as e:
                print(f"[SAM3 Native] Failed to close active preview session: {e}")
            self.current_session_id = None
            self.current_img_hash = None
            self.current_box = None
            self.current_text_prompt = None

        try:
            # Start session with CPU offloading to reduce VRAM usage
            response = self.video_predictor.handle_request(
                request=dict(
                    type="start_session",
                    resource_path=resource_path,
                    offload_video_to_cpu=True,
                )
            )
            session_id = response["session_id"]
            
            # Reset session just in case
            self.video_predictor.handle_request(
                request=dict(
                    type="reset_session",
                    session_id=session_id,
                )
            )
            
            obj_id = 1
            
            # Determine W, H
            cap = cv2.VideoCapture(resource_path) if isinstance(resource_path, str) and resource_path.endswith('.mp4') else None
            W, H = 1920, 1080
            if cap:
                ret, frame = cap.read()
                if ret:
                    H, W = frame.shape[:2]
                cap.release()
            elif isinstance(resource_path, str) and os.path.isdir(resource_path):
                import glob
                imgs = glob.glob(os.path.join(resource_path, "*.jpg"))
                if imgs:
                    img = cv2.imread(imgs[0])
                    if img is not None:
                        H, W = img.shape[:2]

            if text_prompt:
                self.video_predictor.handle_request(
                    request=dict(
                        type="add_prompt",
                        session_id=session_id,
                        frame_index=start_f,
                        text=text_prompt,
                        obj_id=obj_id,
                    )
                )
                
            if box:
                # Convert absolute xyxy → normalized xywh for SAM3 API
                x1n = max(0.0, min(1.0, box[0] / W))
                y1n = max(0.0, min(1.0, box[1] / H))
                wn  = max(0.0, min(1.0, (box[2] - box[0]) / W))
                hn  = max(0.0, min(1.0, (box[3] - box[1]) / H))
                box_tensor = torch.tensor([[x1n, y1n, wn, hn]], dtype=torch.float32)
                box_labels = torch.tensor([1], dtype=torch.int32)
                self.video_predictor.handle_request(
                    request=dict(
                        type="add_prompt",
                        session_id=session_id,
                        frame_index=start_f,
                        bounding_boxes=box_tensor,
                        bounding_box_labels=box_labels,
                        obj_id=obj_id,
                    )
                )
                    
            if points and labels:
                pts_rel = abs_to_rel_coords(np.array(points), W, H, coord_type="point")
                pts_tensor = torch.tensor(pts_rel, dtype=torch.float32)
                lbl_tensor = torch.tensor(labels, dtype=torch.int32)
                self.video_predictor.handle_request(
                    request=dict(
                        type="add_prompt",
                        session_id=session_id,
                        frame_index=start_f,
                        points=pts_tensor,
                        point_labels=lbl_tensor,
                        obj_id=obj_id,
                    )
                )

            # Propagate in chunks to avoid VRAM exhaustion on large videos
            CHUNK_SIZE = 50  # process 50 frames at a time
            num_frames_to_track = end_f - start_f + 1
            current_start = start_f
            frames_done = 0

            while frames_done < num_frames_to_track:
                chunk_size = min(CHUNK_SIZE, num_frames_to_track - frames_done)

                for response in self.video_predictor.handle_stream_request(
                    request=dict(
                        type="propagate_in_video",
                        session_id=session_id,
                        start_frame_index=current_start,
                        max_frame_num_to_track=chunk_size,
                        propagation_direction="forward",
                    )
                ):
                    out = response["outputs"]

                    frame_polygons = []
                    frame_boxes = []
                    H_video, W_video = None, None

                    # SAM3 returns: {"out_obj_ids", "out_binary_masks" [N,H,W], "out_boxes_xywh" [N,4] normalized}
                    out_binary_masks = out.get("out_binary_masks", None)
                    out_boxes_xywh = out.get("out_boxes_xywh", None)

                    if out_binary_masks is not None and len(out_binary_masks) > 0:
                        H_video, W_video = out_binary_masks[0].shape[:2]

                        for mask in out_binary_masks:
                            mask_uint8 = mask.astype(bool).astype(np.uint8) * 255
                            contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                            if contours:
                                largest_contour = max(contours, key=cv2.contourArea)
                                polygon = [(float(pt[0][0]), float(pt[0][1])) for pt in largest_contour]
                                frame_polygons.append(polygon)
                            else:
                                frame_polygons.append(None)

                    if out_boxes_xywh is not None and len(out_boxes_xywh) > 0 and H_video and W_video:
                        for b in out_boxes_xywh:
                            # b is [x, y, w, h] normalized; convert to absolute [x1, y1, x2, y2]
                            x1 = int(b[0] * W_video)
                            y1 = int(b[1] * H_video)
                            x2 = int((b[0] + b[2]) * W_video)
                            y2 = int((b[1] + b[3]) * H_video)
                            frame_boxes.append([x1, y1, x2, y2])

                    yield True, {"polygons": frame_polygons, "boxes": frame_boxes}

                frames_done += chunk_size
                current_start += chunk_size
                # Free unused CUDA memory between chunks
                torch.cuda.empty_cache()
                
            self.video_predictor.handle_request(
                request=dict(
                    type="close_session",
                    session_id=session_id,
                )
            )

        except Exception as e:
            import traceback
            traceback.print_exc()
            yield False, f"SAM3 Native tracking failed: {str(e)}"

    def clear_session(self):
        """
        Closes any active session and resets state variables.
        """
        if self.video_predictor and self.current_session_id is not None:
            try:
                self.video_predictor.handle_request({"type": "close_session", "session_id": self.current_session_id})
            except Exception as e:
                print(f"[SAM3 Native] Failed to close session on clear: {e}")
        self.current_session_id = None
        self.current_img_hash = None
        self.current_box = None
        self.current_text_prompt = None
