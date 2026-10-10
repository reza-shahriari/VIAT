import os
import shutil
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

# SAM2-style point labels for the top-left / bottom-right corners of a box prompt.
BOX_TOP_LEFT_LABEL = 2
BOX_BOTTOM_RIGHT_LABEL = 3

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp")


def _patch_sam3_multiplex_output():
    """
    Works around an upstream SAM3.1 bug that silently drops every click/box (instance) prompt.

    `Sam3MultiplexTracking._build_sam2_output` returns `{}` whenever the frame has no entry in
    `cached_frame_outputs` -- discarding the refined mask it was just handed. That cache is only
    filled by a prior *text/detector* propagation (upstream's notebook always runs one first), so
    in a fresh session both `add_prompt` with points and the subsequent partial propagation return
    empty outputs on every frame even though the tracker produced a good mask.
    """
    from sam3.model.sam3_multiplex_tracking import Sam3MultiplexTracking

    if getattr(Sam3MultiplexTracking._build_sam2_output, "_viat_patched", False):
        return

    def _build_sam2_output(self, inference_state, frame_idx, refined_obj_id_to_mask=None):
        obj_id_to_mask = dict(inference_state["cached_frame_outputs"].get(frame_idx, {}))
        if refined_obj_id_to_mask:
            obj_id_to_mask.update(refined_obj_id_to_mask)
        return obj_id_to_mask

    _build_sam2_output._viat_patched = True
    Sam3MultiplexTracking._build_sam2_output = _build_sam2_output


def _get_resource_size(resource_path):
    """Returns (width, height) of a video file or a folder of frames, or None if unreadable."""
    if os.path.isdir(resource_path):
        for name in sorted(os.listdir(resource_path)):
            if name.lower().endswith(IMAGE_EXTS):
                img = cv2.imread(os.path.join(resource_path, name))
                if img is not None:
                    return img.shape[1], img.shape[0]
        return None
    cap = cv2.VideoCapture(resource_path)
    try:
        ret, frame = cap.read()
        return (frame.shape[1], frame.shape[0]) if ret else None
    finally:
        cap.release()


def _materialize_frames(resource, first_f, last_f):
    """
    Writes frames [first_f, last_f] of `resource` (a video path, or a list of image paths) into a
    new temp folder as 00000.<ext>, 00001.<ext>, ... -- the numeric naming SAM3's folder loader
    requires. Image files are symlinked rather than copied where the OS allows it.
    Returns the folder path, or None if no frames could be read. The caller must delete it.
    """
    frames_dir = tempfile.mkdtemp(prefix="viat_sam3_track_")
    count = 0
    if isinstance(resource, (list, tuple)):
        for local_idx, src in enumerate(resource[first_f:last_f + 1]):
            dst = os.path.join(frames_dir, f"{local_idx:05d}{os.path.splitext(src)[1].lower()}")
            if not dst.endswith(IMAGE_EXTS):
                img = cv2.imread(src)
                if img is None:
                    break
                cv2.imwrite(os.path.splitext(dst)[0] + ".jpg", img)
            else:
                try:
                    os.symlink(os.path.abspath(src), dst)
                except OSError:
                    shutil.copyfile(src, dst)
            count += 1
    else:
        cap = cv2.VideoCapture(resource)
        try:
            cap.set(cv2.CAP_PROP_POS_FRAMES, first_f)
            for local_idx in range(last_f - first_f + 1):
                ret, frame = cap.read()
                if not ret:
                    break
                cv2.imwrite(os.path.join(frames_dir, f"{local_idx:05d}.jpg"), frame,
                            [cv2.IMWRITE_JPEG_QUALITY, 95])
                count += 1
        finally:
            cap.release()
    if count == 0:
        shutil.rmtree(frames_dir, ignore_errors=True)
        return None
    return frames_dir


def _mask_to_polygon(mask):
    """Largest external contour of a binary mask as [(x, y), ...], or None if the mask is empty."""
    mask_uint8 = np.asarray(mask, dtype=bool).astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    largest_contour = max(contours, key=cv2.contourArea)
    return [(float(pt[0][0]), float(pt[0][1])) for pt in largest_contour]

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

    def _get_device(self):
        """
        Returns the torch device the underlying SAM3 model lives on.
        The vendored SAM3 predictor unconditionally calls `.cuda()` on its model, but
        prompt tensors we build ourselves default to CPU, which crashes the Triton NMS
        kernel with a device mismatch. Query the real device instead of assuming.
        """
        import torch
        try:
            return next(self.video_predictor.model.parameters()).device
        except Exception:
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _build_prompt_request(self, session_id, frame_idx, width, height, points=None, labels=None, box=None, text_prompt=None):
        """
        Builds the `add_prompt` request for a session, or returns None if there is no usable prompt.

        Clicks and boxes go through SAM3's instance (SAM2-style) path so they segment *that* object:
        the box is encoded as corner points with labels 2/3, exactly like SAM2's prompt encoder.
        (Sending it as `bounding_boxes` instead makes it a detector *exemplar* -- "find things that
        look like this" -- which returns unrelated/partial masks and crashes on propagation.)
        A text prompt is only used on its own, through the detector path.
        """
        import torch

        device = self._get_device()
        rel_points, point_labels = [], []
        if box is not None:
            x1, y1, x2, y2 = box
            rel_points += [[x1 / width, y1 / height], [x2 / width, y2 / height]]
            point_labels += [BOX_TOP_LEFT_LABEL, BOX_BOTTOM_RIGHT_LABEL]
        if points and labels:
            rel_points += [[x / width, y / height] for x, y in points]
            point_labels += [int(l) for l in labels]

        if rel_points:
            rel_points = [[min(1.0, max(0.0, x)), min(1.0, max(0.0, y))] for x, y in rel_points]
            print(f"[SAM3 Native] Instance prompt: {len(rel_points)} points, labels={point_labels}")
            return {
                "type": "add_prompt",
                "session_id": session_id,
                "frame_index": frame_idx,
                "obj_id": 1,
                "points": torch.tensor(rel_points, dtype=torch.float32, device=device),
                "point_labels": torch.tensor(point_labels, dtype=torch.int32, device=device),
            }
        if text_prompt:
            print(f"[SAM3 Native] Text prompt: {text_prompt}")
            return {
                "type": "add_prompt",
                "session_id": session_id,
                "frame_index": frame_idx,
                "text": text_prompt,
                "output_prob_thresh": 0.1,
            }
        return None

    def load_model(self, model_type="sam3.1_l.pt"):
        """
        Loads the SAM3 model.
        Returns (success_bool, message_string).
        """
        if not check_sam3():
            return False, "SAM3 native package is not installed."

        if self.current_model_type == model_type and self.video_predictor is not None:
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
                    _patch_sam3_multiplex_output()
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
        
        # Other VIAT managers (e.g. zero_shot_manager's HF `device_map="auto"` loads) can
        # reset torch's ambient "current device" out from under us at any point. SAM3's
        # Triton kernels launch on whatever the current device is *at call time*, not on
        # the device the model weights actually live on, so a stale current device here
        # makes Triton try to access pointers from the wrong GPU and crash with
        # "Pointer argument ... cannot be accessed from Triton". Re-pin every call.
        torch.cuda.set_device(self._get_device())

        # Hash image to see if it's new
        img_bytes = image_array.tobytes()
        img_hash = hashlib.md5(img_bytes).hexdigest()

        # Check if the box or text prompt has changed (indicating a new object/prompt query)
        box_tuple = tuple(box) if box is not None else None
        prompt_changed = (box_tuple != self.current_box) or (text_prompt != self.current_text_prompt)

        # Always close any previous session so prompts from previous objects/queries never bleed into this one
        if self.current_session_id is not None:
            try:
                self.video_predictor.handle_request({"type": "close_session", "session_id": self.current_session_id})
            except Exception as e:
                print(f"[SAM3 Native] Failed to close session: {e}")
            self.current_session_id = None

        # Save image if hash changed or file does not exist
        img_path = os.path.join(self.temp_dir, "00000.jpg")
        if img_hash != self.current_img_hash or not os.path.exists(img_path):
            cv2.imwrite(img_path, cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR))
            self.current_img_hash = img_hash

        # This session only ever holds a single image, so there's no VRAM pressure to
        # justify offloading frames to CPU -- and doing so anyway leaves the box/NMS
        # grounding path operating on a mix of CPU and CUDA tensors, which crashes the
        # Triton NMS kernel with "Pointer argument ... cannot be accessed from Triton".
        res = self.video_predictor.handle_request({"type": "start_session", "resource_path": self.temp_dir, "offload_video_to_cpu": False})
        self.current_session_id = res["session_id"]
        self.current_box = box_tuple
        self.current_text_prompt = text_prompt

        IMG_HEIGHT, IMG_WIDTH = image_array.shape[:2]

        prompt_req = self._build_prompt_request(
            self.current_session_id, 0, IMG_WIDTH, IMG_HEIGHT,
            points=points, labels=labels, box=box, text_prompt=text_prompt)
        if prompt_req is None:
            self.clear_session()
            return None

        polygon_pts = None
        try:
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

            if isinstance(out_binary_masks, torch.Tensor):
                out_binary_masks = out_binary_masks.detach().cpu().numpy()

            # Instance (click/box) prompts are always obj_id=1. Text prompts return detector ids
            # that can also include 1 by coincidence, so take the most confident detection instead.
            mask = None
            if "points" in prompt_req and out_obj_ids is not None:
                if isinstance(out_obj_ids, torch.Tensor):
                    out_obj_ids = out_obj_ids.detach().cpu().numpy()
                for i, oid in enumerate(out_obj_ids):
                    if int(oid) == 1:
                        mask = out_binary_masks[i]
                        break
            if mask is None:
                out_probs = outputs.get("out_probs", None)
                best = int(np.argmax(np.asarray(out_probs))) if out_probs is not None and len(out_probs) == len(out_binary_masks) else 0
                mask = out_binary_masks[best]

            if mask is None:
                return None

            if isinstance(mask, torch.Tensor):
                mask = mask.detach().cpu().numpy()

            polygon_pts = _mask_to_polygon(mask)
            if polygon_pts is None:
                print("[SAM3 Native] mask is empty")
        finally:
            # Clean up session immediately so no state or prompt history lingers for future queries
            if self.current_session_id is not None:
                try:
                    self.video_predictor.handle_request({"type": "close_session", "session_id": self.current_session_id})
                except Exception:
                    pass
                self.current_session_id = None
            
        return polygon_pts

    def predict_text_in_box(self, image_array, box, text_prompt, pad_ratio=0.1):
        """
        Finds `text_prompt` *inside* `box` (e.g. the helmet within a person box) and returns its
        polygon in full-image coordinates, or None if nothing matching was found there.

        This is what auto-label's Helper classes need. It is deliberately separate from
        predict_mask_from_prompt(box=..., text_prompt=...), where the box selects the object itself
        and the text is ignored. Running the text prompt on a crop both restricts the search to the
        box and gives SAM3 more pixels for small sub-parts.
        """
        img_h, img_w = image_array.shape[:2]
        x1, y1, x2, y2 = box
        pad_x, pad_y = (x2 - x1) * pad_ratio, (y2 - y1) * pad_ratio
        cx1, cy1 = max(0, int(x1 - pad_x)), max(0, int(y1 - pad_y))
        cx2, cy2 = min(img_w, int(x2 + pad_x)), min(img_h, int(y2 + pad_y))
        if cx2 - cx1 < 8 or cy2 - cy1 < 8:
            return None

        crop = np.ascontiguousarray(image_array[cy1:cy2, cx1:cx2])
        polygon = self.predict_mask_from_prompt(crop, text_prompt=text_prompt)
        if not polygon:
            return None
        return [(x + cx1, y + cy1) for x, y in polygon]

    def track_video_from_prompt(self, resource_path, start_f, end_f, points=None, labels=None, box=None, text_prompt=None):
        """
        Tracks an object in video using SAM3.
        Yields (success_bool, {"polygons": [...], "boxes": [...]}) for each frame.
        """
        import torch

        if not self.video_predictor:
            yield False, "SAM3 Video Predictor not loaded."
            return

        # See the comment in predict_mask_from_prompt: re-pin the ambient CUDA device
        # before every call, since other VIAT managers can silently reset it.
        torch.cuda.set_device(self._get_device())

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

        session_id = None
        frames_dir = None
        try:
            # SAM3 decodes *every* frame of its resource into RAM up front (fp16 1008x1008, ~6 MB
            # per frame), so handing it the whole video costs RAM for the entire video even when
            # tracking a short range. Give it only the frames being tracked, re-indexed from 0.
            first_f = min(start_f, end_f)
            frames_dir = _materialize_frames(resource_path, first_f, max(start_f, end_f))
            if frames_dir is None:
                yield False, f"SAM3 Native tracking failed: could not read frames from {resource_path}"
                return
            resource_path, start_f, end_f = frames_dir, start_f - first_f, end_f - first_f

            # Start session with CPU offloading to reduce VRAM usage
            response = self.video_predictor.handle_request(
                request=dict(
                    type="start_session",
                    resource_path=resource_path,
                    offload_video_to_cpu=True,
                )
            )
            session_id = response["session_id"]

            size = _get_resource_size(resource_path)
            if size is None:
                yield False, f"SAM3 Native tracking failed: could not read frames from {resource_path}"
                return
            W, H = size

            prompt_req = self._build_prompt_request(
                session_id, start_f, W, H,
                points=points, labels=labels, box=box, text_prompt=text_prompt)
            if prompt_req is None:
                yield False, "No prompt provided for SAM3 tracking."
                return

            prompt_res = self.video_predictor.handle_request(request=prompt_req)

            is_backward = start_f > end_f
            step = -1 if is_backward else 1
            # The caller assigns each yielded result to the next frame in order, so this must yield
            # exactly one result per frame from start_f to end_f. SAM3's propagation doesn't make
            # that easy on its own: forward runs re-yield the prompted frame, backward runs skip it,
            # and both can overshoot `max_frame_num_to_track` by one. So take the prompted frame
            # from add_prompt, then only accept propagated frames that are the next expected one.
            yield True, self._outputs_to_frame_result(prompt_res.get("outputs"))
            next_f = start_f + step

            def _in_range(f):
                return (f - end_f) * step <= 0

            # SAM3.1's batched grounding prefetches detector/backbone features in windows that
            # assume a forward run starts at frame 0; starting mid-video leaves the next frame's
            # features empty and text-prompt tracking crashes on the 2nd frame. Upstream already
            # disables it for add_prompt; do the same for that one propagation case.
            model = getattr(self.video_predictor, "model", None)
            orig_batched_grounding = getattr(model, "use_batched_grounding", None)
            if orig_batched_grounding and not is_backward and start_f > 0:
                model.use_batched_grounding = False
            try:
                if _in_range(next_f):
                    for response in self.video_predictor.handle_stream_request(
                        request=dict(
                            type="propagate_in_video",
                            session_id=session_id,
                            start_frame_index=start_f,
                            max_frame_num_to_track=abs(end_f - start_f) + 1,
                            propagation_direction="backward" if is_backward else "forward",
                        )
                    ):
                        if response["frame_index"] != next_f:
                            continue
                        yield True, self._outputs_to_frame_result(response["outputs"])
                        next_f += step
                        if not _in_range(next_f):
                            break
            finally:
                if orig_batched_grounding is not None:
                    model.use_batched_grounding = orig_batched_grounding

            # Pad any frames propagation never reached so the caller's frame counter stays aligned.
            while _in_range(next_f):
                yield True, {"polygons": [], "boxes": []}
                next_f += step

        except Exception as e:
            import traceback
            traceback.print_exc()
            yield False, f"SAM3 Native tracking failed: {str(e)}"
        finally:
            # Always close the session to free VRAM, even if prompting/propagation throws
            if session_id is not None:
                try:
                    self.video_predictor.handle_request(
                        request=dict(type="close_session", session_id=session_id)
                    )
                except Exception as close_e:
                    print(f"[SAM3 Native] Failed to close tracking session: {close_e}")
            if frames_dir is not None:
                shutil.rmtree(frames_dir, ignore_errors=True)

    @staticmethod
    def _outputs_to_frame_result(outputs):
        """Converts one frame of SAM3 outputs to {"polygons": [...], "boxes": [[x1, y1, x2, y2], ...]}."""
        import torch

        frame_polygons, frame_boxes = [], []
        if not outputs:
            return {"polygons": frame_polygons, "boxes": frame_boxes}

        # SAM3 returns: {"out_obj_ids", "out_binary_masks" [N,H,W], "out_boxes_xywh" [N,4] normalized}
        out_binary_masks = outputs.get("out_binary_masks", None)
        out_boxes_xywh = outputs.get("out_boxes_xywh", None)
        if out_binary_masks is None or len(out_binary_masks) == 0:
            return {"polygons": frame_polygons, "boxes": frame_boxes}
        if isinstance(out_binary_masks, torch.Tensor):
            out_binary_masks = out_binary_masks.detach().cpu().numpy()
        if isinstance(out_boxes_xywh, torch.Tensor):
            out_boxes_xywh = out_boxes_xywh.detach().cpu().numpy()

        H_video, W_video = out_binary_masks[0].shape[:2]
        for i, mask in enumerate(out_binary_masks):
            polygon = _mask_to_polygon(mask)
            if polygon is None:
                continue
            # Keep polygons and boxes index-aligned: the caller pairs boxes[0] with polygons[0]
            frame_polygons.append(polygon)
            b = out_boxes_xywh[i]
            # b is [x, y, w, h] normalized; convert to absolute [x1, y1, x2, y2]
            frame_boxes.append([
                int(b[0] * W_video),
                int(b[1] * H_video),
                int((b[0] + b[2]) * W_video),
                int((b[1] + b[3]) * H_video),
            ])
        return {"polygons": frame_polygons, "boxes": frame_boxes}

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
