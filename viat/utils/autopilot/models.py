"""Detector backends for Autopilot (local weights only, own model instances).

Every backend returns ``[(box_xyxy, score, label_text), ...]`` for a BGR image.
``ref_score`` is the raw score at which that model is typically "confident"; it is
how scores from different models are put on one scale before fusion (YOLO-World
reports a real tank at 0.25 where SAM 3 would say 0.8).
"""
import cv2

try:
    import torch
except ImportError:
    torch = None

from ..zero_shot_manager import ZeroShotManager
from .weights import resolve_hf, resolve_file, missing_message, CHECKPOINTS

try:
    from PIL import Image
except ImportError:
    Image = None


def _device():
    return 'cuda' if torch is not None and torch.cuda.is_available() else 'cpu'


class Backend:
    ref_score = 0.3
    constant = False  # reports a fixed score: may vote, never start a track

    def __init__(self, key, log=print):
        self.key = key
        self.log = log
        self.prompts = []

    def load(self):
        raise NotImplementedError

    def set_prompts(self, prompts):
        self.prompts = list(prompts)

    def predict(self, image):
        raise NotImplementedError


class UltralyticsBackend(Backend):
    """YOLO-World / YOLOE / YOLO / SAM 3 text through the existing wrappers."""

    def __init__(self, key, log=print):
        super().__init__(key, log)
        self.ref_score = 0.5 if 'sam3' in key.lower() else 0.3
        self.mgr = ZeroShotManager()

    def load(self):
        if resolve_file(self.key) is None:  # the wrapper would download it otherwise
            return False, missing_message(self.key)
        ok, msg = self.mgr.load_model(self.key)
        return ok, msg

    def set_prompts(self, prompts):
        super().set_prompts(prompts)
        self.mgr.set_classes(list(prompts))
        # YOLO-World/YOLOE wrappers park the model on the CPU while setting classes.
        model = getattr(self.mgr.detector, 'model', None)
        if torch is not None and torch.cuda.is_available() and model is not None and hasattr(model, 'to'):
            try:
                model.to('cuda')
            except Exception as exc:
                self.log(f"[Autopilot] could not move {self.key} to GPU: {exc}")

    def predict(self, image):
        dets = self.mgr.predict(image, score_threshold=0.0)
        return [(d['box'], float(d.get('score', 0.0)), str(d.get('class_name', ''))) for d in dets]


class ConstantScoreBackend(UltralyticsBackend):
    """Florence-2 / LocateAnything: boxes without a usable confidence."""
    constant = True
    ref_score = 1.0

    def load(self):
        if resolve_hf(self.key) is None:
            return False, missing_message(self.key)
        ok, msg = self.mgr.load_model(self.key, checkpoints_dir=CHECKPOINTS)
        return ok, msg

    def set_prompts(self, prompts):
        Backend.set_prompts(self, prompts)
        self.mgr.set_classes(list(prompts))


class GroundingDinoBackend(Backend):
    ref_score = 0.35

    def load(self):
        path = resolve_hf(self.key)
        if path is None:
            return False, missing_message(self.key)
        try:
            from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
            self.processor = AutoProcessor.from_pretrained(path, local_files_only=True)
            self.model = AutoModelForZeroShotObjectDetection.from_pretrained(path, local_files_only=True)
            self.model = self.model.to(_device()).eval()
            return True, 'ok'
        except Exception as exc:
            return False, f"Grounding DINO failed to load from {path}: {exc}"

    def predict(self, image):
        if not self.prompts:
            return []
        pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        text = ' . '.join(p.lower() for p in self.prompts) + ' .'
        inputs = self.processor(images=pil, text=text, return_tensors='pt').to(_device())
        with torch.no_grad():
            outputs = self.model(**inputs)
        proc = self.processor.post_process_grounded_object_detection
        try:
            res = proc(outputs, inputs['input_ids'], threshold=0.05, text_threshold=0.05,
                       target_sizes=[pil.size[::-1]])[0]
        except TypeError:  # older transformers naming
            res = proc(outputs, inputs['input_ids'], box_threshold=0.05, text_threshold=0.05,
                       target_sizes=[pil.size[::-1]])[0]
        labels = res.get('text_labels', res.get('labels', []))
        out = []
        for box, score, label in zip(res['boxes'].cpu().numpy(), res['scores'].cpu().numpy(), labels):
            out.append(([float(v) for v in box], float(score), str(label)))
        return out


class Owlv2Backend(Backend):
    ref_score = 0.15

    def load(self):
        path = resolve_hf(self.key)
        if path is None:
            return False, missing_message(self.key)
        try:
            from transformers import Owlv2Processor, Owlv2ForObjectDetection
            self.processor = Owlv2Processor.from_pretrained(path, local_files_only=True)
            self.model = Owlv2ForObjectDetection.from_pretrained(path, local_files_only=True).to(_device()).eval()
            return True, 'ok'
        except Exception as exc:
            return False, f"OWLv2 failed to load from {path}: {exc}"

    def predict(self, image):
        if not self.prompts:
            return []
        h, w = image.shape[:2]
        pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        queries = [[f"a photo of a {p}" for p in self.prompts]]
        inputs = self.processor(text=queries, images=pil, return_tensors='pt').to(_device())
        with torch.no_grad():
            outputs = self.model(**inputs)
        side = max(h, w)  # OWLv2 pads the image to a square before resizing
        post = getattr(self.processor, 'post_process_grounded_object_detection', None) or \
            self.processor.post_process_object_detection
        res = post(outputs=outputs, threshold=0.02, target_sizes=torch.tensor([[side, side]]))[0]
        out = []
        for box, score, label in zip(res['boxes'].cpu().numpy(), res['scores'].cpu().numpy(),
                                     res['labels'].cpu().numpy()):
            x1, y1, x2, y2 = [float(v) for v in box]
            x1, x2 = max(0.0, x1), min(float(w), x2)
            y1, y2 = max(0.0, y1), min(float(h), y2)
            if x2 > x1 and y2 > y1:
                out.append(([x1, y1, x2, y2], float(score), self.prompts[int(label)]))
        return out


def make_backend(key, log=print):
    k = key.lower()
    if 'grounding-dino' in k or 'groundingdino' in k:
        return GroundingDinoBackend(key, log)
    if 'owlv2' in k or 'owlvit' in k:
        return Owlv2Backend(key, log)
    if 'florence' in k or 'locateanything' in k:
        return ConstantScoreBackend(key, log)
    return UltralyticsBackend(key, log)


def match_prompt(label, prompt_class):
    """Map a returned label to a class index; tolerant to Grounding DINO phrase merging."""
    key = label.strip().lower()
    if key in prompt_class:
        return prompt_class[key]
    best, best_len = None, 0
    for p, c in prompt_class.items():
        if (p in key or key in p) and len(p) > best_len and key:
            best, best_len = c, len(p)
    return best

