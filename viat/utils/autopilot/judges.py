"""Second-opinion judges for Autopilot tracks.

Detectors propose; judges decide whether a track is a real object of its class. Each
judge looks at a few crops sampled along the track (one decision per track, not
per frame) and returns p_real in [0, 1]:

  clip      CLIP crop classifier: the track's class against the other classes and
            decoys ("part of a vehicle", "a shadow", ...). Cheap, good at class
            confusion and part-vs-whole, weak on tiny crops.
  sam_zoom  zoom-in re-detection: crop around the box with context, let SAM 3 look
            again at higher effective resolution with the class prompt.
  vlm       local vision-language model (Qwen-VL): multiple-choice question about
            the boxed object, scored from the answer-letter probabilities.
  learned   logistic regression trained on your review decisions / labeled videos,
            using the track features plus the other judges' answers.

They run as a cascade (cheap first; the VLM only where the others are unsure) and
their combined answer can auto-reject a track (kept visible in the review panel).
"""
import time

import cv2
import numpy as np

try:
    import torch
except ImportError:
    torch = None

try:
    from PIL import Image
except ImportError:
    Image = None

from ..zero_shot_manager import Sam3TextDetector
from .geometry import iou_matrix
from .learned import LearnedJudge
from .weights import resolve_hf, resolve_file, missing_message, CHECKPOINTS

# Judges trusted to auto-reject. The learned judge ranks tracks for review only: on 8 held-out videos it
# had the best AUC (0.974) but lost 240 real boxes at t=0.3, while clip+sam_zoom lost none.
STRONG = ('sam_zoom', 'vlm')
WEIGHTS = {'clip': 1.0, 'sam_zoom': 1.5, 'vlm': 2.0}


def _device():
    return 'cuda' if torch is not None and torch.cuda.is_available() else 'cpu'


# ----------------------------------------------------------------------------- crops
def expand_box(box, factor, min_side, width, height):
    x1, y1, x2, y2 = [float(v) for v in box]
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    w, h = max((x2 - x1) * factor, min_side), max((y2 - y1) * factor, min_side)
    ex1, ey1 = int(max(0, cx - w / 2)), int(max(0, cy - h / 2))
    ex2, ey2 = int(min(width, cx + w / 2)), int(min(height, cy + h / 2))
    return ex1, ey1, ex2, ey2


class TrackCrops:
    """Crops of one track: tight zoom crops, context crops, and boxed full-frame thumbnails."""

    def __init__(self):
        self.zoom, self.context, self.context_box, self.full = [], [], [], []


def sample_frames_for(track, k):
    rows = [r for r in track['boxes'] if r[6] == 'd'] or track['boxes']
    if len(rows) <= k:
        return rows
    idx = np.linspace(0, len(rows) - 1, k).round().astype(int)
    return [rows[i] for i in idx]


def collect_crops(source, tracks, k, want_full, is_cancelled):
    """Read every needed frame once (sorted) and cut the crops for every track."""
    requests = {}
    for t in tracks:
        for r in sample_frames_for(t, k):
            requests.setdefault(int(r[0]), []).append((t['id'], r[1:5]))
    crops = {t['id']: TrackCrops() for t in tracks}
    if not requests:
        return crops
    frames = sorted(requests)
    for f, frame in _read_frames(source, frames):
        if is_cancelled():
            break
        H, W = frame.shape[:2]
        for tid, box in requests[f]:
            c = crops[tid]
            x1, y1, x2, y2 = expand_box(box, 1.25, 16, W, H)
            if x2 - x1 < 2 or y2 - y1 < 2:
                continue
            c.zoom.append(frame[y1:y2, x1:x2].copy())
            cx1, cy1, cx2, cy2 = expand_box(box, 3.0, 160, W, H)
            c.context.append(frame[cy1:cy2, cx1:cx2].copy())
            c.context_box.append([box[0] - cx1, box[1] - cy1, box[2] - cx1, box[3] - cy1])
            if want_full:
                s = 768.0 / max(H, W)
                thumb = cv2.resize(frame, (int(W * s), int(H * s))) if s < 1 else frame.copy()
                b = [int(v * min(s, 1.0)) for v in box]
                cv2.rectangle(thumb, (b[0], b[1]), (b[2], b[3]), (0, 0, 255), max(2, int(thumb.shape[1] / 300)))
                c.full.append(cv2.imencode('.jpg', thumb, [cv2.IMWRITE_JPEG_QUALITY, 90])[1])
    return crops


def _read_frames(source, frames):
    if source.video_path:
        cap = cv2.VideoCapture(source.video_path)
        try:
            pos = -1
            for f in frames:
                if f != pos + 1 and not (pos < f <= pos + 30):
                    cap.set(cv2.CAP_PROP_POS_FRAMES, f)
                    pos = f - 1
                frame = None
                while pos < f:
                    ok, frame = cap.read()
                    pos += 1
                    if not ok:
                        frame = None
                        break
                if frame is not None:
                    yield f, frame
        finally:
            cap.release()
    else:
        for f in frames:
            frame = cv2.imread(source.image_files[f]) if f < len(source.image_files) else None
            if frame is not None:
                yield f, frame


# ---------------------------------------------------------------------------- judges
class ClipJudge:
    name = 'clip'

    def __init__(self, cfg, names, log=print):
        self.cfg, self.names, self.log = cfg, names, log

    def load(self):
        path = resolve_hf(self.cfg.clip_model)
        if path is None:
            return False, missing_message(self.cfg.clip_model)
        from transformers import CLIPModel, CLIPProcessor
        self.model = CLIPModel.from_pretrained(path, local_files_only=True).to(_device()).eval()
        self.processor = CLIPProcessor.from_pretrained(path, local_files_only=True)
        texts, owner = [], []
        for ci, c in enumerate(self.cfg.classes):
            for p in (c.get('prompts') or [c['name']]):
                texts.append(f"a photo of a {p}")
                owner.append(ci)
        for d in self.cfg.decoys:
            texts.append(f"a photo of {d}")
            owner.append(-1)
        self.owner = np.array(owner)
        with torch.no_grad():
            tok = self.processor(text=texts, return_tensors='pt', padding=True).to(_device())
            emb = self.model.get_text_features(**tok)
            emb = getattr(emb, 'pooler_output', emb)
            self.text_emb = torch.nn.functional.normalize(emb, dim=-1)
        return True, 'ok'

    def image_embeddings(self, images):
        pil = [Image.fromarray(cv2.cvtColor(im, cv2.COLOR_BGR2RGB)) for im in images]
        with torch.no_grad():
            inp = self.processor(images=pil, return_tensors='pt').to(_device())
            emb = self.model.get_image_features(**inp)
            emb = getattr(emb, 'pooler_output', emb)
            return torch.nn.functional.normalize(emb, dim=-1)

    def judge(self, track, crops):
        if not crops.zoom:
            return None, {}
        img = self.image_embeddings(crops.zoom)
        with torch.no_grad():
            logits = self.model.logit_scale.exp() * img @ self.text_emb.T
        probs = logits.softmax(-1).float().cpu().numpy()
        if track['class'] not in self.names:
            return None, {}
        ci = self.names.index(track['class'])
        p_class = probs[:, self.owner == ci].sum(1)
        per_class = np.array([probs[:, self.owner == c].sum(1).mean() for c in range(len(self.names))])
        decoy = probs[:, self.owner == -1].mean(0)
        info = {'suggest': self.names[int(per_class.argmax())], 'suggest_p': float(per_class.max())}
        if len(decoy):
            j = int(decoy.argmax())
            info['decoy'] = self.cfg.decoys[j]
            info['decoy_p'] = float(decoy[j])
        return float(np.mean(p_class)), info


class SamZoomJudge:
    name = 'sam_zoom'

    def __init__(self, cfg, names, log=print):
        self.cfg, self.names, self.log = cfg, names, log

    def load(self):
        if resolve_file(self.cfg.sam_judge_model) is None:
            return False, missing_message(self.cfg.sam_judge_model)
        self.det = Sam3TextDetector()
        return self.det.load_model(self.cfg.sam_judge_model, CHECKPOINTS)

    def judge(self, track, crops):
        if not crops.context:
            return None, {}
        cls = next((c for c in self.cfg.classes if c['name'] == track['class']), {'name': track['class']})
        self.det.set_classes(list(cls.get('prompts') or [cls['name']]))
        scores = []
        for img, box in zip(crops.context, crops.context_box):
            dets = self.det.predict(img)
            best = 0.0
            if dets:
                ious = iou_matrix([box], [d['box'] for d in dets])[0]
                for d, iou in zip(dets, ious):
                    if iou >= 0.3:
                        best = max(best, float(d['score']))
            scores.append(best)
        # SAM 3 is calibrated around 0.5: map 0.5 -> 0.75 so "found it" reads as likely real
        return float(np.clip(np.mean(scores) * 1.5, 0, 1)), {'raw': float(np.mean(scores))}


class VlmJudge:
    name = 'vlm'
    LETTERS = 'ABCDEFGHIJ'

    def __init__(self, cfg, names, log=print):
        self.cfg, self.names, self.log = cfg, names, log

    def load(self):
        path = resolve_hf(self.cfg.vlm_model)
        if path is None:
            return False, missing_message(self.cfg.vlm_model)
        from transformers import AutoProcessor, AutoModelForImageTextToText
        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        self.processor = AutoProcessor.from_pretrained(path, local_files_only=True)
        self.model = AutoModelForImageTextToText.from_pretrained(
            path, local_files_only=True, torch_dtype=dtype).to(_device()).eval()
        tok = self.processor.tokenizer
        self.letter_ids = [tok.encode(L, add_special_tokens=False)[0] for L in self.LETTERS]
        return True, 'ok'

    def _options(self, track):
        others = [n for n in self.names if n != track['class']][:6]
        opts = [track['class']] + others + ['only a part of a larger object', 'background / nothing / a shadow']
        return opts

    def judge(self, track, crops):
        n = min(len(crops.zoom), len(crops.full), 3)
        if n == 0:
            return None, {}
        opts = self._options(track)
        question = ("The first image is a video frame with a red box. The second image is a close-up of the "
                    "red box. What is inside the red box?\n" +
                    '\n'.join(f"{self.LETTERS[i]}) {o}" for i, o in enumerate(opts)) +
                    "\nAnswer with one letter.")
        probs = []
        for i in range(n):
            full = cv2.imdecode(crops.full[i], cv2.IMREAD_COLOR)
            zoom = crops.zoom[i]
            s = max(1.0, 336.0 / max(1, min(zoom.shape[:2])))
            zoom = cv2.resize(zoom, (int(zoom.shape[1] * s), int(zoom.shape[0] * s)), interpolation=cv2.INTER_CUBIC)
            images = [Image.fromarray(cv2.cvtColor(im, cv2.COLOR_BGR2RGB)) for im in (full, zoom)]
            messages = [{'role': 'user', 'content': [{'type': 'image', 'image': images[0]},
                                                     {'type': 'image', 'image': images[1]},
                                                     {'type': 'text', 'text': question}]}]
            inputs = self.processor.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                                                        return_dict=True, return_tensors='pt').to(_device())
            with torch.no_grad():
                logits = self.model(**inputs).logits[0, -1]
            letter_logits = logits[self.letter_ids[:len(opts)]].float()
            probs.append(letter_logits.softmax(-1).cpu().numpy())
        p = np.mean(probs, 0)
        best = int(p.argmax())
        return float(p[0]), {'answer': opts[best], 'answer_p': float(p[best])}


JUDGES = {'clip': ClipJudge, 'sam_zoom': SamZoomJudge, 'vlm': VlmJudge}


# --------------------------------------------------------------------------- cascade
def combine(ps):
    vals = [(WEIGHTS[k], v) for k, v in ps.items() if k in WEIGHTS and v is not None]
    if not vals:
        return None
    return float(sum(w * v for w, v in vals) / sum(w for w, _ in vals))


def run_judges(source, tracks, cfg, names, fps, workdir=None, progress=None, is_cancelled=None, log=print):
    """Score every track with the configured judges and route it. Mutates ``tracks``."""
    progress = progress or (lambda *a: None)
    is_cancelled = is_cancelled or (lambda: False)
    info = {'loaded': [], 'missing': [], 'seconds': {}}
    judges = {}
    for name in cfg.judges:
        if name not in JUDGES:
            continue
        j = JUDGES[name](cfg, names, log)
        try:
            ok, msg = j.load()
        except Exception as exc:
            ok, msg = False, f"{name} failed to load: {exc}"
        if ok:
            judges[name] = j
            info['loaded'].append(name)
            log(f"[Autopilot] judge {name} ready")
        else:
            info['missing'].append(msg)
            log(f"[Autopilot] judge {name} unavailable: {msg}")
    learned = LearnedJudge.load() if 'learned' in cfg.judges else None  # shared across videos
    if learned is not None:
        info['loaded'].append('learned')
    elif 'learned' in cfg.judges:
        info['missing'].append('learned judge not trained yet (review some tracks or run autopilot_benchmark.py)')
    if not judges and learned is None:
        return info

    progress('Judging: reading crops', 0, 1)
    crops = collect_crops(source, tracks, cfg.judge_crops, 'vlm' in judges, is_cancelled)
    total = len(tracks)
    for n, t in enumerate(tracks):
        if is_cancelled():
            break
        ps, notes = {}, {}
        for name in ('clip', 'sam_zoom', 'vlm'):
            if name not in judges:
                continue
            if name == 'vlm':  # the expensive judge only where the cheap ones are unsure
                pre = combine(ps)
                if not (cfg.vlm_on == 'all' or pre is None or 0.2 < pre < 0.8 or t['route'] == 'review'):
                    continue
            t0 = time.time()
            try:
                ps[name], notes[name] = judges[name].judge(t, crops[t['id']])
            except Exception as exc:
                log(f"[Autopilot] judge {name} error: {exc}")
            info['seconds'][name] = round(info['seconds'].get(name, 0.0) + time.time() - t0, 2)
        if learned is not None:
            ps['learned'] = learned.predict(t['features'], ps)
        _route(t, ps, notes, cfg)
        if n % 5 == 0 or n == total - 1:
            progress('Judging tracks', n + 1, total)
    return info


def _route(t, ps, notes, cfg):
    t['judges'] = {k: (round(v, 3) if v is not None else None) for k, v in ps.items()}
    t['judge_notes'] = notes
    pc = combine(ps)  # decides auto-rejection
    p = ps.get('learned') if ps.get('learned') is not None else pc  # ranks for review
    if p is None:
        return
    t['judge_p'] = round(p, 3)
    strong = pc is not None and (any(ps.get(k) is not None for k in STRONG) or cfg.clip_can_reject)
    reasons = t['reasons']
    mid = int(t['boxes'][len(t['boxes']) // 2][0])
    detail = ', '.join(f"{k} {v:.2f}" for k, v in ps.items() if v is not None)
    clip = notes.get('clip') or {}
    if clip.get('suggest') and clip['suggest'] != t['class'] and clip.get('suggest_p', 0) > 0.5:
        reasons.append({'code': 'judge_class', 'text': f"CLIP thinks it is a {clip['suggest']}",
                        'frame': mid, 'weight': 0.3})
    vlm = notes.get('vlm') or {}
    if vlm.get('answer') and vlm['answer'] != t['class']:
        reasons.append({'code': 'judge_vlm', 'text': f"VLM answer: {vlm['answer']}", 'frame': mid, 'weight': 0.3})
    if cfg.auto_reject and strong and pc <= cfg.reject_below and t['route'] != 'accepted_by_user':
        t['route'] = 'rejected_by_judge'
        reasons.insert(0, {'code': 'judge_reject', 'text': f"judges say not a real {t['class']} ({detail})",
                           'frame': mid, 'weight': 1.0})
        t['risk'] = 1.0
        return
    if p < 0.5:
        reasons.append({'code': 'judge_doubt', 'text': f"judges doubt it ({detail})", 'frame': mid,
                        'weight': round(0.5 - p, 3)})
    keep = 1.0
    for r in reasons:
        keep *= 1.0 - min(max(r['weight'], 0.0), 0.99)
    risk = 1.0 - keep
    if p >= cfg.confirm_above:
        risk *= 0.5
        reasons.append({'code': 'judge_confirm', 'text': f"judges confirm ({detail})", 'frame': mid, 'weight': 0.0})
    reasons.sort(key=lambda r: -r['weight'])
    t['risk'] = round(risk, 3)
    t['route'] = 'review' if t['risk'] >= cfg.review_risk else 'accept'
