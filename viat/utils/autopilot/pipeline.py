"""Autopilot: unattended detect -> track -> repair -> risk-rank pipeline for one video.

Qt-free, so the same code runs from the GUI worker thread and from the headless
``autopilot_label.py`` script. Per-frame detections and camera motion are cached on
disk in chunks, so an interrupted run resumes where it stopped and re-running with
different tracking settings does not repeat the (slow) detection pass.
"""
import hashlib
import json
import os
import pickle
import time
from dataclasses import dataclass, field, asdict

import cv2
import numpy as np

from .detection import CandidateDetector
from .fusion import calibration, fuse_frame
from .judges import run_judges
from .motion import MotionEstimator, CameraPath, split_shots
from .tracker import LifecycleTracker
from .repair import TrackRec, Repairer, class_vote
from .risk import score_track, find_possible_misses
from .audit import sample_frames

RESULT_VERSION = 2
CHUNK = 500


class Cancelled(Exception):
    pass


@dataclass
class AutopilotConfig:
    # what to label: [{'name': 'car', 'prompts': ['car', 'sedan']}, ...]
    classes: list = field(default_factory=list)
    models: list = field(default_factory=list)
    start_frame: int = 0
    end_frame: int = -1
    # detection (more = slower but better recall)
    stride: int = 1
    tiles: int = 0
    tile_overlap: float = 0.25
    flip_tta: bool = False
    # Scores below are on the calibrated scale: each model's typical "confident"
    # raw score maps to 0.5 (YOLO-World 0.3, SAM 3 0.5, Grounding DINO 0.35, OWLv2 0.15).
    candidate_score: float = 0.08
    fuse_iou: float = 0.6
    video_calibration: bool = True
    # tracking life cycle
    low_score: float = 0.15
    high_score: float = 0.4
    birth_score: float = 0.5
    birth_explained_iou: float = 0.3
    match_iou: float = 0.2
    low_match_iou: float = 0.4
    cross_class_factor: float = 0.8
    dist_match: float = 1.0
    min_hits: int = 3
    tentative_max_miss: int = 2
    max_lost_sec: float = 1.5
    cut_detection: bool = True
    # offline repair
    ext_max_sec: float = 2.0
    ext_iou: float = 0.4
    ext_bridge: int = 2
    link_gap_sec: float = 2.0
    link_dist: float = 1.5
    link_app: float = 0.5
    dup_iou: float = 0.5
    dup_min_frames: int = 6
    gap_fill_sec: float = 1.0
    min_track_frames: int = 4
    # review
    review_risk: float = 0.35
    miss_score: float = 0.33
    miss_min_sec: float = 0.3
    promote_min_sec: float = 1.0
    promote_coverage: float = 0.6
    audit_size: int = 30
    seed: int = 0
    # judges (second opinions on every track): clip, sam_zoom, vlm, learned
    judges: list = field(default_factory=lambda: ['clip', 'sam_zoom', 'learned'])
    judge_crops: int = 5
    clip_model: str = 'openai/clip-vit-base-patch32'
    sam_judge_model: str = 'sam3.1_l.pt'
    vlm_model: str = 'Qwen/Qwen2.5-VL-7B-Instruct'
    vlm_on: str = 'uncertain'  # 'uncertain' tracks only, or 'all'
    decoys: list = field(default_factory=lambda: [
        'background', 'part of a vehicle', 'a wheel', 'an antenna', 'a shadow', 'a building',
        'a tree', 'the ground', 'the sky', 'a blurry image'])
    auto_reject: bool = True
    reject_below: float = 0.3  # measured on 10 labeled videos: 0 real boxes lost
    confirm_above: float = 0.8
    clip_can_reject: bool = False

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, d):
        known = {k: v for k, v in (d or {}).items() if k in cls.__dataclass_fields__}
        return cls(**known)

    def class_names(self):
        return [c['name'] for c in self.classes]

    def prompt_table(self):
        """Flat prompt list for the detectors and a prompt -> class-index map."""
        prompts, mapping = [], {}
        for ci, c in enumerate(self.classes):
            for p in (c.get('prompts') or [c['name']]):
                p = p.strip()
                if p and p.lower() not in mapping:
                    mapping[p.lower()] = ci
                    prompts.append(p)
        return prompts, mapping

    def model_cache_key(self, model):
        """Cache key of one detector's raw candidates (independent of the other detectors)."""
        _, mapping = self.prompt_table()
        key = json.dumps([model, sorted(mapping.items()), self.class_names(), self.tiles, self.tile_overlap,
                          self.flip_tta, self.candidate_score, self.fuse_iou, 'v2'], sort_keys=True)
        name = ''.join(ch if ch.isalnum() else '_' for ch in os.path.basename(model))[:30]
        return f"{name}_{hashlib.sha1(key.encode()).hexdigest()[:10]}"


class FrameSource:
    """Sequential frames from a video file or an ordered list of image files."""

    def __init__(self, video_path=None, image_files=None):
        self.video_path = video_path
        self.image_files = list(image_files or [])
        if video_path:
            cap = cv2.VideoCapture(video_path)
            self.count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.fps = float(cap.get(cv2.CAP_PROP_FPS) or 0) or 30.0
            self.width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cap.release()
        else:
            self.count = len(self.image_files)
            self.fps = 30.0
            first = cv2.imread(self.image_files[0]) if self.image_files else None
            self.height, self.width = first.shape[:2] if first is not None else (0, 0)

    @property
    def identity(self):
        path = self.video_path or (self.image_files[0] if self.image_files else 'empty')
        try:
            st = os.stat(path)
            stamp = f"{st.st_size}:{int(st.st_mtime)}"
        except OSError:
            stamp = ''
        return f"{os.path.abspath(path)}|{stamp}|{self.count}"

    def frames(self, start, end):
        if self.video_path:
            cap = cv2.VideoCapture(self.video_path)
            try:
                cap.set(cv2.CAP_PROP_POS_FRAMES, start)
                for f in range(start, end + 1):
                    ok, frame = cap.read()
                    if not ok:
                        break
                    yield f, frame
            finally:
                cap.release()
        else:
            for f in range(start, end + 1):
                frame = cv2.imread(self.image_files[f]) if f < len(self.image_files) else None
                if frame is not None:
                    yield f, frame


def default_workdir():
    if os.name == 'nt':
        base = os.path.join(os.environ.get('APPDATA', os.path.expanduser('~')), 'VideoAnnotationTool')
    else:
        base = os.path.join(os.path.expanduser('~'), '.config', 'VideoAnnotationTool')
    return os.path.join(base, 'autopilot')


def source_dir(source, workdir=None):
    name = os.path.splitext(os.path.basename(source.video_path or 'images'))[0][:40]
    digest = hashlib.sha1(source.identity.encode()).hexdigest()[:10]
    return os.path.join(workdir or default_workdir(), f"{name}_{digest}")


def result_path(source, workdir=None):
    return os.path.join(source_dir(source, workdir), 'result.json')


class ChunkCache:
    """``frame -> value`` stored in pickled chunks of 500 frames (resumable, crash-safe)."""

    def __init__(self, directory):
        self.dir = directory
        os.makedirs(directory, exist_ok=True)
        self.chunks = {}
        self.dirty = set()

    def _path(self, c):
        return os.path.join(self.dir, f"chunk_{c:06d}.pkl")

    def _chunk(self, c):
        if c not in self.chunks:
            data = {}
            try:
                with open(self._path(c), 'rb') as fh:
                    data = pickle.load(fh)
                if not isinstance(data, dict) or 'det' in data:  # pre-v2 layout
                    data = {}
            except (OSError, EOFError, pickle.UnpicklingError):
                data = {}
            self.chunks[c] = data
        return self.chunks[c]

    def has(self, f):
        return f in self._chunk(f // CHUNK)

    def get(self, f):
        return self._chunk(f // CHUNK).get(f)

    def put(self, f, value):
        self._chunk(f // CHUNK)[f] = value
        self.dirty.add(f // CHUNK)

    def flush(self):
        for c in list(self.dirty):
            tmp = self._path(c) + '.tmp'
            with open(tmp, 'wb') as fh:
                pickle.dump(self.chunks[c], fh, protocol=pickle.HIGHEST_PROTOCOL)
            os.replace(tmp, self._path(c))
        self.dirty.clear()


def run_autopilot(source, cfg, progress=None, is_cancelled=None, log=print, workdir=None):
    """Run the whole pipeline and return a JSON-serialisable result dict."""
    t0 = time.time()
    progress = progress or (lambda stage, done, total: None)
    is_cancelled = is_cancelled or (lambda: False)
    start = max(0, int(cfg.start_frame))
    end = int(cfg.end_frame) if cfg.end_frame is not None and cfg.end_frame >= 0 else source.count - 1
    end = min(end, source.count - 1)
    if end < start:
        raise ValueError('Empty frame range')
    names = cfg.class_names()
    if not names:
        raise ValueError('No classes configured')
    if not cfg.models:
        raise ValueError('No detectors configured')
    fps = source.fps

    # ---------------------------------------------------------- 1. detect + motion
    # Motion and every model's detections are cached separately, so runs with a
    # different set of detectors reuse whatever was already computed.
    base = source_dir(source, workdir)
    motion_cache = ChunkCache(os.path.join(base, 'motion'))
    det_caches = {k: ChunkCache(os.path.join(base, 'dets_' + cfg.model_cache_key(k))) for k in cfg.models}
    want_det = lambda f: (f - start) % max(1, cfg.stride) == 0  # noqa: E731
    first_missing = None
    for f in range(start, end + 1):
        if not motion_cache.has(f) or (want_det(f) and not all(c.has(f) for c in det_caches.values())):
            first_missing = f
            break
    detector = CandidateDetector(cfg, log=log)
    load_errors = {}
    if first_missing is not None:
        estimator = MotionEstimator()
        read_from = max(start, first_missing - 1)  # previous frame primes the motion estimate
        total = end - first_missing + 1
        done = 0
        for f, frame in source.frames(read_from, end):
            if is_cancelled():
                motion_cache.flush()
                for c in det_caches.values():
                    c.flush()
                raise Cancelled()
            mot = estimator.update(frame)
            if f < first_missing:
                continue
            if not motion_cache.has(f):
                motion_cache.put(f, (mot[0].astype(np.float32), mot[1], mot[2]))
            mot = motion_cache.get(f)
            need = want_det(f) or (f > start and cfg.cut_detection and mot[2] < 0.6)
            if need:
                missing = [k for k, c in det_caches.items() if not c.has(f) and k not in load_errors]
                if missing:
                    for k in missing:
                        if not detector.has(k):
                            err = detector.load_one(k)
                            if err:
                                load_errors[k] = err
                                log(f"[Autopilot] skipped model {err}")
                    out = detector.detect(frame, [k for k in missing if k not in load_errors])
                    for k, v in out.items():
                        det_caches[k].put(f, v)
            done += 1
            if done % 25 == 0 or f == end:
                progress('Detecting', done, total)
            if done % 200 == 0:
                motion_cache.flush()
                for c in det_caches.values():
                    c.flush()
        motion_cache.flush()
        for c in det_caches.values():
            c.flush()
    else:
        log('[Autopilot] detections loaded from cache')
    if load_errors and len(load_errors) == len(cfg.models):
        raise RuntimeError('No detector could be loaded:\n' + '\n'.join(load_errors.values()))
    motion, records = {}, {}
    for f in range(start, end + 1):
        mot = motion_cache.get(f)
        if mot is None:
            break  # video ended early
        motion[f] = mot
        per_model = {k: c.get(f) for k, c in det_caches.items() if c.has(f)}
        records[f] = per_model or None
    frames = sorted(motion)
    motion[start] = (np.array([[1, 0, 0], [0, 1, 0]], np.float32), 1.0, 1.0)

    # calibrate each model's scores for this video, then fuse models per frame
    raw = records
    keys = [k for k in cfg.models if any(r is not None and k in r for r in raw.values())]
    if not keys:
        raise RuntimeError('No detections were produced (all detectors failed?)')
    scale = calibration(raw, keys, cfg)
    records = {f: (fuse_frame(r, keys, scale, cfg, len(names)) if r is not None else None)
               for f, r in raw.items()}
    del raw

    # ---------------------------------------------------------------- 2. tracking
    shots = split_shots(frames, motion, cfg.cut_detection)
    shot_of = {}
    for si, (s, e) in enumerate(shots):
        for f in range(s, e + 1):
            shot_of[f] = si
    path = CameraPath(motion, shots)
    tracker = LifecycleTracker(cfg, fps)
    finished = []
    for si, (s, e) in enumerate(shots):
        if is_cancelled():
            raise Cancelled()
        for f in range(s, e + 1):
            affine = None if f == s else np.asarray(motion[f][0], np.float64)
            tracker.step(f, records.get(f), affine)
        tracker.flush()
        progress('Tracking', si + 1, len(shots))
    for t in tracker.finished:
        boxes = {f: v for f, v in t.obs.items()}
        finished.append(TrackRec(t.id, shot_of[min(boxes)], boxes, t.cls_sum))
    n_rejected = len(tracker.rejected)

    # ------------------------------------------------------------------ 3. repair
    progress('Repairing tracks', 0, 1)
    repairer = Repairer(cfg, fps, records, path, shots)
    tracks, n_dup = repairer.run(finished)
    kept = []
    for t in tracks:
        if len(t.det_frames()) < cfg.min_hits or len(t.boxes) < cfg.min_track_frames:
            n_rejected += 1
        else:
            kept.append(t)
    tracks = kept
    progress('Repairing tracks', 1, 1)

    # Offline we can afford hindsight: weak candidates that persist for a long time are
    # almost always real objects the detector is poorly calibrated for. Promote them to
    # tracks (always flagged for review) instead of leaving them as possible misses.
    names_idx = {n: i for i, n in enumerate(names)}
    n_promoted = 0
    if cfg.promote_min_sec > 0:
        min_len = max(cfg.min_track_frames, int(round(cfg.promote_min_sec * fps)))
        next_id = max([t.id for t in tracks] + [tracker.next_id]) + 1
        promoted = []
        for m in find_possible_misses(records, tracks, frames, path, cfg, fps, names):
            if len(m['boxes']) < min_len or m['coverage'] < cfg.promote_coverage:
                continue
            cls_sum = np.zeros(len(names), np.float32)
            cls_sum[names_idx.get(m['class'], 0)] = sum(r[5] for r in m['boxes'])
            boxes = {int(r[0]): (np.asarray(r[1:5], np.float32), float(r[5]), -1, 'd') for r in m['boxes']}
            trk = TrackRec(next_id, shot_of[m['start']], boxes, cls_sum)
            trk.promoted = True
            next_id += 1
            promoted.append(trk)
        if promoted:
            repairer.fill_gaps(promoted)
            tracks.extend(promoted)
            n_promoted = len(promoted)

    # -------------------------------------------------------------------- 4. risk
    by_frame = {}
    for t in tracks:
        for f, v in t.boxes.items():
            by_frame.setdefault(f, []).append((t.id, v[0]))
    ctx = {'fps': fps, 'path': path, 'records': records, 'n_models': len(keys), 'names': names,
           'by_frame': by_frame,
           'desc': repairer._mean_desc, 'bhatt': Repairer.bhattacharyya}
    out_tracks = []
    for t in sorted(tracks, key=lambda t: t.start):
        risk, reasons, feats = score_track(t, ctx)
        top, margin, shares = class_vote(t, names)
        route = 'review' if risk >= cfg.review_risk else 'accept'
        focus = reasons[0][2] if reasons else t.frames[len(t.frames) // 2]
        out_tracks.append({
            'id': int(t.id), 'class': names[top], 'class_margin': round(margin, 3),
            'class_shares': {k: round(v, 3) for k, v in shares.items()},
            'route': route, 'risk': round(risk, 3), 'focus_frame': int(focus),
            'reasons': [{'code': c, 'text': txt, 'frame': fr, 'weight': round(w, 3)} for c, txt, fr, w in reasons],
            'start': int(t.start), 'end': int(t.end), 'shot': int(t.shot),
            'n_det': len(t.det_frames()), 'links': int(t.links),
            'ext_back': int(t.ext['back']), 'ext_fwd': int(t.ext['fwd']),
            'promoted': bool(getattr(t, 'promoted', False)),
            'mean_score': round(t.mean_score(), 3),
            'features': feats,
            # [frame, x1, y1, x2, y2, score, kind]
            'boxes': [[int(f)] + [round(float(v), 1) for v in t.boxes[f][0]] +
                      [round(float(t.boxes[f][1]), 3), t.boxes[f][3]] for f in t.frames],
        })
    misses = find_possible_misses(records, tracks, frames, path, cfg, fps, names)
    for i, m in enumerate(misses):
        m['id'] = i + 1
        m['status'] = 'open'

    # -------------------------------------------------------------------- 5. judges
    judge_info = {}
    if cfg.judges and out_tracks:
        judge_info = run_judges(source, out_tracks, cfg, names, fps, workdir=workdir, progress=progress,
                                is_cancelled=is_cancelled, log=log)

    n_review = sum(1 for t in out_tracks if t['route'] == 'review')
    n_judge_rejected = sum(1 for t in out_tracks if t['route'] == 'rejected_by_judge')
    result = {
        'version': RESULT_VERSION,
        'created': time.strftime('%Y-%m-%d %H:%M:%S'),
        'source': {'video': source.video_path, 'images': bool(source.image_files), 'identity': source.identity,
                   'fps': fps, 'frame_count': source.count, 'width': source.width, 'height': source.height},
        'config': cfg.to_dict(),
        'start_frame': start, 'end_frame': end,
        'shots': [[int(s), int(e)] for s, e in shots],
        'tracks': out_tracks,
        'misses': misses,
        'stats': {
            'frames': len(frames), 'detected_frames': sum(1 for f in frames if records.get(f) is not None),
            'tracks': len(out_tracks) - n_judge_rejected, 'review': n_review,
            'accepted': len(out_tracks) - n_review - n_judge_rejected, 'judge_rejected': n_judge_rejected,
            'judges': judge_info, 'calibration': {k: (round(v, 3) if v else None) for k, v in scale.items()},
            'rejected': n_rejected, 'duplicates_removed': n_dup, 'promoted_low_conf': n_promoted, 'possible_misses': len(misses),
            'boxes': sum(len(t['boxes']) for t in out_tracks if t['route'] != 'rejected_by_judge'), 'seconds': round(time.time() - t0, 1),
            'models_loaded': keys, 'model_errors': list(load_errors.values()),
        },
        'audit': {'frames': sample_frames(frames, cfg.audit_size, seed=cfg.seed), 'decisions': {}},
    }
    log(f"[Autopilot] {result['stats']['tracks']} tracks ({n_review} to review, {n_judge_rejected} rejected by "
        f"judges), {len(misses)} possible misses, {n_rejected} filtered, {result['stats']['seconds']} s")
    return result


def save_result(path, result):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + '.tmp'
    with open(tmp, 'w', encoding='utf-8') as fh:
        json.dump(result, fh)
    os.replace(tmp, path)


def load_result(path):
    with open(path, 'r', encoding='utf-8') as fh:
        return json.load(fh)
