"""Tools > Autopilot: runs the unattended labeling pipeline in a background thread,
writes its tracks into the project as unverified "autopilot" boxes and drives the
review panel (flagged tracks, possible misses, random-frame audit).

The existing Auto Label / tracking tools are not touched; Autopilot only adds boxes
whose ``source`` starts with ``autopilot`` and carry ``track_id`` / ``autopilot_id``.
"""
import os
import traceback

import numpy as np
from PyQt5.QtCore import Qt, QThread, QRect, pyqtSignal
from PyQt5.QtGui import QColor
from PyQt5.QtWidgets import QMessageBox, QWidget, QHBoxLayout, QProgressBar, QPushButton, QLabel

from ..annotation import BoundingBox
from ..logger import logger
from ..utils.autopilot import (AutopilotConfig, FrameSource, Cancelled, run_autopilot, save_result,
                               load_result, result_path, sample_frames)
from ..utils.autopilot.learned import log_decision, train as train_learned_judge
from ..widgets.autopilot_dialog import AutopilotDialog
from ..widgets.autopilot_review_dock import AutopilotReviewDock

SOURCE_BY_KIND = {'d': 'autopilot', 'e': 'autopilot_ext', 'i': 'autopilot_interp'}


def _is_autopilot(ann):
    return str(getattr(ann, 'source', '') or '').startswith('autopilot')


class AutopilotWorker(QThread):
    progress = pyqtSignal(str, int, int)
    done = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(self, source, cfg):
        super().__init__()
        self.source = source
        self.cfg = cfg
        self._cancel = False

    def cancel(self):
        self._cancel = True

    def run(self):
        try:
            res = run_autopilot(self.source, self.cfg, progress=self.progress.emit,
                                is_cancelled=lambda: self._cancel, log=logger.info)
            self.done.emit(res)
        except Cancelled:
            self.failed.emit('cancelled')
        except Exception as exc:
            logger.error(traceback.format_exc())
            self.failed.emit(str(exc))


class AutopilotController:
    def __init__(self, main_window):
        self.mw = main_window
        self.result = None
        self.result_file = None
        self.worker = None
        self.dock = None
        self.last_config = None
        self.apply_opts = {'replace_previous': True, 'skip_existing': True}
        self._progress_widget = None
        self._clear_preview()

    # ------------------------------------------------------------------ media
    def _source(self):
        mw = self.mw
        if getattr(mw, 'is_image_dataset', False) and getattr(mw, 'image_files', None):
            return FrameSource(image_files=mw.image_files)
        if getattr(mw, 'video_filename', None):
            return FrameSource(video_path=mw.video_filename)
        return None

    def _matches_current(self, warn=True):
        src = self._source()
        ok = bool(self.result and src and self.result['source'].get('identity') == src.identity)
        if not ok and warn:
            QMessageBox.information(self.mw, 'Autopilot',
                                    'The loaded Autopilot result belongs to another video. '
                                    'Use "Load result for this video" or run Autopilot again.')
        return ok

    def _checkpoints_dir(self):
        return os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                            'checkpoints')

    def class_names(self):
        return list(self.mw.canvas.class_colors.keys())

    def _collect_exemplars(self, cfg):
        """Collect verified or manual non-Autopilot boxes to use as visual exemplar seeds."""
        self.mw.update_frame_annotations()
        wanted = {name.lower() for name in cfg.class_names()}
        limit = max(1, int(cfg.exemplar_limit))
        counts = {}
        examples = []

        all_frames = sorted(self.mw.frame_annotations)
        in_range = [f for f in all_frames if cfg.start_frame <= f <= cfg.end_frame]
        out_of_range = [f for f in all_frames if f < cfg.start_frame or f > cfg.end_frame]
        search_frames = in_range + out_of_range

        for frame in search_frames:
            for ann in self.mw.frame_annotations.get(frame, []):
                name = str(getattr(ann, 'class_name', '') or '')
                key = name.lower()
                if key not in wanted or _is_autopilot(ann):
                    continue
                is_valid = getattr(ann, 'verified', False) or getattr(ann, 'source', '') == 'manual'
                if not is_valid:
                    continue
                if counts.get(key, 0) >= limit:
                    continue
                rect = ann.rect
                if rect.width() < 3 or rect.height() < 3:
                    continue
                examples.append({'frame': int(frame), 'class': name,
                                 'box': [int(rect.x()), int(rect.y()),
                                         int(rect.x() + rect.width()),
                                         int(rect.y() + rect.height())]})
                counts[key] = counts.get(key, 0) + 1
                if all(counts.get(k, 0) >= limit for k in wanted):
                    break
            if all(counts.get(k, 0) >= limit for k in wanted):
                break

        logger.info(f"[Autopilot] Collected {len(examples)} visual exemplar(s) across {len(counts)} class(es).")
        return examples

    def _scan_seed_counts(self):
        """Count verified/manual non-Autopilot boxes in the project per class."""
        self.mw.update_frame_annotations()
        counts = {}
        for frame, anns in self.mw.frame_annotations.items():
            for ann in anns:
                if _is_autopilot(ann):
                    continue
                is_valid = getattr(ann, 'verified', False) or getattr(ann, 'source', '') == 'manual'
                if not is_valid:
                    continue
                rect = ann.rect
                if rect.width() < 3 or rect.height() < 3:
                    continue
                name = str(getattr(ann, 'class_name', '') or '').strip()
                if name:
                    counts[name] = counts.get(name, 0) + 1
        return counts

    # -------------------------------------------------------------------- run
    def open_dialog(self):
        if self.worker is not None and self.worker.isRunning():
            QMessageBox.information(self.mw, 'Autopilot', 'Autopilot is already running.')
            return
        src = self._source()
        if src is None or src.count <= 0:
            QMessageBox.warning(self.mw, 'Autopilot', 'Please open a video (or an ordered frame folder) first.')
            return
        # Flush a just-drawn box before the dialog optionally turns it into a
        # visual exemplar.
        self.mw.update_frame_annotations()
        last = self.last_config
        if last is None and os.path.exists(result_path(src)):
            try:
                last = load_result(result_path(src)).get('config')
            except (OSError, ValueError):
                last = None
        seed_counts = self._scan_seed_counts()
        dlg = AutopilotDialog(self.mw, self.class_names(), self.mw.current_frame, src.count,
                              self._checkpoints_dir(), last, seed_counts=seed_counts)
        if not dlg.exec_():
            return
        cfg = dlg.get_config()
        if cfg.use_exemplars:
            cfg.exemplars = self._collect_exemplars(cfg)
            if not cfg.exemplars:
                QMessageBox.information(
                    self.mw, 'Autopilot visual examples',
                    'No manually verified boxes were found in this frame range. '
                    'Autopilot will continue with text prompts only.')
                cfg.use_exemplars = False
        self.apply_opts = dlg.apply_options()
        self.last_config = cfg.to_dict()
        self.start(cfg, src)

    def start(self, cfg, src):
        self._clear_preview()
        self.worker = AutopilotWorker(src, cfg)
        self.worker.progress.connect(self._on_progress)
        self.worker.done.connect(self._on_done)
        self.worker.failed.connect(self._on_failed)
        w = QWidget()
        lay = QHBoxLayout(w)
        lay.setContentsMargins(0, 0, 0, 0)
        self._progress_label = QLabel('Autopilot: loading models...')
        self._progress_bar = QProgressBar()
        self._progress_bar.setRange(0, 0)
        cancel = QPushButton('Cancel Autopilot')
        cancel.clicked.connect(self.worker.cancel)
        lay.addWidget(self._progress_label)
        lay.addWidget(self._progress_bar)
        lay.addWidget(cancel)
        self._progress_widget = w
        self.mw.statusBar.addPermanentWidget(w)
        self.worker.start()

    def _on_progress(self, stage, done, total):
        self._progress_label.setText(f'Autopilot: {stage} {done}/{total}')
        self._progress_bar.setRange(0, max(1, total))
        self._progress_bar.setValue(done)

    def _remove_progress(self):
        if self._progress_widget is not None:
            self.mw.statusBar.removeWidget(self._progress_widget)
            self._progress_widget.deleteLater()
            self._progress_widget = None

    def _on_failed(self, msg):
        self._remove_progress()
        if msg == 'cancelled':
            self.mw.statusBar.showMessage('Autopilot cancelled. Detections so far are cached; '
                                          'running again resumes from there.', 8000)
        else:
            QMessageBox.warning(self.mw, 'Autopilot failed', msg)

    def _on_done(self, res):
        self._remove_progress()
        src = self.worker.source
        current = self._source()
        if current is None or current.identity != src.identity:
            # The user opened another video while Autopilot ran: never write into it.
            save_result(result_path(src), res)
            QMessageBox.information(
                self.mw, 'Autopilot finished',
                f"Autopilot finished for {os.path.basename(src.video_path or 'the image folder')}, "
                f"but another video is open now.\nThe result is saved: open that video and use "
                f"Tools > Autopilot Review Panel to add the boxes and review them.")
            return
        self.result = res
        self.result_file = result_path(src)
        self._apply(res, **self.apply_opts)
        self._save()
        self.show_dock()
        s = res['stats']
        QMessageBox.information(
            self.mw, 'Autopilot finished',
            f"Labeled {s['tracks']} objects ({s['boxes']} boxes) in {s['seconds'] / 60:.1f} min.\n\n"
            f"Please review: {s['review']} flagged tracks and {s['possible_misses']} possible misses.\n"
            f"Then check the random frames in the 'Random audit' tab.")

    # ------------------------------------------------------------------ apply
    def _ensure_class(self, name):
        canvas = self.mw.canvas
        if name in canvas.class_colors:
            return
        rng = np.random.default_rng(abs(hash(name)) % (2 ** 32))
        canvas.class_colors[name] = QColor(*[int(v) for v in rng.integers(40, 255, 3)])
        if getattr(canvas, 'class_attributes', None) is None:
            canvas.class_attributes = {}
        canvas.class_attributes.setdefault(name, {})
        self.mw.class_attributes = canvas.class_attributes
        self._classes_added = True

    def _next_label_id(self):
        best = 0
        for anns in self.mw.frame_annotations.values():
            for a in anns:
                tid = (getattr(a, 'attributes', None) or {}).get('track_id')
                try:
                    best = max(best, int(tid))
                except (TypeError, ValueError):
                    pass
        return best + 1

    def _make_box(self, row, cls, label_id, verified=False):
        f, x1, y1, x2, y2, score, kind = row[:7]
        metadata = row[7] if len(row) > 7 and isinstance(row[7], dict) else {}
        cfg = (self.result or {}).get('config') or {}
        slack = float(cfg.get('box_slack', 0.0) or 0.0)
        if slack > 0:
            bw = x2 - x1
            bh = y2 - y1
            x1 -= bw * slack
            x2 += bw * slack
            y1 -= bh * slack
            y2 += bh * slack
        W, H = (self.result.get('source') or {}).get('width') or 0, (self.result.get('source') or {}).get('height') or 0
        if W and H:
            x1, x2 = max(0.0, min(x1, W - 1)), max(0.0, min(x2, W))
            y1, y2 = max(0.0, min(y1, H - 1)), max(0.0, min(y2, H))
        if x2 - x1 < 1 or y2 - y1 < 1:
            return None
        attrs = dict(self.mw.get_default_attributes_for_class(cls) or {})
        attrs['track_id'] = label_id
        attrs['autopilot_id'] = label_id
        detector_scores = metadata.get('detector_scores') or {}
        if detector_scores:
            attrs['autopilot_detector_scores'] = {
                str(name): round(float(value), 3)
                for name, value in detector_scores.items()
            }
            attrs['autopilot_detectors'] = list(attrs['autopilot_detector_scores'])
        if 'nmod' in metadata:
            try:
                attrs['autopilot_detector_votes'] = int(metadata['nmod'])
            except (TypeError, ValueError):
                pass
        segmentation = metadata.get('segmentation') or metadata.get('polygon')
        ann = BoundingBox(QRect(int(round(x1)), int(round(y1)), int(round(x2 - x1)), int(round(y2 - y1))), cls,
                          attributes=attrs, color=self.mw.canvas.class_colors.get(cls, QColor(0, 255, 0)),
                          source=SOURCE_BY_KIND.get(kind, 'autopilot'), score=float(score),
                          segmentation=segmentation)
        ann.verified = verified
        return ann

    @staticmethod
    def _set_track_provenance(ann, track):
        """Attach the track-level detector summary to a project annotation."""
        if ann is None:
            return
        detectors = track.get('detectors') or {}
        if detectors:
            models = list(detectors)
            ann.attributes['autopilot_detector_models'] = models
            ann.attributes.setdefault('autopilot_detectors', models)
            ann.attributes['autopilot_detector_summary'] = {
                str(name): {
                    key: value for key, value in (summary or {}).items()
                    if key in ('frames', 'coverage', 'mean_score', 'max_score')
                }
                for name, summary in detectors.items()
            }

    def _apply(self, res, replace_previous=True, skip_existing=True):
        mw = self.mw
        mw.update_frame_annotations()  # flush canvas edits before touching the frame lists
        start, end = res['start_frame'], res['end_frame']
        mw.save_undo_state(range(start, end + 1))
        fa = mw.frame_annotations
        if replace_previous:
            for f in range(start, end + 1):
                if f in fa:
                    fa[f] = [a for a in fa[f] if not _is_autopilot(a)]
        self._classes_added = False
        next_id = self._next_label_id()
        added = 0
        for trk in res['tracks']:
            if trk['route'] in ('deleted_by_user', 'rejected_by_judge'):
                continue
            self._ensure_class(trk['class'])
            trk['label_id'] = next_id
            next_id += 1
            verified = trk['route'] == 'accepted_by_user' or trk.get('verified', False)
            skipped = 0
            for row in trk['boxes']:
                f = int(row[0])
                if skip_existing and fa.get(f):
                    manual = [a.rect for a in fa[f] if not _is_autopilot(a)]
                    if manual and self._max_iou(row[1:5], manual) >= 0.6:
                        skipped += 1
                        continue
                ann = self._make_box(row, trk['class'], trk['label_id'], verified)
                if ann is not None:
                    self._set_track_provenance(ann, trk)
                    fa.setdefault(f, []).append(ann)
                    added += 1
            trk['skipped_existing'] = skipped
        if self._classes_added and hasattr(mw, 'refresh_class_ui'):
            mw.refresh_class_ui()
        mw.project_modified = True
        self._refresh_canvas()
        mw.statusBar.showMessage(f'Autopilot: added {added} boxes.', 6000)
        logger.info(f'[Autopilot] applied {added} boxes to frames {start}-{end}')

    @staticmethod
    def _max_iou(box, rects):
        x1, y1, x2, y2 = box
        best = 0.0
        for r in rects:
            ix = max(0.0, min(x2, r.x() + r.width()) - max(x1, r.x()))
            iy = max(0.0, min(y2, r.y() + r.height()) - max(y1, r.y()))
            inter = ix * iy
            union = (x2 - x1) * (y2 - y1) + r.width() * r.height() - inter
            if union > 0:
                best = max(best, inter / union)
        return best

    def _refresh_canvas(self):
        self._clear_preview()
        mw = self.mw
        mw.load_current_frame_annotations()
        mw.update_annotation_list()
        mw.canvas.update()

    def _clear_preview(self):
        """Remove temporary Autopilot-only overlays without touching the project."""
        canvas = getattr(self.mw, 'canvas', None)
        if canvas is None:
            return
        preview = getattr(canvas, 'preview_annotations', None)
        if preview:
            canvas.preview_annotations = []
            canvas.update()

    def show_rejected_track(self, tid, frame=None):
        """Show an auto-rejected track on the canvas without adding it to the project."""
        if not self.result or not self._matches_current(warn=False):
            return
        trk = self.track_by_id(tid, include_deleted=True)
        if not trk or trk.get('route') != 'rejected_by_judge':
            return
        self._clear_preview()
        target = int(frame if frame is not None else trk.get('focus_frame', trk['start']))
        rows = list(trk.get('boxes', []))
        if rows and not any(int(row[0]) == target for row in rows):
            target = int(min(rows, key=lambda row: abs(int(row[0]) - target))[0])
        self.go_to_frame(target)
        # If the requested frame was already current, go_to_frame() does not reload
        # annotations, so explicitly clear any stale preview before adding this one.
        self._clear_preview()
        label_id = trk.get('label_id', trk.get('id', tid))
        overlays = []
        for row in rows:
            if int(row[0]) != target:
                continue
            ann = self._make_box(row, trk['class'], label_id, verified=False)
            if ann is not None:
                self._set_track_provenance(ann, trk)
                ann.is_autopilot_preview = True
                ann.preview_kind = 'rejected'
                ann.color = QColor(255, 92, 72)
                overlays.append(ann)
        canvas = getattr(self.mw, 'canvas', None)
        if canvas is not None:
            canvas.preview_annotations = overlays
            canvas.update()

    def step_rejected_track(self, tid, direction):
        """Step backward or forward through the frames where a rejected track was detected."""
        trk = self.track_by_id(tid, include_deleted=True)
        if not trk or trk.get('route') != 'rejected_by_judge':
            return
        rows = trk.get('boxes', [])
        if not rows:
            return
        frames = sorted(set(int(r[0]) for r in rows))
        cur = self.mw.current_frame
        if cur not in frames:
            target = frames[0] if direction > 0 else frames[-1]
        else:
            idx = frames.index(cur)
            next_idx = min(len(frames) - 1, max(0, idx + direction))
            target = frames[next_idx]
        self.show_rejected_track(tid, target)

    def rerun_with_detectors(self, models):
        """Start a cached Autopilot rerun using only the selected detector models."""
        if self.worker is not None and self.worker.isRunning():
            QMessageBox.information(self.mw, 'Autopilot', 'Autopilot is already running.')
            return
        if not models:
            QMessageBox.warning(self.mw, 'Autopilot', 'Select at least one detector.')
            return
        src = self._source()
        if src is None or not self.result or not self._matches_current():
            return
        cfg_data = self.result.get('config') or self.last_config or {}
        cfg = AutopilotConfig.from_dict(cfg_data)
        cfg.models = list(models)
        self.last_config = cfg.to_dict()
        self.start(cfg, src)
        if self.mw.statusBar:
            self.mw.statusBar.showMessage(
                'Autopilot: rerunning with selected detectors; cached detections will be reused.', 6000)

    def _save(self):
        if self.result and self.result_file:
            try:
                save_result(self.result_file, self.result)
            except OSError as exc:
                logger.error(f'[Autopilot] could not save result: {exc}')

    def reapply_result(self):
        if not self.result or not self._matches_current():
            return
        if QMessageBox.question(self.mw, 'Autopilot', 'Write the Autopilot result into the project again? '
                                'Earlier Autopilot boxes in the range are replaced; manual labels are kept.') \
                != QMessageBox.Yes:
            return
        self._apply(self.result, **self.apply_opts)
        self._save()
        self.show_dock()

    def load_saved_result(self):
        src = self._source()
        path = result_path(src) if src else None
        if not path or not os.path.exists(path):
            QMessageBox.information(self.mw, 'Autopilot', 'No saved Autopilot result for this video yet.')
            return
        self.result = load_result(path)
        self.result_file = path
        if any('label_id' not in t for t in self.result['tracks'] if t['route'] != 'deleted_by_user'):
            # produced headless (autopilot_label.py): it has not been written into the project yet
            if QMessageBox.question(self.mw, 'Autopilot', 'This result has not been added to the project yet. '
                                    'Add its boxes now?') == QMessageBox.Yes:
                self._apply(self.result, **self.apply_opts)
                self._save()
        self.show_dock()

    # ---------------------------------------------------------------- dock
    def show_dock(self):
        if self.dock is None:
            self.dock = AutopilotReviewDock(self, self.mw)
            self.mw.addDockWidget(Qt.RightDockWidgetArea, self.dock)
        self.dock.show()
        self.dock.raise_()
        self.dock.refresh()

    def track_by_id(self, tid, include_deleted=False):
        if not self.result:
            return None
        for t in self.result['tracks']:
            if t['id'] == tid:
                return t if include_deleted or t['route'] != 'deleted_by_user' else None
        return None

    def _track_boxes(self, trk):
        lid = trk.get('label_id')
        out = []
        for f in range(trk['start'], trk['end'] + 1):
            for a in self.mw.frame_annotations.get(f, []):
                if (getattr(a, 'attributes', None) or {}).get('autopilot_id') == lid:
                    out.append((f, a))
        return out

    def go_to_frame(self, frame):
        frame = int(frame)
        if self.mw.current_frame != frame:
            self.mw.update_frame_annotations()
            self.mw.set_current_frame(frame)

    def show_track(self, tid, frame):
        if not self._matches_current(warn=False):
            return
        trk = self.track_by_id(tid)
        if not trk:
            return
        self._clear_preview()
        self.go_to_frame(frame)
        self._select_on_current(trk)

    def _select_on_current(self, trk):
        for a in self.mw.frame_annotations.get(self.mw.current_frame, []):
            if (getattr(a, 'attributes', None) or {}).get('autopilot_id') == trk.get('label_id'):
                self.mw.sync_annotation_selection(a)
                return

    def step_track(self, tid, direction):
        trk = self.track_by_id(tid)
        if not trk or not self._matches_current():
            return
        step = max(1, int(round(self.result['source'].get('fps', 30) / 2)))
        cur = self.mw.current_frame
        if not trk['start'] <= cur <= trk['end']:
            target = trk['start'] if direction > 0 else trk['end']
        else:
            target = min(trk['end'], max(trk['start'], cur + direction * step))
        self.go_to_frame(target)
        self._select_on_current(trk)

    def _decide(self, trk, route, label=None):
        trk['route'] = route
        if label is not None:  # every review decision teaches the learned judge
            video = os.path.basename((self.result['source'].get('video') or '')) if self.result else ''
            try:
                log_decision(trk, label, source='user', video=video)
            except OSError as exc:
                logger.error(f'[Autopilot] could not log decision: {exc}')
        self._save()

    def accept_track(self, tid):
        trk = self.track_by_id(tid)
        if not trk or not self._matches_current():
            return
        self.mw.update_frame_annotations()
        for _f, a in self._track_boxes(trk):
            a.verified = True
        self.mw.project_modified = True
        self._decide(trk, 'accepted_by_user', label=1)
        self.mw.canvas.update()

    def delete_track(self, tid):
        trk = self.track_by_id(tid)
        if not trk or not self._matches_current():
            return
        mw = self.mw
        mw.update_frame_annotations()
        mw.save_undo_state(range(trk['start'], trk['end'] + 1))
        lid = trk.get('label_id')
        for f in range(trk['start'], trk['end'] + 1):
            if f in mw.frame_annotations:
                mw.frame_annotations[f] = [a for a in mw.frame_annotations[f]
                                           if (getattr(a, 'attributes', None) or {}).get('autopilot_id') != lid]
        mw.project_modified = True
        self._decide(trk, 'deleted_by_user', label=0)
        self._refresh_canvas()

    def fill_track_gaps(self, tid):
        """Linearly interpolate every frame missing inside the track's span."""
        trk = self.track_by_id(tid)
        if not trk or not self._matches_current():
            return
        mw = self.mw
        mw.update_frame_annotations()
        mw.save_undo_state(range(trk['start'], trk['end'] + 1))
        rows = {int(r[0]): r for r in trk['boxes']}
        have = {f for f, _a in self._track_boxes(trk)}
        frames = sorted(rows)
        added = 0
        for f0, f1 in zip(frames[:-1], frames[1:]):
            gap = f1 - f0
            for k in range(1, gap):
                f = f0 + k
                if f in have:
                    continue
                t = k / gap
                a, b = rows[f0], rows[f1]
                row = [f] + [a[i] * (1 - t) + b[i] * t for i in range(1, 5)] + [min(a[5], b[5]), 'i']
                ann = self._make_box(row, trk['class'], trk['label_id'], verified=True)
                if ann is not None:
                    self._set_track_provenance(ann, trk)
                    mw.frame_annotations.setdefault(f, []).append(ann)
                    trk['boxes'].append(row)
                    added += 1
        trk['boxes'].sort(key=lambda r: r[0])
        mw.project_modified = True
        self._save()
        self._refresh_canvas()
        mw.statusBar.showMessage(f'Autopilot: filled {added} frames in track #{trk["label_id"]}.', 5000)

    def relabel_track(self, tid, new_class):
        trk = self.track_by_id(tid)
        if not trk or not self._matches_current():
            return
        self.mw.update_frame_annotations()
        color = self.mw.canvas.class_colors.get(new_class, QColor(0, 255, 0))
        for _f, a in self._track_boxes(trk):
            a.class_name = new_class
            a.color = color
            a.verified = True
        trk['class'] = new_class
        self.mw.project_modified = True
        self._decide(trk, 'accepted_by_user', label=1)
        self._refresh_canvas()

    def restore_track(self, tid):
        """Bring back a track the judges auto-rejected (it was real after all)."""
        trk = self.track_by_id(tid)
        if not trk or trk['route'] != 'rejected_by_judge' or not self._matches_current():
            return
        mw = self.mw
        mw.update_frame_annotations()
        mw.save_undo_state(range(trk['start'], trk['end'] + 1))
        self._ensure_class(trk['class'])
        trk['label_id'] = self._next_label_id()
        for row in trk['boxes']:
            ann = self._make_box(row, trk['class'], trk['label_id'], verified=True)
            if ann is not None:
                self._set_track_provenance(ann, trk)
                mw.frame_annotations.setdefault(int(row[0]), []).append(ann)
        mw.project_modified = True
        self._decide(trk, 'accepted_by_user', label=1)
        self._refresh_canvas()

    def confirm_rejection(self, tid):
        """Agree with the judges: the rejected track was not real."""
        trk = self.track_by_id(tid)
        if trk and trk['route'] == 'rejected_by_judge':
            self._clear_preview()
            trk['confirmed_reject'] = True
            self._decide(trk, 'rejected_by_judge', label=0)

    def retrain_judge(self):
        try:
            s = train_learned_judge()
        except ValueError as exc:
            QMessageBox.information(self.mw, 'Learned judge', f'Not trained yet: {exc}')
            return
        top = ', '.join(f'{k} ({w:+.2f})' for k, w in s['top_features'][:4])
        QMessageBox.information(self.mw, 'Learned judge',
                                f"Trained on {s['rows']} decisions ({s['real']} real).\n"
                                f"Cross-validated AUC: {s['cv_auc']:.3f} (1.0 = perfect, 0.5 = guessing).\n"
                                f"Most influential: {top}\n\nIt is used from the next Autopilot run.")

    def accept_unflagged(self):
        if not self.result or not self._matches_current():
            return
        self.mw.update_frame_annotations()
        n = 0
        for trk in self.result['tracks']:
            if trk['route'] == 'accept':
                for _f, a in self._track_boxes(trk):
                    a.verified = True
                trk['verified'] = True
                n += 1
        self.mw.project_modified = True
        self._save()
        self.mw.canvas.update()
        self.mw.statusBar.showMessage(f'Autopilot: marked {n} unflagged tracks as verified.', 6000)

    # ----------------------------------------------------------- possible misses
    def _miss(self, mid):
        for m in (self.result or {}).get('misses', []):
            if m['id'] == mid:
                return m
        return None

    def show_miss(self, mid, frame=None):
        m = self._miss(mid)
        if not m or not self._matches_current(warn=False):
            return
        self._clear_preview()
        target = int(frame if frame is not None else m.get('frame', m['start']))
        rows = list(m.get('boxes', []))
        if rows and not any(int(row[0]) == target for row in rows):
            target = int(min(rows, key=lambda row: abs(int(row[0]) - target))[0])
        self.go_to_frame(target)
        self._clear_preview()
        overlays = []
        for row in rows:
            if int(row[0]) != target:
                continue
            r = [row[0], row[1], row[2], row[3], row[4], row[5], 'd']
            ann = self._make_box(r, m['class'], f"miss-{m['id']}", verified=False)
            if ann is not None:
                ann.is_autopilot_preview = True
                ann.preview_kind = 'miss'
                ann.color = QColor(255, 170, 0)
                overlays.append(ann)
        canvas = getattr(self.mw, 'canvas', None)
        if canvas is not None:
            canvas.preview_annotations = overlays
            canvas.update()

    def step_miss(self, mid, direction):
        """Step backward or forward through frames where a candidate miss was detected."""
        m = self._miss(mid)
        if not m:
            return
        rows = m.get('boxes', [])
        if not rows:
            return
        frames = sorted(set(int(r[0]) for r in rows))
        cur = self.mw.current_frame
        if cur not in frames:
            target = frames[0] if direction > 0 else frames[-1]
        else:
            idx = frames.index(cur)
            next_idx = min(len(frames) - 1, max(0, idx + direction))
            target = frames[next_idx]
        self.show_miss(mid, target)

    def resolve_miss(self, mid, add):
        m = self._miss(mid)
        if not m or not self._matches_current():
            return
        if not add:
            m['status'] = 'dismissed'
            self._save()
            return
        mw = self.mw
        mw.update_frame_annotations()
        mw.save_undo_state(range(m['start'], m['end'] + 1))
        rows = [[r[0], r[1], r[2], r[3], r[4], r[5], 'd'] for r in m['boxes']]
        filled = []
        for a, b in zip(rows[:-1], rows[1:]):  # bridge short holes linearly
            filled.append(a)
            gap = b[0] - a[0]
            if 1 < gap <= 10:
                for k in range(1, gap):
                    t = k / gap
                    filled.append([a[0] + k] + [a[i] * (1 - t) + b[i] * t for i in range(1, 5)] + [min(a[5], b[5]), 'i'])
        filled.append(rows[-1])
        self._ensure_class(m['class'])
        trk = {'id': max([t['id'] for t in self.result['tracks']] + [0]) + 1, 'label_id': self._next_label_id(),
               'class': m['class'], 'class_margin': 1.0, 'class_shares': {m['class']: 1.0},
               'route': 'accepted_by_user', 'risk': 0.0, 'focus_frame': m['frame'], 'reasons': [],
               'start': m['start'], 'end': m['end'], 'shot': -1, 'n_det': len(rows), 'links': 0,
               'ext_back': 0, 'ext_fwd': 0, 'mean_score': m['mean_score'], 'boxes': filled, 'from_miss': mid}
        for row in filled:
            ann = self._make_box(row, trk['class'], trk['label_id'], verified=True)
            if ann is not None:
                mw.frame_annotations.setdefault(int(row[0]), []).append(ann)
        self.result['tracks'].append(trk)
        m['status'] = 'added'
        mw.project_modified = True
        self._save()
        self._refresh_canvas()

    # ------------------------------------------------------------------- audit
    def audit_decisions(self):
        return (self.result or {}).get('audit', {}).get('decisions', {})

    def mark_audit(self, frame, mark):
        if not self.result:
            return
        self.result['audit'].setdefault('decisions', {})[str(int(frame))] = mark
        self._save()

    def extend_audit(self, n):
        if not self.result:
            return
        audit = self.result['audit']
        frames = list(range(self.result['start_frame'], self.result['end_frame'] + 1))
        new = sample_frames(frames, n, seed=len(audit['frames']) + 1, exclude=audit['frames'])
        audit['frames'] = audit['frames'] + new
        self._save()
