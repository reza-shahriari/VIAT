"""Review panel for Autopilot results: flagged tracks, possible misses, random-frame audit."""
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QColor, QKeySequence
from PyQt5.QtWidgets import (QDockWidget, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QListWidget,
                             QListWidgetItem, QPushButton, QTabWidget, QCheckBox, QComboBox,
                             QDoubleSpinBox, QShortcut, QSpinBox)

from ..utils.autopilot import upper_bound, samples_needed

ROLE_ID = Qt.UserRole
ROUTE_COLORS = {'review': QColor(230, 140, 40), 'accept': QColor(80, 170, 80),
                'accepted_by_user': QColor(60, 140, 220), 'deleted_by_user': QColor(150, 150, 150)}


class AutopilotReviewDock(QDockWidget):
    def __init__(self, controller, parent=None):
        super().__init__('Autopilot Review', parent)
        self.controller = controller
        self.setAllowedAreas(Qt.LeftDockWidgetArea | Qt.RightDockWidgetArea)
        self.setObjectName('AutopilotReviewDock')
        root = QWidget()
        lay = QVBoxLayout(root)
        self.summary = QLabel('No Autopilot result loaded.')
        self.summary.setWordWrap(True)
        lay.addWidget(self.summary)
        self.tabs = QTabWidget()
        lay.addWidget(self.tabs)
        self.tabs.addTab(self._build_tracks_tab(), 'Flagged tracks')
        self.tabs.addTab(self._build_misses_tab(), 'Possible misses')
        self.tabs.addTab(self._build_rejected_tab(), 'Auto-rejected')
        self.tabs.addTab(self._build_audit_tab(), 'Random audit')
        foot = QHBoxLayout()
        b_reload = QPushButton('Load result for this video')
        b_reload.clicked.connect(self.controller.load_saved_result)
        b_reapply = QPushButton('Re-apply to project')
        b_reapply.setToolTip('Write the result boxes into the project again (replaces earlier Autopilot boxes).')
        b_reapply.clicked.connect(self.controller.reapply_result)
        b_train = QPushButton('Retrain judge')
        b_train.setToolTip('Train the learned judge from every Accept/Delete/Restore decision so far')
        b_train.clicked.connect(self.controller.retrain_judge)
        foot.addWidget(b_reload)
        foot.addWidget(b_reapply)
        foot.addWidget(b_train)
        lay.addLayout(foot)
        self.setWidget(root)

    # ------------------------------------------------------------------- tracks
    def _build_tracks_tab(self):
        w = QWidget()
        lay = QVBoxLayout(w)
        self.show_all = QCheckBox('Show every track (not only flagged)')
        self.show_all.toggled.connect(self.refresh)
        lay.addWidget(self.show_all)
        self.track_list = QListWidget()
        self.track_list.currentItemChanged.connect(lambda cur, _prev: self._show_track(cur))
        lay.addWidget(self.track_list)
        self.reason_label = QLabel('')
        self.reason_label.setWordWrap(True)
        lay.addWidget(self.reason_label)
        nav = QHBoxLayout()
        b_prev, b_next = QPushButton('<< Step back'), QPushButton('Step forward >>')
        b_prev.setToolTip('Jump half a second back along the selected track')
        b_next.setToolTip('Jump half a second forward along the selected track')
        b_prev.clicked.connect(lambda: self._step(-1))
        b_next.clicked.connect(lambda: self._step(1))
        nav.addWidget(b_prev)
        nav.addWidget(b_next)
        lay.addLayout(nav)
        act = QHBoxLayout()
        b_ok = QPushButton('Accept track (A)')
        b_del = QPushButton('Delete track (D)')
        b_ok.clicked.connect(lambda: self._track_action('accept'))
        b_del.clicked.connect(lambda: self._track_action('delete'))
        act.addWidget(b_ok)
        act.addWidget(b_del)
        lay.addLayout(act)
        rel = QHBoxLayout()
        self.relabel_combo = QComboBox()
        b_fill = QPushButton('Fill gaps')
        b_fill.setToolTip('Interpolate every frame the track skips (e.g. a close-up the detector missed)')
        b_fill.clicked.connect(lambda: self._track_action('fill'))
        act.addWidget(b_fill)
        b_rel = QPushButton('Relabel track')
        b_rel.clicked.connect(lambda: self._track_action('relabel'))
        rel.addWidget(self.relabel_combo, 1)
        rel.addWidget(b_rel)
        lay.addLayout(rel)
        b_accept_rest = QPushButton('Accept all unflagged tracks')
        b_accept_rest.setToolTip('Mark every low-risk track as verified (do this after the audit passes).')
        b_accept_rest.clicked.connect(self.controller.accept_unflagged)
        lay.addWidget(b_accept_rest)
        for key, action in (('A', 'accept'), ('D', 'delete')):
            sc = QShortcut(QKeySequence(key), w)
            sc.setContext(Qt.WidgetWithChildrenShortcut)
            sc.activated.connect(lambda a=action: self._track_action(a))
        return w

    def _selected_track(self):
        item = self.track_list.currentItem()
        return item.data(ROLE_ID) if item else None

    def _show_track(self, item):
        if not item:
            self.reason_label.setText('')
            return
        trk = self.controller.track_by_id(item.data(ROLE_ID))
        if not trk:
            return
        lines = [f"Track #{trk['label_id']} {trk['class']}  frames {trk['start'] + 1}-{trk['end'] + 1}, "
                 f"{trk['n_det']} detections, avg score {trk['mean_score']:.2f}, risk {trk['risk']:.2f}"]
        for r in trk['reasons']:
            lines.append(f"- {r['text']}")
        self.reason_label.setText('\n'.join(lines))
        self.controller.show_track(trk['id'], trk.get('focus_frame', trk['start']))

    def _step(self, direction):
        tid = self._selected_track()
        if tid is not None:
            self.controller.step_track(tid, direction)

    def _track_action(self, action):
        tid = self._selected_track()
        if tid is None:
            return
        row = self.track_list.currentRow()
        if action == 'accept':
            self.controller.accept_track(tid)
        elif action == 'delete':
            self.controller.delete_track(tid)
        elif action == 'fill':
            self.controller.fill_track_gaps(tid)
        elif action == 'relabel' and self.relabel_combo.currentText():
            self.controller.relabel_track(tid, self.relabel_combo.currentText())
        self.refresh()
        if self.track_list.count():
            self.track_list.setCurrentRow(min(row, self.track_list.count() - 1))

    # ------------------------------------------------------------------- misses
    def _build_misses_tab(self):
        w = QWidget()
        lay = QVBoxLayout(w)
        info = QLabel('Objects the detectors kept seeing with medium confidence but that never became a track. '
                      'Add the real ones; dismiss the rest.')
        info.setWordWrap(True)
        lay.addWidget(info)
        self.miss_list = QListWidget()
        self.miss_list.currentItemChanged.connect(
            lambda cur, _p: cur and self.controller.show_miss(cur.data(ROLE_ID)))
        lay.addWidget(self.miss_list)
        row = QHBoxLayout()
        b_add, b_dis = QPushButton('Add as track'), QPushButton('Dismiss')
        b_add.clicked.connect(lambda: self._miss_action(True))
        b_dis.clicked.connect(lambda: self._miss_action(False))
        row.addWidget(b_add)
        row.addWidget(b_dis)
        lay.addLayout(row)
        return w

    def _miss_action(self, add):
        item = self.miss_list.currentItem()
        if not item:
            return
        row = self.miss_list.currentRow()
        self.controller.resolve_miss(item.data(ROLE_ID), add)
        self.refresh()
        if self.miss_list.count():
            self.miss_list.setCurrentRow(min(row, self.miss_list.count() - 1))

    # ----------------------------------------------------------------- rejected
    def _build_rejected_tab(self):
        w = QWidget()
        lay = QVBoxLayout(w)
        info = QLabel('Tracks the judges rejected automatically. They are not in the project; '
                      'restore any that are real (this also teaches the learned judge).')
        info.setWordWrap(True)
        lay.addWidget(info)
        self.rejected_list = QListWidget()
        self.rejected_list.currentItemChanged.connect(lambda cur, _p: self._show_rejected(cur))
        lay.addWidget(self.rejected_list)
        self.rejected_reason = QLabel('')
        self.rejected_reason.setWordWrap(True)
        lay.addWidget(self.rejected_reason)
        row = QHBoxLayout()
        b_restore, b_ok = QPushButton('Restore (it is real)'), QPushButton('Correctly rejected')
        b_restore.clicked.connect(lambda: self._rejected_action(True))
        b_ok.clicked.connect(lambda: self._rejected_action(False))
        row.addWidget(b_restore)
        row.addWidget(b_ok)
        lay.addLayout(row)
        return w

    def _show_rejected(self, item):
        if not item:
            self.rejected_reason.setText('')
            return
        trk = self.controller.track_by_id(item.data(ROLE_ID))
        if not trk:
            return
        self.rejected_reason.setText('\n'.join(f"- {r['text']}" for r in trk['reasons'][:5]))
        self.controller.go_to_frame(trk.get('focus_frame', trk['start']))

    def _rejected_action(self, restore):
        item = self.rejected_list.currentItem()
        if not item:
            return
        row = self.rejected_list.currentRow()
        if restore:
            self.controller.restore_track(item.data(ROLE_ID))
        else:
            self.controller.confirm_rejection(item.data(ROLE_ID))
        self.refresh()
        if self.rejected_list.count():
            self.rejected_list.setCurrentRow(min(row, self.rejected_list.count() - 1))

    # -------------------------------------------------------------------- audit
    def _build_audit_tab(self):
        w = QWidget()
        lay = QVBoxLayout(w)
        info = QLabel('Check these randomly chosen frames. Mark a frame "Has error" if any box is wrong or '
                      'missing (then fix it). The result is a proven upper bound on how many frames are wrong.')
        info.setWordWrap(True)
        lay.addWidget(info)
        self.audit_list = QListWidget()
        self.audit_list.currentItemChanged.connect(
            lambda cur, _p: cur and self.controller.go_to_frame(cur.data(ROLE_ID)))
        lay.addWidget(self.audit_list)
        row = QHBoxLayout()
        b_ok, b_err = QPushButton('Correct (Y)'), QPushButton('Has error (N)')
        b_ok.clicked.connect(lambda: self._audit_mark('ok'))
        b_err.clicked.connect(lambda: self._audit_mark('error'))
        row.addWidget(b_ok)
        row.addWidget(b_err)
        lay.addLayout(row)
        tgt = QHBoxLayout()
        tgt.addWidget(QLabel('Target error rate:'))
        self.target = QDoubleSpinBox()
        self.target.setRange(0.1, 50.0)
        self.target.setSuffix(' %')
        self.target.setValue(5.0)
        self.target.valueChanged.connect(self._update_audit_stats)
        tgt.addWidget(self.target)
        self.more_n = QSpinBox()
        self.more_n.setRange(1, 1000)
        self.more_n.setValue(30)
        b_more = QPushButton('Add random frames')
        b_more.clicked.connect(lambda: (self.controller.extend_audit(self.more_n.value()), self.refresh()))
        tgt.addWidget(self.more_n)
        tgt.addWidget(b_more)
        lay.addLayout(tgt)
        self.audit_stats = QLabel('')
        self.audit_stats.setWordWrap(True)
        lay.addWidget(self.audit_stats)
        for key, mark in (('Y', 'ok'), ('N', 'error')):
            sc = QShortcut(QKeySequence(key), w)
            sc.setContext(Qt.WidgetWithChildrenShortcut)
            sc.activated.connect(lambda m=mark: self._audit_mark(m))
        return w

    def _audit_mark(self, mark):
        item = self.audit_list.currentItem()
        if not item:
            return
        self.controller.mark_audit(item.data(ROLE_ID), mark)
        self.refresh()
        decisions = self.controller.audit_decisions()
        for i in range(self.audit_list.count()):  # jump to the next unchecked frame
            if str(self.audit_list.item(i).data(ROLE_ID)) not in decisions:
                self.audit_list.setCurrentRow(i)
                return

    def _update_audit_stats(self):
        decisions = self.controller.audit_decisions()
        n = len(decisions)
        k = sum(1 for v in decisions.values() if v == 'error')
        target = self.target.value() / 100.0
        if n == 0:
            need = samples_needed(target, 0)
            self.audit_stats.setText(f'No frames checked yet. {need} error-free frames would prove '
                                     f'<= {target:.1%} of frames are wrong.')
            return
        ub = upper_bound(k, n)
        text = (f'Checked {n} frames, {k} with errors (observed {k / n:.1%}).\n'
                f'With 95% confidence at most {ub:.1%} of all frames contain an error.')
        if ub <= target:
            text += f'\nTarget <= {target:.1%} reached.'
        else:
            need = samples_needed(target, k)
            if need is None or k / n >= target:
                text += f'\nThe error rate looks above {target:.1%}: fix the flagged tracks or re-run with higher quality.'
            else:
                text += f'\nCheck about {need - n} more random frames (with no new errors) to prove <= {target:.1%}.'
        self.audit_stats.setText(text)

    # ------------------------------------------------------------------ refresh
    def refresh(self):
        res = self.controller.result
        self.track_list.blockSignals(True)
        self.miss_list.blockSignals(True)
        self.audit_list.blockSignals(True)
        self.rejected_list.blockSignals(True)
        self.rejected_list.clear()
        cur_track, cur_audit = self._selected_track(), self.audit_list.currentItem()
        cur_audit = cur_audit.data(ROLE_ID) if cur_audit else None
        self.track_list.clear()
        self.miss_list.clear()
        self.audit_list.clear()
        if not res:
            self.summary.setText('No Autopilot result loaded. Run Tools > Autopilot first.')
        else:
            s = res['stats']
            open_review = sum(1 for t in res['tracks'] if t['route'] == 'review')
            open_miss = sum(1 for m in res['misses'] if m.get('status') == 'open')
            self.summary.setText(
                f"{s['tracks']} tracks, {s['boxes']} boxes over {s['frames']} frames. "
                f"To review: {open_review} tracks, {open_miss} possible misses. "
                f"Auto-rejected by judges: {s.get('judge_rejected', 0)}; filtered as noise: {s['rejected']}.")
            tracks = [t for t in res['tracks'] if t['route'] != 'rejected_by_judge' and
                      (self.show_all.isChecked() or t['route'] == 'review')]
            for t in sorted(tracks, key=lambda t: -t['risk']):
                first = t['reasons'][0]['text'] if t['reasons'] else 'no issues found'
                item = QListWidgetItem(f"#{t['label_id']} {t['class']}  risk {t['risk']:.2f}  "
                                       f"[{t['start'] + 1}-{t['end'] + 1}]  {first}")
                item.setData(ROLE_ID, t['id'])
                item.setForeground(ROUTE_COLORS.get(t['route'], QColor(0, 0, 0)))
                self.track_list.addItem(item)
                if t['id'] == cur_track:
                    self.track_list.setCurrentItem(item)
            for t in res['tracks']:
                if t['route'] != 'rejected_by_judge' or t.get('confirmed_reject'):
                    continue
                box = t['boxes'][len(t['boxes']) // 2]
                item = QListWidgetItem(f"#{t['id']} {t['class']}  judge p {t.get('judge_p', 0):.2f}  "
                                       f"[{t['start'] + 1}-{t['end'] + 1}]  ~{int(box[3] - box[1])}x{int(box[4] - box[2])} px")
                item.setData(ROLE_ID, t['id'])
                self.rejected_list.addItem(item)
            for m in res['misses']:
                if m.get('status') != 'open':
                    continue
                item = QListWidgetItem(f"{m['class']}  score {m['mean_score']:.2f}  "
                                       f"[{m['start'] + 1}-{m['end'] + 1}]  {len(m['boxes'])} frames")
                item.setData(ROLE_ID, m['id'])
                self.miss_list.addItem(item)
            decisions = res['audit'].get('decisions', {})
            for f in res['audit'].get('frames', []):
                d = decisions.get(str(f))
                mark = {'ok': 'OK', 'error': 'ERROR'}.get(d, '...')
                item = QListWidgetItem(f"Frame {f + 1}   {mark}")
                item.setData(ROLE_ID, f)
                if d == 'error':
                    item.setForeground(QColor(210, 60, 60))
                elif d == 'ok':
                    item.setForeground(QColor(60, 150, 60))
                self.audit_list.addItem(item)
                if f == cur_audit:
                    self.audit_list.setCurrentItem(item)
            names = self.controller.class_names()
            current = self.relabel_combo.currentText()
            self.relabel_combo.clear()
            self.relabel_combo.addItems(names)
            if current in names:
                self.relabel_combo.setCurrentText(current)
        self.track_list.blockSignals(False)
        self.miss_list.blockSignals(False)
        self.audit_list.blockSignals(False)
        self.rejected_list.blockSignals(False)
        self.tabs.setTabText(2, f"Auto-rejected ({self.rejected_list.count()})")
        self.tabs.setTabText(0, f"Flagged tracks ({self.track_list.count()})")
        self.tabs.setTabText(1, f"Possible misses ({self.miss_list.count()})")
        self._update_audit_stats()
