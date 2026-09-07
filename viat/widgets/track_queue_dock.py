import json
from PyQt5.QtWidgets import (
    QDockWidget, QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QListWidget, QListWidgetItem, QPushButton, QPlainTextEdit,
    QMessageBox, QFileDialog, QAbstractItemView
)
from PyQt5.QtCore import Qt, pyqtSignal

STATUS_COLORS = {
    'pending': '#888888',
    'running': '#2196F3',
    'done': '#4CAF50',
    'failed': '#F44336',
    'lost': '#FF9800',
    'stopped': '#9E9E9E',
}


class TrackQueueDock(QDockWidget):
    """Dock widget for building and running a batch queue of SAM tracking
    jobs: add the current prompt as a job (without running it), stack up
    jobs across many videos over the course of a session, then run them
    all sequentially (e.g. overnight) with 'Run Queue'."""

    add_current_requested = pyqtSignal()
    run_queue_requested = pyqtSignal()
    stop_queue_requested = pyqtSignal()
    remove_job_requested = pyqtSignal(str)      # job_id
    move_job_requested = pyqtSignal(str, int)   # job_id, direction (-1 up, +1 down)
    clear_queue_requested = pyqtSignal()
    save_queue_requested = pyqtSignal()
    load_queue_requested = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__("Track Queue", parent)
        self.setAllowedAreas(Qt.LeftDockWidgetArea | Qt.RightDockWidgetArea)
        self.setup_ui()

    def setup_ui(self):
        self.widget = QWidget()
        layout = QVBoxLayout(self.widget)

        layout.addWidget(QLabel("Queued Jobs:"))
        self.list_widget = QListWidget()
        self.list_widget.setSelectionMode(QAbstractItemView.SingleSelection)
        self.list_widget.setToolTip(
            "Each row is one tracking job: a prompt on a specific video/range.\n"
            "Add prompts here as you go, then Run Queue to process them all in sequence."
        )
        layout.addWidget(self.list_widget)

        row1 = QHBoxLayout()
        self.btn_add_current = QPushButton("+ Add Current Prompt")
        self.btn_add_current.setToolTip(
            "Capture the current SAM prompt (points/box/text), class, model, and scope\n"
            "as a queued job. Does not run it immediately."
        )
        self.btn_add_current.clicked.connect(self.add_current_requested.emit)
        row1.addWidget(self.btn_add_current)
        layout.addLayout(row1)

        row2 = QHBoxLayout()
        self.btn_remove = QPushButton("Remove")
        self.btn_remove.clicked.connect(self._on_remove_clicked)
        self.btn_up = QPushButton("Move Up")
        self.btn_up.clicked.connect(lambda: self._on_move_clicked(-1))
        self.btn_down = QPushButton("Move Down")
        self.btn_down.clicked.connect(lambda: self._on_move_clicked(1))
        row2.addWidget(self.btn_remove)
        row2.addWidget(self.btn_up)
        row2.addWidget(self.btn_down)
        layout.addLayout(row2)

        row3 = QHBoxLayout()
        self.btn_clear = QPushButton("Clear All")
        self.btn_clear.clicked.connect(self._on_clear_clicked)
        self.btn_save = QPushButton("Save Queue")
        self.btn_save.clicked.connect(self.save_queue_requested.emit)
        self.btn_load = QPushButton("Load Queue")
        self.btn_load.clicked.connect(self.load_queue_requested.emit)
        row3.addWidget(self.btn_clear)
        row3.addWidget(self.btn_save)
        row3.addWidget(self.btn_load)
        layout.addLayout(row3)

        row4 = QHBoxLayout()
        self.btn_run = QPushButton("\u25b6 Run Queue")
        self.btn_run.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold;")
        self.btn_run.clicked.connect(self.run_queue_requested.emit)
        self.btn_stop = QPushButton("\u25a0 Stop After Current Job")
        self.btn_stop.clicked.connect(self.stop_queue_requested.emit)
        row4.addWidget(self.btn_run)
        row4.addWidget(self.btn_stop)
        layout.addLayout(row4)

        layout.addWidget(QLabel("Batch Log:"))
        self.log_view = QPlainTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setMaximumBlockCount(2000)
        layout.addWidget(self.log_view)

        self.setWidget(self.widget)

    # -- population / status -------------------------------------------------
    def set_jobs(self, jobs):
        """jobs: list of dicts each with at least 'job_id', 'display', 'status'."""
        self.list_widget.clear()
        for job in jobs:
            item = QListWidgetItem(self._format_job(job))
            item.setData(Qt.UserRole, job.get('job_id'))
            item.setToolTip(job.get('display', ''))
            self.list_widget.addItem(item)

    def update_job_status(self, job_id, status):
        for i in range(self.list_widget.count()):
            item = self.list_widget.item(i)
            if item.data(Qt.UserRole) == job_id:
                text = item.text().split(' \u2014 ')[0]
                item.setText(f"{text} \u2014 {status.upper()}")
                return

    def append_log(self, job_id, message):
        prefix = f"[{job_id[:8]}] " if job_id else ""
        self.log_view.appendPlainText(f"{prefix}{message}")

    def _format_job(self, job):
        base = job.get('display', 'Job')
        status = job.get('status', 'pending')
        return f"{base} \u2014 {status.upper()}"

    # -- selection helpers -----------------------------------------------
    def _selected_job_id(self):
        item = self.list_widget.currentItem()
        if not item:
            return None
        return item.data(Qt.UserRole)

    def _on_remove_clicked(self):
        job_id = self._selected_job_id()
        if job_id:
            self.remove_job_requested.emit(job_id)

    def _on_move_clicked(self, direction):
        job_id = self._selected_job_id()
        if job_id:
            self.move_job_requested.emit(job_id, direction)

    def _on_clear_clicked(self):
        reply = QMessageBox.question(
            self, "Clear Queue", "Remove all queued jobs?",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No
        )
        if reply == QMessageBox.Yes:
            self.clear_queue_requested.emit()
