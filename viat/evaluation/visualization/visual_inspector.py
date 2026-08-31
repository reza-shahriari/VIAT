import os
import cv2
import numpy as np
from PyQt5.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QComboBox,
    QSlider,
    QSpinBox,
    QSplitter,
    QListWidget,
    QListWidgetItem,
    QFrame
)
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap, QColor, QPen, QPainter, QFont

class VisualInspectorWidget(QWidget):
    """
    Interactive Visual Inspector ("SHOW") widget for side-by-side or overlaid
    Ground Truth vs Detection error browsing.
    """
    frame_selected = pyqtSignal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.gt_boxes = {} # frame_idx -> list of [cls, x, y, w, h]
        self.dt_boxes = {} # frame_idx -> list of [cls, x, y, w, h, conf, status]
        self.image_paths = {} # frame_idx -> img_path
        self.current_frame = 0
        self.max_frame = 0
        self.setup_ui()

    def setup_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(4, 4, 4, 4)

        # Control Bar
        ctrl_bar = QHBoxLayout()
        ctrl_bar.addWidget(QLabel("<b>Filter View:</b>"))

        self.combo_filter = QComboBox()
        self.combo_filter.addItems(["Show All BBoxes", "Only False Positives (FP)", "Only False Negatives (FN)", "Only True Positives (TP)"])
        self.combo_filter.currentIndexChanged.connect(self.update_display)
        ctrl_bar.addWidget(self.combo_filter)

        ctrl_bar.addSpacing(15)
        ctrl_bar.addWidget(QLabel("<b>Class Legend:</b>"))
        legend_lbl = QLabel(
            '<span style="color:#00FF00;">■ GT</span> &nbsp;&nbsp;'
            '<span style="color:#00E5FF;">■ TP</span> &nbsp;&nbsp;'
            '<span style="color:#FF3333;">■ FP</span> &nbsp;&nbsp;'
            '<span style="color:#FF9900;">■ FN</span>'
        )
        ctrl_bar.addWidget(legend_lbl)
        ctrl_bar.addStretch()

        main_layout.addLayout(ctrl_bar)

        # Main Splitter: Canvas Left, Frame List Right
        splitter = QSplitter(Qt.Horizontal)

        # Image Viewer Label
        self.img_label = QLabel()
        self.img_label.setAlignment(Qt.AlignCenter)
        self.img_label.setStyleSheet("background-color: #121212; border: 1px solid rgba(255,255,255,0.1); border-radius: 4px;")
        self.img_label.setMinimumSize(400, 300)
        splitter.addWidget(self.img_label)

        # Sidebar with error frame list
        sidebar = QWidget()
        sidebar_layout = QVBoxLayout(sidebar)
        sidebar_layout.setContentsMargins(2, 2, 2, 2)
        sidebar_layout.addWidget(QLabel("<b>Error Frames List:</b>"))

        self.list_frames = QListWidget()
        self.list_frames.itemClicked.connect(self.on_frame_item_clicked)
        sidebar_layout.addWidget(self.list_frames)

        splitter.addWidget(sidebar)
        splitter.setSizes([600, 200])
        main_layout.addWidget(splitter)

        # Bottom Navigation Controls
        nav_bar = QHBoxLayout()
        self.btn_prev = QPushButton("◄ Prev Frame")
        self.btn_prev.clicked.connect(self.prev_frame)
        nav_bar.addWidget(self.btn_prev)

        self.slider_frame = QSlider(Qt.Horizontal)
        self.slider_frame.setRange(0, 0)
        self.slider_frame.valueChanged.connect(self.set_frame)
        nav_bar.addWidget(self.slider_frame)

        self.spin_frame = QSpinBox()
        self.spin_frame.setRange(0, 0)
        self.spin_frame.valueChanged.connect(self.set_frame)
        nav_bar.addWidget(self.spin_frame)

        self.lbl_frame_count = QLabel("Frame: 0 / 0")
        nav_bar.addWidget(self.lbl_frame_count)

        self.btn_next = QPushButton("Next Frame ►")
        self.btn_next.clicked.connect(self.next_frame)
        nav_bar.addWidget(self.btn_next)

        main_layout.addLayout(nav_bar)

    def load_data(self, image_paths, gt_boxes, dt_boxes):
        """
        Load dataset images and bounding boxes for visualization.
        """
        self.image_paths = image_paths or {}
        self.gt_boxes = gt_boxes or {}
        self.dt_boxes = dt_boxes or {}

        all_indices = sorted(list(set(list(self.image_paths.keys()) + list(self.gt_boxes.keys()) + list(self.dt_boxes.keys()))))
        if all_indices:
            self.max_frame = max(all_indices)
            self.slider_frame.setRange(0, self.max_frame)
            self.spin_frame.setRange(0, self.max_frame)
            self.lbl_frame_count.setText(f"Frame: 0 / {self.max_frame}")

        # Populate error list
        self.list_frames.clear()
        for idx in all_indices:
            dt_list = self.dt_boxes.get(idx, [])
            fps = sum(1 for d in dt_list if len(d) >= 7 and d[6] == 'FP')
            fns = sum(1 for d in dt_list if len(d) >= 7 and d[6] == 'FN')
            if fps > 0 or fns > 0:
                item = QListWidgetItem(f"Frame #{idx}  [FP: {fps}, FN: {fns}]")
                item.setData(Qt.UserRole, idx)
                self.list_frames.addItem(item)

        self.set_frame(0)

    def prev_frame(self):
        if self.current_frame > 0:
            self.set_frame(self.current_frame - 1)

    def next_frame(self):
        if self.current_frame < self.max_frame:
            self.set_frame(self.current_frame + 1)

    def set_frame(self, frame_idx):
        self.current_frame = frame_idx
        self.slider_frame.setValue(frame_idx)
        self.spin_frame.setValue(frame_idx)
        self.lbl_frame_count.setText(f"Frame: {frame_idx} / {self.max_frame}")
        self.update_display()
        self.frame_selected.emit(frame_idx)

    def on_frame_item_clicked(self, item):
        frame_idx = item.data(Qt.UserRole)
        if frame_idx is not None:
            self.set_frame(frame_idx)

    def update_display(self):
        img_path = self.image_paths.get(self.current_frame)
        if img_path and os.path.exists(img_path):
            img = cv2.imread(img_path)
            if img is not None:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            else:
                img = self.create_blank_canvas()
        else:
            img = self.create_blank_canvas()

        h, w, ch = img.shape
        qimg = QImage(img.data, w, h, ch * w, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg)

        # Draw overlays using QPainter
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.Antialiasing)

        filter_mode = self.combo_filter.currentText()

        # Draw GT boxes (Green)
        if "False Positives" not in filter_mode:
            gt_list = self.gt_boxes.get(self.current_frame, [])
            pen_gt = QPen(QColor(0, 255, 0), 2, Qt.SolidLine)
            painter.setPen(pen_gt)
            font = QFont("SansSerif", 9, QFont.Bold)
            painter.setFont(font)

            for box in gt_list:
                if len(box) >= 5:
                    cls_name, x, y, bw, bh = str(box[0]), int(box[1]), int(box[2]), int(box[3]), int(box[4])
                    painter.drawRect(x, y, bw, bh)
                    painter.drawText(x, max(12, y - 4), f"GT: {cls_name}")

        # Draw Detection boxes
        dt_list = self.dt_boxes.get(self.current_frame, [])
        for box in dt_list:
            if len(box) >= 5:
                cls_name, x, y, bw, bh = str(box[0]), int(box[1]), int(box[2]), int(box[3]), int(box[4])
                conf = box[5] if len(box) >= 6 else 1.0
                status = box[6] if len(box) >= 7 else 'TP'

                if "Only False Positives" in filter_mode and status != 'FP':
                    continue
                if "Only False Negatives" in filter_mode and status != 'FN':
                    continue
                if "Only True Positives" in filter_mode and status != 'TP':
                    continue

                if status == 'TP':
                    pen_dt = QPen(QColor(0, 229, 255), 2, Qt.SolidLine)
                elif status == 'FP':
                    pen_dt = QPen(QColor(255, 51, 51), 2, Qt.DashLine)
                else: # FN
                    pen_dt = QPen(QColor(255, 153, 0), 2, Qt.DotLine)

                painter.setPen(pen_dt)
                painter.drawRect(x, y, bw, bh)
                painter.drawText(x, y + bh + 14, f"DT: {cls_name} ({conf:.2f}) [{status}]")

        painter.end()

        # Scale pixmap to fit widget
        scaled_pix = pixmap.scaled(self.img_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.img_label.setPixmap(scaled_pix)

    def create_blank_canvas(self, w=1280, h=720):
        canvas = np.zeros((h, w, 3), dtype=np.uint8) + 30
        cv2.putText(canvas, f"Frame #{self.current_frame} (Canvas Preview)", (w // 4, h // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (200, 200, 200), 2)
        return canvas


from PyQt5.QtWidgets import QDockWidget

class VisualInspectorDock(QDockWidget):
    """
    Dock widget for the Visual Inspector, integrating it directly into the VIAT main window.
    """
    def __init__(self, parent=None):
        super().__init__("Visual Inspector (SHOW)", parent)
        self.setObjectName("VisualInspectorDock")
        self.setAllowedAreas(Qt.LeftDockWidgetArea | Qt.RightDockWidgetArea | Qt.BottomDockWidgetArea)

        self.inspector_widget = VisualInspectorWidget(self)
        self.setWidget(self.inspector_widget)

        if parent:
            self.inspector_widget.frame_selected.connect(parent.set_current_frame)
