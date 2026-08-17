import os
from PyQt5.QtWidgets import (
    QDockWidget,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QSpinBox,
    QCheckBox,
    QPushButton,
    QMessageBox,
    QFileDialog,
    QGroupBox,
)
from PyQt5.QtCore import Qt


class CropSettingsDock(QDockWidget):
    """Dock widget for managing dataset cropping settings and export."""

    def __init__(self, main_window):
        super().__init__("Crop Settings", main_window)
        self.main_window = main_window
        self.setAllowedAreas(Qt.LeftDockWidgetArea | Qt.RightDockWidgetArea)
        
        self.init_ui()

    def init_ui(self):
        """Initialize the UI components."""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        # Settings Group
        settings_group = QGroupBox("Crop Box Settings")
        settings_layout = QVBoxLayout()
        
        # Fixed Size Inputs
        size_layout = QHBoxLayout()
        
        self.width_spin = QSpinBox()
        self.width_spin.setRange(10, 4000)
        self.width_spin.setValue(640)
        self.width_spin.setToolTip("Crop width in pixels")
        
        self.height_spin = QSpinBox()
        self.height_spin.setRange(10, 4000)
        self.height_spin.setValue(640)
        self.height_spin.setToolTip("Crop height in pixels")
        
        self.apply_size_btn = QPushButton("Apply Size")
        self.apply_size_btn.clicked.connect(self.apply_fixed_size)
        
        size_layout.addWidget(QLabel("W:"))
        size_layout.addWidget(self.width_spin)
        size_layout.addWidget(QLabel("H:"))
        size_layout.addWidget(self.height_spin)
        
        settings_layout.addLayout(size_layout)
        settings_layout.addWidget(self.apply_size_btn)
        
        # Center Box Button
        self.center_btn = QPushButton("Center Crop Box")
        self.center_btn.clicked.connect(self.center_crop_box)
        settings_layout.addWidget(self.center_btn)
        
        # Options
        self.track_object_cb = QCheckBox("Track selected object automatically")
        self.track_object_cb.setChecked(False)
        self.track_object_cb.setToolTip("If enabled, the crop box will follow the selected object's bounding box center across frames.")
        settings_layout.addWidget(self.track_object_cb)
        
        settings_group.setLayout(settings_layout)
        layout.addWidget(settings_group)
        
        # Scope Group
        scope_group = QGroupBox("Crop Box Scope")
        scope_layout = QVBoxLayout()
        
        self.btn_all_frames = QPushButton("Apply to All Frames")
        self.btn_all_frames.clicked.connect(self.apply_to_all_frames)
        
        self.btn_this_frame = QPushButton("Apply to This Frame Only")
        self.btn_this_frame.clicked.connect(self.apply_to_this_frame_only)
        
        self.btn_start_to_here = QPushButton("Apply from Start to Here")
        self.btn_start_to_here.clicked.connect(self.apply_from_start_to_here)
        
        self.btn_here_to_end = QPushButton("Apply from Here to End")
        self.btn_here_to_end.clicked.connect(self.apply_from_here_to_end)
        
        scope_layout.addWidget(self.btn_all_frames)
        scope_layout.addWidget(self.btn_this_frame)
        scope_layout.addWidget(self.btn_start_to_here)
        scope_layout.addWidget(self.btn_here_to_end)
        scope_group.setLayout(scope_layout)
        
        layout.addWidget(scope_group)
        
        # Export Group
        export_group = QGroupBox("Export")
        export_layout = QVBoxLayout()
        
        self.export_mp4_btn = QPushButton("Export as .mp4 Video")
        self.export_mp4_btn.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold;")
        self.export_mp4_btn.clicked.connect(lambda: self.export_crop("mp4"))
        
        self.export_dataset_btn = QPushButton("Export as Image Dataset")
        self.export_dataset_btn.clicked.connect(lambda: self.export_crop("images"))
        
        export_layout.addWidget(self.export_mp4_btn)
        export_layout.addWidget(self.export_dataset_btn)
        
        export_group.setLayout(export_layout)
        layout.addWidget(export_group)

        # Add stretch to push everything to top
        layout.addStretch()
        
        self.setWidget(widget)

    def apply_fixed_size(self):
        """Apply the specified fixed size to the crop box."""
        if not hasattr(self.main_window, "canvas"):
            return
            
        w = self.width_spin.value()
        h = self.height_spin.value()
        
        canvas = self.main_window.canvas
        if canvas.crop_rect is not None:
            # Resize from center
            center = canvas.crop_rect.center()
            canvas.crop_rect.setWidth(w)
            canvas.crop_rect.setHeight(h)
            canvas.crop_rect.moveCenter(center)
            
            # Ensure it's within bounds
            if canvas.pixmap:
                if canvas.crop_rect.left() < 0: canvas.crop_rect.moveLeft(0)
                if canvas.crop_rect.top() < 0: canvas.crop_rect.moveTop(0)
                if canvas.crop_rect.right() > canvas.pixmap.width(): canvas.crop_rect.moveRight(canvas.pixmap.width())
                if canvas.crop_rect.bottom() > canvas.pixmap.height(): canvas.crop_rect.moveBottom(canvas.pixmap.height())
                
            canvas.update()
            canvas.cropRectChanged.emit(canvas.crop_rect)
        else:
            QMessageBox.information(self, "Crop Mode", "Please draw a crop box first.")

    def center_crop_box(self):
        """Center the crop box in the video frame."""
        if not hasattr(self.main_window, "canvas") or not self.main_window.canvas.pixmap:
            return
            
        canvas = self.main_window.canvas
        if canvas.crop_rect is not None:
            cx = canvas.pixmap.width() // 2
            cy = canvas.pixmap.height() // 2
            from PyQt5.QtCore import QPoint
            canvas.crop_rect.moveCenter(QPoint(cx, cy))
            canvas.update()
            
            # Since the signal isn't emitted automatically from direct attribute changes, emit it here
            canvas.cropRectChanged.emit(canvas.crop_rect)
        else:
            QMessageBox.information(self, "Crop Mode", "Please draw a crop box first.")
            
    def apply_to_all_frames(self):
        if not hasattr(self.main_window, "canvas") or self.main_window.canvas.crop_rect is None:
            return
        rect = self.main_window.canvas.crop_rect
        self.main_window.frame_crops.clear()
        from PyQt5.QtCore import QRect
        self.main_window.frame_crops[0] = QRect(rect)
        QMessageBox.information(self, "Crop Scope", "Crop box applied to all frames.")
        
    def apply_to_this_frame_only(self):
        if not hasattr(self.main_window, "canvas") or self.main_window.canvas.crop_rect is None:
            return
        rect = self.main_window.canvas.crop_rect
        cur_frame = self.main_window.current_frame
        
        prev_rect = None
        closest_frame = -1
        for f in self.main_window.frame_crops:
            if f < cur_frame and f > closest_frame:
                closest_frame = f
        if closest_frame >= 0:
            prev_rect = self.main_window.frame_crops[closest_frame]
            
        from PyQt5.QtCore import QRect
        self.main_window.frame_crops[cur_frame] = QRect(rect)
        if prev_rect and (cur_frame + 1) not in self.main_window.frame_crops:
            self.main_window.frame_crops[cur_frame + 1] = QRect(prev_rect)
            
        QMessageBox.information(self, "Crop Scope", "Crop box applied to this frame only.")
        
    def apply_from_start_to_here(self):
        if not hasattr(self.main_window, "canvas") or self.main_window.canvas.crop_rect is None:
            return
        rect = self.main_window.canvas.crop_rect
        cur_frame = self.main_window.current_frame
        
        keys_to_remove = [k for k in self.main_window.frame_crops if k < cur_frame]
        for k in keys_to_remove:
            del self.main_window.frame_crops[k]
            
        from PyQt5.QtCore import QRect
        self.main_window.frame_crops[0] = QRect(rect)
        QMessageBox.information(self, "Crop Scope", "Crop box applied from start to this frame.")
        
    def apply_from_here_to_end(self):
        if not hasattr(self.main_window, "canvas") or self.main_window.canvas.crop_rect is None:
            return
        rect = self.main_window.canvas.crop_rect
        cur_frame = self.main_window.current_frame
        
        keys_to_remove = [k for k in self.main_window.frame_crops if k > cur_frame]
        for k in keys_to_remove:
            del self.main_window.frame_crops[k]
            
        from PyQt5.QtCore import QRect
        self.main_window.frame_crops[cur_frame] = QRect(rect)
        QMessageBox.information(self, "Crop Scope", "Crop box applied from this frame to the end.")

    def export_crop(self, format_type="mp4"):
        """Trigger the export process."""
        if not hasattr(self.main_window, "canvas") or self.main_window.canvas.crop_rect is None:
            QMessageBox.warning(self, "Export Error", "Please draw a crop box first.")
            return
            
        if not hasattr(self.main_window, "video_filename") or not self.main_window.video_filename:
            QMessageBox.warning(self, "Export Error", "Please open a video first.")
            return
            
        output_dir = QFileDialog.getExistingDirectory(self, "Select Export Directory")
        if not output_dir:
            return
            
        # Get track ID if tracking is enabled
        track_id = None
        if self.track_object_cb.isChecked() and self.main_window.canvas.selected_annotation:
            ann = self.main_window.canvas.selected_annotation
            track_id = (getattr(ann, 'attributes', None) or {}).get('actor_id') or (getattr(ann, 'attributes', None) or {}).get('track_id') or ann.id
            
        from viat.utils.crop_exporter import CropExporter
        exporter = CropExporter(
            main_window=self.main_window,
            video_path=self.main_window.video_filename,
            frame_annotations=self.main_window.frame_annotations,
            base_crop_rect=self.main_window.canvas.crop_rect,
            frame_crops=getattr(self.main_window, 'frame_crops', {}),
            track_id=track_id,
            class_colors=self.main_window.canvas.class_colors
        )
        
        success, msg = exporter.export(output_dir, format_type)
        if success:
            QMessageBox.information(self, "Export Complete", msg)
        else:
            QMessageBox.critical(self, "Export Failed", msg)
