import cv2
import numpy as np
from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QDoubleSpinBox, QSpinBox,
    QPushButton, QProgressBar, QMessageBox, QFormLayout
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
import traceback

class SceneDetectWorker(QThread):
    progress_updated = pyqtSignal(int, int)
    finished_processing = pyqtSignal(list, list) # video_groups list, single_images list
    error_occurred = pyqtSignal(str)

    def __init__(self, image_files, threshold, min_scene_len):
        super().__init__()
        self.image_files = image_files
        self.threshold = threshold
        self.min_scene_len = min_scene_len
        self.is_cancelled = False

    def run(self):
        try:
            from scenedetect.detectors import ContentDetector
            
            # Use PySceneDetect's FrameTimecode and ContentDetector
            try:
                from scenedetect import FrameTimecode
            except ImportError:
                from scenedetect.frame_timecode import FrameTimecode
                
            detector = ContentDetector(threshold=self.threshold, min_scene_len=self.min_scene_len)
            fps = 30.0
            
            cuts = [0]
            total_frames = len(self.image_files)
            
            for i, path in enumerate(self.image_files):
                if self.is_cancelled:
                    return
                    
                frame = cv2.imread(path)
                if frame is not None:
                    height, width = frame.shape[:2]
                    scale = 256.0 / max(width, height)
                    if scale < 1.0:
                        small = cv2.resize(frame, (int(width * scale), int(height * scale)))
                    else:
                        small = frame
                        
                    tc = FrameTimecode(timecode=i, fps=fps)
                    detected_cuts = detector.process_frame(tc, small)
                    
                    if detected_cuts:
                        cuts.append(i)
                
                if i % 10 == 0:
                    self.progress_updated.emit(i + 1, total_frames)
                    
            cuts.append(total_frames) # End boundary
            
            video_groups = {}
            single_images = []
            
            scene_idx = 1
            for i in range(len(cuts) - 1):
                start_idx = cuts[i]
                end_idx = cuts[i+1]
                length = end_idx - start_idx
                
                if length >= 2:
                    video_groups[f"AutoScene_{scene_idx}"] = list(range(start_idx, end_idx))
                    scene_idx += 1
                elif length == 1:
                    single_images.append(start_idx)
                    
            self.progress_updated.emit(total_frames, total_frames)
            self.finished_processing.emit([video_groups], single_images)
            
        except Exception as e:
            self.error_occurred.emit(f"Error: {str(e)}\n{traceback.format_exc()}")

class VideoSceneDetectWorker(QThread):
    progress_updated = pyqtSignal(int, int)
    finished_processing = pyqtSignal(list, list) # video_groups list, single_images list
    error_occurred = pyqtSignal(str)

    def __init__(self, video_path, threshold=27.0, min_scene_len=15):
        super().__init__()
        self.video_path = video_path
        self.threshold = threshold
        self.min_scene_len = min_scene_len
        self.is_cancelled = False

    def run(self):
        try:
            from scenedetect import open_video, SceneManager
            from scenedetect.detectors import ContentDetector
            
            video = open_video(self.video_path)
            scene_manager = SceneManager()
            scene_manager.add_detector(ContentDetector(threshold=self.threshold, min_scene_len=self.min_scene_len))
            
            # Start detection
            scene_manager.detect_scenes(video, show_progress=False)
            
            if self.is_cancelled:
                return
                
            scene_list = scene_manager.get_scene_list()
            
            video_groups = {}
            single_images = []
            
            if not scene_list:
                # Get total frames somehow, fallback to 1, usually video_manager has it anyway, but we just want to say scene_1 is whole thing
                video_groups["AutoScene_1"] = [] # Empty list handled later, or just return empty
                self.finished_processing.emit([video_groups], [])
                return
            
            scene_idx = 1
            for i, scene in enumerate(scene_list):
                start_idx = scene[0].get_frames()
                end_idx = scene[1].get_frames()
                
                length = end_idx - start_idx
                
                if length >= 2:
                    video_groups[f"AutoScene_{scene_idx}"] = list(range(start_idx, end_idx))
                    scene_idx += 1
                elif length == 1:
                    single_images.append(start_idx)
            
            self.progress_updated.emit(100, 100)
            self.finished_processing.emit([video_groups], single_images)
            
        except Exception as e:
            self.error_occurred.emit(f"Error: {str(e)}\n{traceback.format_exc()}")


class SceneDetectDialog(QDialog):
    def __init__(self, parent=None, image_files=None, video_path=None):
        super().__init__(parent)
        self.image_files = image_files or []
        self.video_path = video_path
        self.video_groups = {}
        self.single_images = {}
        self.setup_ui()

    def setup_ui(self):
        self.setWindowTitle("Auto-group by Scene Cuts")
        self.setMinimumWidth(450)
        
        layout = QVBoxLayout(self)
        
        info_label = QLabel(
            "This uses <b>PySceneDetect</b> to analyze consecutive frames for visual cuts.<br>"
            "Frames with a visual difference above the threshold will start a new video group.<br><br>"
            "- <b>Threshold:</b> Lower = more cuts. Default is 27.0.<br>"
            "- <b>Min Scene Length:</b> Minimum frames per scene. Prevents cutting every few frames."
        )
        info_label.setWordWrap(True)
        layout.addWidget(info_label)
        
        # Settings
        form_layout = QFormLayout()
        
        self.spin_threshold = QDoubleSpinBox()
        self.spin_threshold.setRange(1.0, 100.0)
        self.spin_threshold.setValue(27.0)
        self.spin_threshold.setSingleStep(1.0)
        form_layout.addRow("Cut Threshold:", self.spin_threshold)
        
        self.spin_min_len = QSpinBox()
        self.spin_min_len.setRange(1, 1000)
        self.spin_min_len.setValue(15)
        form_layout.addRow("Min Scene Length (frames):", self.spin_min_len)
        
        layout.addLayout(form_layout)
        
        # Progress
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        layout.addWidget(self.progress_bar)
        
        # Buttons
        btn_layout = QHBoxLayout()
        self.btn_start = QPushButton("Start Detection")
        self.btn_start.clicked.connect(self.start_detection)
        
        self.btn_cancel = QPushButton("Cancel")
        self.btn_cancel.clicked.connect(self.cancel)
        
        btn_layout.addWidget(self.btn_start)
        btn_layout.addWidget(self.btn_cancel)
        layout.addLayout(btn_layout)
        
        self.worker = None

    def start_detection(self):
        if not self.image_files and not self.video_path:
            QMessageBox.warning(self, "Error", "No images or video to process.")
            return
            
        self.btn_start.setEnabled(False)
        self.spin_threshold.setEnabled(False)
        self.spin_min_len.setEnabled(False)
        
        if self.video_path:
            self.worker = VideoSceneDetectWorker(self.video_path, self.spin_threshold.value(), self.spin_min_len.value())
        else:
            self.worker = SceneDetectWorker(self.image_files, self.spin_threshold.value(), self.spin_min_len.value())
            
        self.worker.progress_updated.connect(self.update_progress)
        self.worker.finished_processing.connect(self.on_finished)
        self.worker.error_occurred.connect(self.on_error)
        self.worker.start()

    def update_progress(self, current, total):
        self.progress_bar.setRange(0, total)
        self.progress_bar.setValue(current)

    def on_finished(self, vg_list, single_images):
        self.video_groups = vg_list[0]
        self.single_images = single_images
        self.accept()

    def on_error(self, error_msg):
        QMessageBox.critical(self, "Error", f"An error occurred:\n{error_msg}")
        self.btn_start.setEnabled(True)
        self.spin_threshold.setEnabled(True)
        self.spin_min_len.setEnabled(True)

    def cancel(self):
        if self.worker and self.worker.isRunning():
            self.worker.is_cancelled = True
            self.worker.wait()
        self.reject()
