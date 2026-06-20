import cv2
from PyQt5.QtWidgets import (
    QDialog,
    QVBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QDialogButtonBox,
    QProgressBar,
    QApplication,
    QMessageBox
)
from PyQt5.QtCore import Qt, QSize
from PyQt5.QtGui import QImage, QPixmap
from viat.utils import create_thumbnail

class ClassInfoDialog(QDialog):
    def __init__(self, main_window, class_name, parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"Class Info: {class_name}")
        self.setMinimumSize(600, 400)
        
        self.main_window = main_window
        self.class_name = class_name
        self.frames_with_class = []
        self.total_instances = 0
        
        self.init_data()
        self.init_ui()
        
    def init_data(self):
        """Scan annotations for the selected class to get counts and frames."""
        self.frames_with_class = []
        self.total_instances = 0
        
        # Iterate over all frame annotations to find instances of the class
        for frame_idx, annotations in self.main_window.frame_annotations.items():
            class_count_in_frame = 0
            for ann in annotations:
                if ann.class_name == self.class_name:
                    class_count_in_frame += 1
                    
            if class_count_in_frame > 0:
                self.total_instances += class_count_in_frame
                self.frames_with_class.append(frame_idx)
                
        # Sort the frame indices
        self.frames_with_class.sort()

    def init_ui(self):
        layout = QVBoxLayout(self)
        
        # Summary label
        summary_text = (f"<b>Class:</b> {self.class_name}<br>"
                        f"<b>Total instances:</b> {self.total_instances}<br>"
                        f"<b>Found in:</b> {len(self.frames_with_class)} images/frames")
        self.info_label = QLabel(summary_text)
        layout.addWidget(self.info_label)
        
        # List widget for thumbnails
        self.thumbnail_list = QListWidget()
        self.thumbnail_list.setViewMode(QListWidget.IconMode)
        self.thumbnail_list.setIconSize(QSize(160, 90))
        self.thumbnail_list.setResizeMode(QListWidget.Adjust)
        self.thumbnail_list.setSpacing(10)
        
        # Handle double click or single click to navigate
        self.thumbnail_list.itemClicked.connect(self.on_thumbnail_clicked)
        
        layout.addWidget(self.thumbnail_list)
        
        # Buttons
        buttons = QDialogButtonBox(QDialogButtonBox.Close)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        
        # Load thumbnails after UI shows up
        from PyQt5.QtCore import QTimer
        QTimer.singleShot(100, self.load_thumbnails)

    def load_thumbnails(self):
        """Generate and load thumbnails for the frames containing the class."""
        if not self.frames_with_class:
            return
            
        # Limit to prevent UI freezing on massive datasets
        max_thumbnails = min(len(self.frames_with_class), 100)
        
        for i in range(max_thumbnails):
            frame_idx = self.frames_with_class[i]
            frame_img = None
            
            # Fetch frame from image dataset
            if getattr(self.main_window, "is_image_dataset", False) and getattr(self.main_window, "image_files", None):
                if frame_idx < len(self.main_window.image_files):
                    img_path = self.main_window.image_files[frame_idx]
                    frame_img = cv2.imread(img_path)
            # Fetch frame from video
            elif self.main_window.cap and self.main_window.cap.isOpened():
                self.main_window.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame_img = self.main_window.cap.read()
            
            if frame_img is not None:
                # Use the create_thumbnail tool function
                thumb_rgb = create_thumbnail(frame_img, (160, 90))
                h, w, c = thumb_rgb.shape
                qimg = QImage(thumb_rgb.data, w, h, w * c, QImage.Format_RGB888)
                pixmap = QPixmap.fromImage(qimg)
                
                # Create item
                item = QListWidgetItem(self.thumbnail_list)
                item.setIcon(pixmap)
                
                # Label item with image name or frame number
                if getattr(self.main_window, "is_image_dataset", False):
                    import os
                    basename = os.path.basename(self.main_window.image_files[frame_idx])
                    # truncate if too long
                    if len(basename) > 15:
                        basename = basename[:12] + "..."
                    item.setText(f"{frame_idx}: {basename}")
                else:
                    item.setText(f"Frame {frame_idx}")
                    
                # Store frame index in user data
                item.setData(Qt.UserRole, frame_idx)
                
            # Keep UI responsive
            QApplication.processEvents()
            
        if len(self.frames_with_class) > max_thumbnails:
            msg = QListWidgetItem(self.thumbnail_list)
            msg.setText(f"...and {len(self.frames_with_class) - max_thumbnails} more")
            msg.setFlags(Qt.NoItemFlags) # not selectable

    def on_thumbnail_clicked(self, item):
        frame_idx = item.data(Qt.UserRole)
        if frame_idx is not None:
            # Navigate main window
            if hasattr(self.main_window, "frame_slider"):
                self.main_window.frame_slider.setValue(frame_idx)
                # optionally, you could keep dialog open or close it
                # self.accept() 
