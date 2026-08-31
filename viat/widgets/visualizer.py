import os
import json
import glob
import cv2
import numpy as np
from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
    QPushButton, QFileDialog, QComboBox, QListWidget, QMessageBox
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QImage, QPixmap, QPainter, QColor, QPen

def compute_iou(box1, box2):
    # COCO format: [x, y, width, height]
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    
    ix1 = max(x1, x2)
    iy1 = max(y1, y2)
    ix2 = min(x1+w1, x2+w2)
    iy2 = min(y1+h1, y2+h2)
    
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    union = w1*h1 + w2*h2 - inter
    return inter / union if union > 0 else 0

class EvaluatorVisualizer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("VIAT - Advanced Evaluation Visualizer")
        self.resize(1200, 800)
        
        self.gt_data = {}
        self.dt_data = {}
        self.categories = {}
        
        self.current_video_name = None
        self.current_frame_id = None
        self.frames_list = []
        
        self.base_video_path = ""
        
        self.init_ui()
        
    def init_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)
        
        # Left Panel (Controls)
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_panel.setFixedWidth(300)
        
        btn_load_json = QPushButton("Load Evaluation JSONs Folder")
        btn_load_json.clicked.connect(self.load_jsons)
        left_layout.addWidget(btn_load_json)
        
        btn_base_path = QPushButton("Set Video/Images Base Path")
        btn_base_path.clicked.connect(self.set_base_path)
        left_layout.addWidget(btn_base_path)
        
        self.combo_video = QComboBox()
        self.combo_video.currentIndexChanged.connect(self.on_video_changed)
        left_layout.addWidget(QLabel("Select Video:"))
        left_layout.addWidget(self.combo_video)
        
        # Filters
        self.combo_class = QComboBox()
        self.combo_class.currentIndexChanged.connect(self.update_view)
        left_layout.addWidget(QLabel("Filter by Class:"))
        left_layout.addWidget(self.combo_class)
        
        self.combo_type = QComboBox()
        self.combo_type.addItems(["All", "TP (True Positive)", "FP (False Positive)", "FN (False Negative)"])
        self.combo_type.currentIndexChanged.connect(self.update_view)
        left_layout.addWidget(QLabel("Filter by Type:"))
        left_layout.addWidget(self.combo_type)
        
        # Next / Prev buttons for quick jumping
        jump_layout = QHBoxLayout()
        btn_prev_filter = QPushButton("<< Prev Match")
        btn_next_filter = QPushButton("Next Match >>")
        btn_prev_filter.clicked.connect(self.prev_match)
        btn_next_filter.clicked.connect(self.next_match)
        jump_layout.addWidget(btn_prev_filter)
        jump_layout.addWidget(btn_next_filter)
        left_layout.addLayout(jump_layout)
        
        self.list_frames = QListWidget()
        self.list_frames.currentRowChanged.connect(self.on_frame_selected)
        left_layout.addWidget(QLabel("Frames:"))
        left_layout.addWidget(self.list_frames)
        
        # Right Panel (Image)
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        
        self.lbl_image = QLabel("Load JSONs to begin")
        self.lbl_image.setAlignment(Qt.AlignCenter)
        self.lbl_image.setStyleSheet("background-color: black;")
        right_layout.addWidget(self.lbl_image)
        
        main_layout.addWidget(left_panel)
        main_layout.addWidget(right_panel)
        
    def set_base_path(self):
        path = QFileDialog.getExistingDirectory(self, "Select Base Directory containing Videos/Images")
        if path:
            self.base_video_path = path
            self.update_view()
            
    def load_jsons(self):
        folder = QFileDialog.getExistingDirectory(self, "Select evaluation_result/json Folder")
        if not folder: return
        
        gt_dir = os.path.join(folder, "GT")
        dt_dir = os.path.join(folder, "DT")
        
        if not os.path.exists(gt_dir) or not os.path.exists(dt_dir):
            self.lbl_image.setText("Invalid folder. Must contain 'GT' and 'DT' subfolders.")
            return
            
        self.gt_data.clear()
        self.dt_data.clear()
        self.combo_video.clear()
        self.combo_class.clear()
        
        self.categories = {}
        
        for gt_file in glob.glob(os.path.join(gt_dir, "*.json")):
            name = os.path.basename(gt_file)
            if "all_video" in name: continue
            
            with open(gt_file, 'r') as f:
                self.gt_data[name] = json.load(f)
                
            dt_file = os.path.join(dt_dir, name)
            if os.path.exists(dt_file):
                with open(dt_file, 'r') as f:
                    self.dt_data[name] = json.load(f)
            
            for cat in self.gt_data[name].get('categories', []):
                self.categories[cat['id']] = cat['name']
                
        self.combo_class.addItem("All Classes", None)
        for cat_id, cat_name in self.categories.items():
            self.combo_class.addItem(cat_name, cat_id)
            
        self.combo_video.addItems(sorted(self.gt_data.keys()))
        
    def on_video_changed(self):
        self.current_video_name = self.combo_video.currentText()
        if not self.current_video_name: return
        
        gt = self.gt_data[self.current_video_name]
        self.frames_list = sorted(gt.get('images', []), key=lambda x: x['id'])
        
        self.list_frames.clear()
        for frame in self.frames_list:
            self.list_frames.addItem(frame['file_name'])
            
        if self.frames_list:
            self.list_frames.setCurrentRow(0)
            
    def on_frame_selected(self, idx):
        if idx < 0 or idx >= len(self.frames_list): return
        self.current_frame_id = self.frames_list[idx]['id']
        self.update_view()
        
    def get_frame_path(self, file_name):
        if os.path.exists(file_name): return file_name
        base = os.path.join(self.base_video_path, file_name)
        if os.path.exists(base): return base
        base_flat = os.path.join(self.base_video_path, os.path.basename(file_name))
        if os.path.exists(base_flat): return base_flat
        return None
        
    def get_matches(self, frame_id, class_filter=None):
        gt_all = self.gt_data[self.current_video_name].get('annotations', [])
        dt_all = self.dt_data.get(self.current_video_name, {}).get('annotations', [])
        
        gt_frame = [g for g in gt_all if g['image_id'] == frame_id]
        dt_frame = [d for d in dt_all if d['image_id'] == frame_id]
        
        if class_filter is not None:
            gt_frame = [g for g in gt_frame if g['category_id'] == class_filter]
            dt_frame = [d for d in dt_frame if d['category_id'] == class_filter]
            
        matched_gt = set()
        matched_dt = set()
        matches = []
        
        dt_frame = sorted(dt_frame, key=lambda x: x.get('score', 0), reverse=True)
        
        for d in dt_frame:
            best_iou = 0
            best_g = None
            for g in gt_frame:
                if g['id'] in matched_gt: continue
                iou = compute_iou(d['bbox'], g['bbox'])
                if iou > best_iou:
                    best_iou = iou
                    best_g = g
            if best_iou >= 0.5:
                matched_gt.add(best_g['id'])
                matched_dt.add(d['id'])
                matches.append((best_g, d))
                
        tp = [d for d in dt_frame if d['id'] in matched_dt]
        fp = [d for d in dt_frame if d['id'] not in matched_dt]
        fn = [g for g in gt_frame if g['id'] not in matched_gt]
        
        return tp, fp, fn
        
    def update_view(self):
        if not self.current_video_name or not self.current_frame_id: return
        
        frame_info = next((f for f in self.frames_list if f['id'] == self.current_frame_id), None)
        if not frame_info: return
        
        path = self.get_frame_path(frame_info['file_name'])
        
        pixmap = None
        if path:
            if path.lower().endswith(('.mp4', '.avi', '.mkv')):
                self.lbl_image.setText("Please set base path to extracted frame images.\nDirect video reading is not fully supported.")
                return
            else:
                img = cv2.imread(path)
                if img is not None:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    h, w, c = img.shape
                    qimg = QImage(img.data, w, h, w*c, QImage.Format_RGB888)
                    pixmap = QPixmap.fromImage(qimg)
        
        if pixmap is None:
            w, h = frame_info.get('width', 1920), frame_info.get('height', 1080)
            pixmap = QPixmap(w, h)
            pixmap.fill(Qt.darkGray)
            
        painter = QPainter(pixmap)
        
        class_filter = self.combo_class.currentData()
        type_filter = self.combo_type.currentText()
        
        tp, fp, fn = self.get_matches(self.current_frame_id, class_filter)
        
        def draw_box(box, color, label):
            x, y, w, h = box
            painter.setPen(QPen(color, 3))
            painter.drawRect(int(x), int(y), int(w), int(h))
            painter.setPen(Qt.white)
            painter.setBackground(color)
            painter.setBackgroundMode(Qt.OpaqueMode)
            painter.drawText(int(x), int(y)-5, label)

        if "All" in type_filter or "TP" in type_filter:
            for d in tp:
                cat_name = self.categories.get(d['category_id'], str(d['category_id']))
                draw_box(d['bbox'], Qt.green, f"TP: {cat_name} {d.get('score',1):.2f}")
                
        if "All" in type_filter or "FP" in type_filter:
            for d in fp:
                cat_name = self.categories.get(d['category_id'], str(d['category_id']))
                draw_box(d['bbox'], Qt.red, f"FP: {cat_name} {d.get('score',1):.2f}")
                
        if "All" in type_filter or "FN" in type_filter:
            for g in fn:
                cat_name = self.categories.get(g['category_id'], str(g['category_id']))
                draw_box(g['bbox'], Qt.blue, f"FN: {cat_name} (Missed)")
                
        painter.end()
        self.lbl_image.setPixmap(pixmap.scaled(self.lbl_image.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def prev_match(self):
        self.jump_match(-1)
        
    def next_match(self):
        self.jump_match(1)
        
    def jump_match(self, direction):
        if not self.frames_list: return
        class_filter = self.combo_class.currentData()
        type_filter = self.combo_type.currentText()
        
        idx = self.list_frames.currentRow()
        while True:
            idx += direction
            if idx < 0 or idx >= len(self.frames_list):
                QMessageBox.information(self, "End of Video", "No more matches found in this direction.")
                break
                
            frame_id = self.frames_list[idx]['id']
            tp, fp, fn = self.get_matches(frame_id, class_filter)
            
            found = False
            if "TP" in type_filter and tp: found = True
            elif "FP" in type_filter and fp: found = True
            elif "FN" in type_filter and fn: found = True
            elif "All" in type_filter and (tp or fp or fn): found = True
            
            if found:
                self.list_frames.setCurrentRow(idx)
                break

if __name__ == "__main__":
    import sys
    from PyQt5.QtWidgets import QApplication
    app = QApplication(sys.argv)
    window = EvaluatorVisualizer()
    window.show()
    sys.exit(app.exec_())
