from PyQt5.QtWidgets import QDialog, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QComboBox, QRadioButton, QPushButton, QButtonGroup, QMessageBox, QSpinBox, QWidget, QListWidget, QListWidgetItem, QCheckBox, QDoubleSpinBox, QTableWidget, QTableWidgetItem, QHeaderView, QAbstractItemView
from PyQt5.QtCore import Qt
import re


class AutoAnnotateDialog(QDialog):

    def __init__(self, current_frame=0, total_frames=1, parent=None):
        super().__init__(parent)
        self.current_frame = current_frame
        self.total_frames = total_frames
        self.setWindowTitle('Auto Annotate Dataset')
        self.setMinimumWidth(500)
        self.setup_ui()

    def setup_ui(self):
        layout = QVBoxLayout(self)
        classes_label_layout = QHBoxLayout()
        classes_label_layout.addWidget(QLabel('Classes Configuration:'))
        self.btn_add_class = QPushButton('Add New Class')
        self.btn_add_class.clicked.connect(lambda : self.add_new_class_row())
        classes_label_layout.addStretch()
        classes_label_layout.addWidget(self.btn_add_class)
        layout.addLayout(classes_label_layout)
        self.classes_table = QTableWidget()
        self.classes_table.setColumnCount(6)
        self.classes_table.setHorizontalHeaderLabels(['Class Name',
            'Action', 'Extract Target (Helper)', 'Ignore Target (Helper)',
            'Rename To (Opt)', 'Deduplicate Against'])
        self.classes_table.horizontalHeader().setSectionResizeMode(QHeaderView
            .ResizeToContents)
        self.classes_table.horizontalHeader().setStretchLastSection(True)
        self.classes_table.setSelectionMode(QAbstractItemView.NoSelection)
        self.classes_table.setMinimumHeight(200)
        layout.addWidget(self.classes_table)
        self.populate_classes_table()
        layout.addWidget(QLabel(
            'Zero-Shot Detection Models (Select one or more):'))
        self.det_model_list = QListWidget()
        self.det_model_list.setSelectionMode(QAbstractItemView.MultiSelection)
        det_models = ['Existing Annotations (Hand labeled)',
            'YOLO-World Small (yolov8s-world.pt)',
            'YOLO-World Medium (yolov8m-world.pt)',
            'YOLO-World Large (yolov8l-world.pt)',
            'YOLO-World v2 Small (yolov8s-worldv2.pt)',
            'YOLO-World v2 Medium (yolov8m-worldv2.pt)',
            'YOLO-World v2 Large (yolov8l-worldv2.pt)',
            'YOLO-World v2 XLarge (yolov8x-worldv2.pt)',
            'YOLOv11 World Small (yolo11s-world.pt)',
            'YOLOv11 World Medium (yolo11m-world.pt)',
            'YOLOv11 World Large (yolo11l-world.pt)',
            'YOLOv11 World XLarge (yolo11x-world.pt)',
            'YOLOE 11l (yoloe-11l-seg.pt)', 'YOLOE 26s (yoloe-26s-seg.pt)',
            'YOLOE 26x (yoloe-26x-seg.pt)',
            'Grounding DINO Tiny (IDEA-Research/grounding-dino-tiny)',
            'Grounding DINO Base (IDEA-Research/grounding-dino-base)',
            'Florence-2 Base (microsoft/Florence-2-base)',
            'Florence-2 Large (microsoft/Florence-2-large)',
            'LocateAnything-3B (nvidia/LocateAnything-3B)',
            'SAM3 Text Prompt (sam3.1_l.pt)']
        self.det_model_list.addItems(det_models)
        for i in range(self.det_model_list.count()):
            if self.det_model_list.item(i).text() == 'yolov8x-worldv2.pt':
                self.det_model_list.item(i).setSelected(True)
        self.det_model_list.setMaximumHeight(80)
        layout.addWidget(self.det_model_list)
        layout.addWidget(QLabel('Segmentation Refiner (Optional):'))
        self.seg_model_combo = QComboBox()
        self.seg_model_combo.addItem('None')
        self.seg_model_combo.addItems(['SAM2 Fast (sam2_s.pt)',
            'SAM2 Huge (sam2_l.pt)', 'SAM3 Fast (sam3_s.pt)',
            'SAM3 Huge (sam3_l.pt)'])
        layout.addWidget(self.seg_model_combo)
        if hasattr(self, 'parent_app') and getattr(self.parent_app,
            'sam_manager', None):
            if getattr(self.parent_app.sam_manager, 'current_model_type', None
                ):
                idx = self.seg_model_combo.findText(self.parent_app.
                    sam_manager.current_model_type)
                if idx >= 0:
                    self.seg_model_combo.setCurrentIndex(idx)
            else:
                self.seg_model_combo.setCurrentIndex(1)
        layout.addWidget(QLabel('Deduplication IoU Threshold:'))
        self.dedup_spin = QDoubleSpinBox()
        self.dedup_spin.setRange(0.0, 1.0)
        self.dedup_spin.setSingleStep(0.05)
        self.dedup_spin.setValue(0.7)
        layout.addWidget(self.dedup_spin)
        self.chk_save_seg = QCheckBox('Save Segmentation Masks (Polygons)')
        self.chk_save_seg.setChecked(True)
        layout.addWidget(self.chk_save_seg)
        scope_layout = QHBoxLayout()
        scope_layout.addWidget(QLabel('Frame Scope:'))
        self.rb_current = QRadioButton('Current Frame Only')
        self.rb_all = QRadioButton('All Frames')
        self.rb_all.setChecked(True)
        self.scope_group = QButtonGroup()
        self.scope_group.addButton(self.rb_current)
        self.scope_group.addButton(self.rb_all)
        scope_layout.addWidget(self.rb_current)
        scope_layout.addWidget(self.rb_all)
        layout.addLayout(scope_layout)
        conf_layout = QHBoxLayout()
        conf_layout.addWidget(QLabel('Confidence Threshold (0-100):'))
        self.threshold_spin = QSpinBox()
        self.threshold_spin.setRange(1, 100)
        self.threshold_spin.setValue(25)
        conf_layout.addWidget(self.threshold_spin)
        layout.addLayout(conf_layout)
        btn_layout = QHBoxLayout()
        self.btn_run = QPushButton('Run Auto Annotate')
        self.btn_run.clicked.connect(self.accept)
        self.btn_cancel = QPushButton('Cancel')
        self.btn_cancel.clicked.connect(self.reject)
        btn_layout.addWidget(self.btn_run)
        btn_layout.addWidget(self.btn_cancel)
        layout.addLayout(btn_layout)

    def on_det_model_changed(self, item):
        is_existing = False
        for i in range(self.det_model_list.count()):
            it = self.det_model_list.item(i)
            if it.checkState(
                ) == Qt.Checked and 'Existing Annotations' in it.text():
                is_existing = True
                break
        self.threshold_widget.setVisible(is_existing)
        if is_existing and self.seg_model_combo.currentIndex() == 0:
            self.seg_model_combo.setCurrentIndex(1)

    def get_config(self):
        classes_config = []
        for row in range(self.classes_table.rowCount()):
            class_name = self.classes_table.item(row, 0).text().strip()
            action_combo = self.classes_table.cellWidget(row, 1)
            extract_input = self.classes_table.cellWidget(row, 2)
            ignore_input = self.classes_table.cellWidget(row, 3)
            rename_input = self.classes_table.cellWidget(row, 4)
            dedup_input = self.classes_table.cellWidget(row, 5)
            if not class_name:
                continue
            action = action_combo.currentText()
            extract_prompt = extract_input.text().strip()
            ignore_prompt = ignore_input.text().strip()
            rename_to = rename_input.text().strip()
            dedup_against = dedup_input.text().strip()
            if action != 'Ignore':
                classes_config.append({'name': class_name, 'action': action,
                    'extract_prompt': extract_prompt, 'ignore_prompt':
                    ignore_prompt, 'rename_to': rename_to, 'dedup_against':
                    dedup_against})
        det_models = []
        for i in range(self.det_model_list.count()):
            item = self.det_model_list.item(i)
            if item.isSelected():
                text = item.text()
                if 'Existing Annotations' in text:
                    det_models.append('existing_annotations')
                else:
                    import re
                    m = re.search('\\((.*?)\\)', text)
                    if m:
                        det_models.append(m.group(1))
                    else:
                        det_models.append(text)
        seg_model = self.seg_model_combo.currentText()
        if 'None' in seg_model:
            seg_model = None
        else:
            import re
            m = re.search('\\((.*?)\\)', seg_model)
            if m:
                seg_model = m.group(1)
        start_frame = 0
        end_frame = self.total_frames - 1
        if self.rb_current.isChecked():
            start_frame = self.current_frame
            end_frame = start_frame
        return {'classes_config': classes_config, 'det_models': det_models,
            'seg_model': seg_model, 'save_segmentation': self.chk_save_seg.
            isChecked(), 'strategy': 'independent', 'start_frame':
            start_frame, 'end_frame': end_frame, 'threshold': self.
            threshold_spin.value(), 'dedup_iou': self.dedup_spin.value()}

    def populate_classes_table(self):
        self.classes_table.setRowCount(0)
        parent = self.parent()
        if parent:
            if hasattr(parent, 'canvas') and hasattr(parent.canvas,
                'class_colors'):
                existing_classes = list(parent.canvas.class_colors.keys())
                for cls_name in existing_classes:
                    self.add_new_class_row(cls_name)

    def add_new_class_row(self, class_name=''):
        row = self.classes_table.rowCount()
        self.classes_table.insertRow(row)
        name_item = QTableWidgetItem(class_name)
        if class_name:
            name_item.setFlags(name_item.flags() & ~Qt.ItemIsEditable)
        self.classes_table.setItem(row, 0, name_item)
        action_combo = QComboBox()
        action_combo.addItems(['Detect (Zero-Shot)',
            'Use as Helper (SAM3 Refine)', 'Remove Labels', 'Ignore'])
        if class_name:
            action_combo.setCurrentText('Ignore')
        else:
            action_combo.setCurrentText('Detect (Zero-Shot)')
        self.classes_table.setCellWidget(row, 1, action_combo)
        extract_input = QLineEdit()
        extract_input.setPlaceholderText('e.g. truck chassis')
        self.classes_table.setCellWidget(row, 2, extract_input)
        ignore_input = QLineEdit()
        ignore_input.setPlaceholderText('e.g. cargo')
        self.classes_table.setCellWidget(row, 3, ignore_input)
        rename_input = QLineEdit()
        rename_input.setPlaceholderText('e.g. Red Car')
        self.classes_table.setCellWidget(row, 4, rename_input)
        dedup_input = QLineEdit()
        dedup_input.setPlaceholderText('e.g. car, vehicle, *')
        self.classes_table.setCellWidget(row, 5, dedup_input)

        def on_action_changed(text, ext_in=extract_input, ign_in=
            ignore_input, ren_in=rename_input, dedup_in=dedup_input):
            if 'Helper' in text:
                ext_in.setEnabled(True)
                ign_in.setEnabled(True)
                ren_in.setEnabled(True)
                dedup_in.setEnabled(False)
                dedup_in.clear()
            elif 'Detect' in text:
                ext_in.setEnabled(True)
                ign_in.setEnabled(True)
                ren_in.setEnabled(False)
                ren_in.clear()
                dedup_in.setEnabled(True)
            else:
                ext_in.setEnabled(False)
                ign_in.setEnabled(False)
                ren_in.setEnabled(False)
                dedup_in.setEnabled(False)
                ext_in.clear()
                ign_in.clear()
                ren_in.clear()
                dedup_in.clear()
        action_combo.currentTextChanged.connect(on_action_changed)
        on_action_changed(action_combo.currentText())
