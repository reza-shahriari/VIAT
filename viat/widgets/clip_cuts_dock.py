import os
from PyQt5.QtWidgets import (QDockWidget, QWidget, QVBoxLayout, QHBoxLayout, 
                             QPushButton, QTableWidget, QTableWidgetItem, QHeaderView,
                             QMessageBox)
from PyQt5.QtCore import Qt

class ClipCutsDock(QDockWidget):
    def __init__(self, title, parent=None):
        super().__init__(title, parent)
        self.main_window = parent
        self.initUI()
        
    def initUI(self):
        self.setAllowedAreas(Qt.LeftDockWidgetArea | Qt.RightDockWidgetArea)
        
        main_widget = QWidget()
        layout = QVBoxLayout(main_widget)
        
        # Table to hold cuts
        self.table = QTableWidget(0, 3)
        self.table.setHorizontalHeaderLabels(['Cut Name', 'Start Frame', 'End Frame'])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table.setSelectionBehavior(QTableWidget.SelectRows)
        layout.addWidget(self.table)
        
        # Button controls
        btn_layout = QHBoxLayout()
        
        self.btn_add = QPushButton('Add Cut')
        self.btn_add.setToolTip('Add a new cut row')
        self.btn_add.clicked.connect(self.add_cut)
        btn_layout.addWidget(self.btn_add)
        
        self.btn_remove = QPushButton('Remove')
        self.btn_remove.setToolTip('Remove selected cut')
        self.btn_remove.clicked.connect(self.remove_cut)
        btn_layout.addWidget(self.btn_remove)
        
        self.btn_clear = QPushButton('Clear All')
        self.btn_clear.clicked.connect(self.clear_all)
        btn_layout.addWidget(self.btn_clear)
        
        layout.addLayout(btn_layout)
        
        # Helper controls for currently selected row
        helper_layout = QHBoxLayout()
        
        self.btn_set_start = QPushButton('Set Start to Current')
        self.btn_set_start.clicked.connect(self.set_start_frame)
        helper_layout.addWidget(self.btn_set_start)
        
        self.btn_set_end = QPushButton('Set End to Current')
        self.btn_set_end.clicked.connect(self.set_end_frame)
        helper_layout.addWidget(self.btn_set_end)
        
        layout.addLayout(helper_layout)
        
        # Export button
        export_layout = QVBoxLayout()
        self.btn_export = QPushButton('Export Cuts')
        self.btn_export.setStyleSheet("background-color: #2196F3; color: white; font-weight: bold; padding: 5px;")
        self.btn_export.clicked.connect(self.trigger_export)
        export_layout.addWidget(self.btn_export)
        
        layout.addLayout(export_layout)
        
        self.setWidget(main_widget)
        
    def add_cut(self):
        row = self.table.rowCount()
        self.table.insertRow(row)
        
        # Default Name
        name_item = QTableWidgetItem(f"cut{row+1}")
        self.table.setItem(row, 0, name_item)
        
        # Default Start (current frame if available)
        start_frame = 0
        if self.main_window and hasattr(self.main_window, 'current_frame'):
            start_frame = self.main_window.current_frame
        
        start_item = QTableWidgetItem(str(start_frame))
        self.table.setItem(row, 1, start_item)
        
        # Default End
        end_item = QTableWidgetItem(str(start_frame + 100))
        self.table.setItem(row, 2, end_item)
        
        self.table.selectRow(row)
        
    def remove_cut(self):
        current_row = self.table.currentRow()
        if current_row >= 0:
            self.table.removeRow(current_row)
            
    def clear_all(self):
        self.table.setRowCount(0)
        
    def set_start_frame(self):
        current_row = self.table.currentRow()
        if current_row >= 0 and self.main_window and hasattr(self.main_window, 'current_frame'):
            self.table.item(current_row, 1).setText(str(self.main_window.current_frame))
            
    def set_end_frame(self):
        current_row = self.table.currentRow()
        if current_row >= 0 and self.main_window and hasattr(self.main_window, 'current_frame'):
            self.table.item(current_row, 2).setText(str(self.main_window.current_frame))
            
    def get_cuts(self):
        \"\"\"Return a list of cuts in format: [{'name': 'cut1', 'start': 0, 'end': 100}, ...]\"\"\"
        cuts = []
        for row in range(self.table.rowCount()):
            name = self.table.item(row, 0).text()
            try:
                start = int(self.table.item(row, 1).text())
                end = int(self.table.item(row, 2).text())
                cuts.append({'name': name, 'start': start, 'end': end})
            except ValueError:
                if self.main_window:
                    QMessageBox.warning(self, "Invalid Input", f"Start and End frames for '{name}' must be integers.")
                continue
        return cuts

    def trigger_export(self):
        if self.main_window and hasattr(self.main_window, 'export_clip_cuts'):
            self.main_window.export_clip_cuts()
