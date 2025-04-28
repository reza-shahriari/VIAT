"""
Dataset management utilities for handling various image dataset structures.

This module provides functions to import and export datasets with different
organizational structures including train/test/validation splits, parallel
image/label folders, and multiple annotation files.
"""

import os
import json
import shutil
import cv2
import numpy as np
from pathlib import Path
from PyQt5.QtWidgets import (
    QMessageBox,
    QHBoxLayout,
    QDialog,
    QVBoxLayout,
    QLabel,
    QListWidget,
    QDialogButtonBox,
    QCheckBox,
    QGroupBox,
    QFormLayout,
    QLineEdit,
    QComboBox,
    QPushButton,
    QFileDialog,
    QProgressBar,
    QApplication,
)
from PyQt5.QtCore import Qt

def detect_folder_type(folder_path):
    """
    Detect if a folder is a simple image folder or a structured dataset.
    
    Args:
        folder_path (str): Path to the folder
        
    Returns:
        str: "simple_folder" or "dataset"
    """
    # Check for common dataset indicators
    dataset_indicators = [
        os.path.isdir(os.path.join(folder_path, "train")),
        os.path.isdir(os.path.join(folder_path, "test")),
        os.path.isdir(os.path.join(folder_path, "val")),
        os.path.isdir(os.path.join(folder_path, "validation")),
        os.path.isdir(os.path.join(folder_path, "images")) and os.path.isdir(os.path.join(folder_path, "labels")),
        os.path.isdir(os.path.join(folder_path, "images")) and os.path.isdir(os.path.join(folder_path, "annotations")),
        os.path.exists(os.path.join(folder_path, "classes.txt")),
    ]
    
    # Check for annotation files
    annotation_files = []
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.endswith(('.json', '.xml')):
                if "coco" in file.lower() or "annotations" in file.lower() or "instances" in file.lower():
                    annotation_files.append(os.path.join(root, file))
    
    # If any dataset indicators are true or we found annotation files, it's likely a dataset
    if any(dataset_indicators) or annotation_files:
        return "dataset"
    else:
        return "simple_folder"
    
def detect_dataset_structure(folder_path):
    """
    Analyze a folder to detect common dataset structures.

    Args:
        folder_path (str): Path to the dataset folder

    Returns:
        dict: Information about the detected dataset structure
    """
    structure_info = {
        "type": "unknown",
        "has_splits": False,
        "splits": [],
        "annotation_files": [],
        "image_folders": [],
        "label_folders": [],
        "total_images": 0,
    }

    # Check for common split folders
    common_splits = ["train", "test", "val", "validation"]
    detected_splits = []

    for split in common_splits:
        split_path = os.path.join(folder_path, split)
        if os.path.isdir(split_path):
            detected_splits.append(split)

    # Check for parallel image/label structure
    images_folder = os.path.join(folder_path, "images")
    labels_folder = os.path.join(folder_path, "labels")
    annotations_folder = os.path.join(folder_path, "annotations")

    has_parallel_structure = os.path.isdir(images_folder) and (
        os.path.isdir(labels_folder) or os.path.isdir(annotations_folder)
    )

    # Check for annotation files
    annotation_files = []
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.endswith((".json", ".xml")):
                if "coco" in file.lower() or "annotations" in file.lower():
                    annotation_files.append(os.path.join(root, file))

    # Determine structure type
    if detected_splits:
        structure_info["type"] = "split_folders"
        structure_info["has_splits"] = True
        structure_info["splits"] = detected_splits

        # Check for images and labels within each split
        for split in detected_splits:
            split_path = os.path.join(folder_path, split)
            split_images = os.path.join(split_path, "images")
            split_labels = os.path.join(split_path, "labels")

            if os.path.isdir(split_images):
                structure_info["image_folders"].append(split_images)
            else:
                # The split folder itself might contain images
                structure_info["image_folders"].append(split_path)

            if os.path.isdir(split_labels):
                structure_info["label_folders"].append(split_labels)

    elif has_parallel_structure:
        structure_info["type"] = "parallel_folders"
        structure_info["image_folders"].append(images_folder)

        if os.path.isdir(labels_folder):
            structure_info["label_folders"].append(labels_folder)
        if os.path.isdir(annotations_folder):
            structure_info["label_folders"].append(annotations_folder)

    else:
        structure_info["type"] = "flat_folder"
        structure_info["image_folders"].append(folder_path)

    # Add annotation files
    structure_info["annotation_files"] = annotation_files

    # Count total images
    image_extensions = [".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"]
    total_images = 0

    for img_folder in structure_info["image_folders"]:
        for root, _, files in os.walk(img_folder):
            for file in files:
                if any(file.lower().endswith(ext) for ext in image_extensions):
                    total_images += 1

    structure_info["total_images"] = total_images

    return structure_info

def import_dataset_dialog(parent, folder_path):
    """
    Show a dialog to import a dataset with various options.
    
    Args:
        parent: Parent widget
        folder_path (str): Path to the dataset folder
        
    Returns:
        dict: Import configuration or None if cancelled
    """
    dialog = QDialog(parent)
    dialog.setWindowTitle("Import Image Dataset")
    dialog.setMinimumWidth(500)
    
    layout = QVBoxLayout(dialog)
    
    # Dataset structure detection
    structure_group = QGroupBox("Dataset Structure")
    structure_layout = QFormLayout(structure_group)
    
    # Detect available structures
    has_train_val = os.path.isdir(os.path.join(folder_path, "train")) and (
        os.path.isdir(os.path.join(folder_path, "val")) or 
        os.path.isdir(os.path.join(folder_path, "validation"))
    )
    
    has_images_labels = os.path.isdir(os.path.join(folder_path, "images")) and (
        os.path.isdir(os.path.join(folder_path, "labels")) or
        os.path.isdir(os.path.join(folder_path, "annotations"))
    )
    
    has_classes_txt = os.path.exists(os.path.join(folder_path, "classes.txt"))
    
    # Find annotation files
    annotation_files = []
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.endswith(('.json', '.xml')):
                if "coco" in file.lower() or "annotations" in file.lower() or "instances" in file.lower():
                    annotation_files.append(os.path.join(root, file))
    
    # Structure selection
    structure_combo = QComboBox()
    available_structures = []
    
    if has_train_val:
        available_structures.append("Split folders (train/val/test)")
    
    if has_images_labels:
        available_structures.append("Parallel folders (images/labels)")
    
    if has_classes_txt:
        available_structures.append("YOLO format")
    
    if annotation_files:
        available_structures.append("Annotation file")
    
    # Always add simple folder option
    available_structures.append("Simple folder (all images)")
    
    structure_combo.addItems(available_structures)
    structure_layout.addRow("Structure:", structure_combo)
    
    # Split folder options
    split_options = QGroupBox("Split Options")
    split_options.setVisible(False)
    split_layout = QFormLayout(split_options)
    
    split_checkboxes = {}
    if has_train_val:
        splits = []
        if os.path.isdir(os.path.join(folder_path, "train")):
            splits.append("train")
        if os.path.isdir(os.path.join(folder_path, "val")):
            splits.append("val")
        if os.path.isdir(os.path.join(folder_path, "validation")):
            splits.append("validation")
        if os.path.isdir(os.path.join(folder_path, "test")):
            splits.append("test")
        
        for split in splits:
            checkbox = QCheckBox(split)
            checkbox.setChecked(True)
            split_layout.addRow("", checkbox)
            split_checkboxes[split] = checkbox
    
    # Annotation file options
    annotation_options = QGroupBox("Annotation File")
    annotation_options.setVisible(False)
    annotation_layout = QFormLayout(annotation_options)
    
    annotation_combo = QComboBox()
    if annotation_files:
        for file_path in annotation_files:
            annotation_combo.addItem(os.path.basename(file_path), file_path)
    annotation_layout.addRow("File:", annotation_combo)
    
    # Show/hide options based on structure selection
    def update_options_visibility():
        selected = structure_combo.currentText()
        split_options.setVisible(selected.startswith("Split folders"))
        annotation_options.setVisible(selected.startswith("Annotation file"))
    
    structure_combo.currentTextChanged.connect(update_options_visibility)
    update_options_visibility()
    
    # Add options to layout
    structure_layout.addWidget(split_options)
    structure_layout.addWidget(annotation_options)
    layout.addWidget(structure_group)
    
    # Recursive option
    recursive_check = QCheckBox("Include images from subfolders")
    recursive_check.setChecked(True)
    layout.addWidget(recursive_check)
    
    # Buttons
    buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
    buttons.accepted.connect(dialog.accept)
    buttons.rejected.connect(dialog.reject)
    layout.addWidget(buttons)
    
    # Show dialog
    if dialog.exec_() == QDialog.Accepted:
        config = {
            "folder_path": folder_path,
            "structure": structure_combo.currentText(),
            "recursive": recursive_check.isChecked(),
        }
        
        # Add structure-specific options
        if config["structure"].startswith("Split folders"):
            config["selected_splits"] = [
                split for split, checkbox in split_checkboxes.items() if checkbox.isChecked()
            ]
        
        if config["structure"].startswith("Annotation file"):
            config["annotation_file"] = annotation_combo.currentData()
        
        return config
    
    return None

def load_dataset(parent, config, frame_annotations, class_colors, BoundingBox):
    """
    Load an image dataset based on the provided configuration.
    
    Args:
        parent: Parent widget for progress dialog
        config (dict): Dataset import configuration
        frame_annotations (dict): Dictionary to store annotations by frame
        class_colors (dict): Dictionary of class colors
        BoundingBox: BoundingBox class for creating annotations
        
    Returns:
        tuple: (image_files, success_message)
    """
    folder_path = config["folder_path"]
    structure = config["structure"]
    recursive = config["recursive"]
    
    # Create progress dialog
    progress = QDialog(parent)
    progress.setWindowTitle("Loading Dataset")
    progress.setFixedSize(400, 100)
    progress_layout = QVBoxLayout(progress)
    
    status_label = QLabel("Scanning for images...")
    progress_layout.addWidget(status_label)
    
    progress_bar = QProgressBar()
    progress_bar.setRange(0, 100)
    progress_layout.addWidget(progress_bar)
    
    # Non-blocking progress dialog
    progress.setModal(False)
    progress.show()
    QApplication.processEvents()
    
    # Find image files based on structure
    image_files = []
    image_extensions = [".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"]
    
    try:
        if structure.startswith("Split folders"):
            # Get selected splits
            selected_splits = config.get("selected_splits", ["train", "val", "test"])
            
            # Find images in each split
            for split in selected_splits:
                split_folder = os.path.join(folder_path, split)
                if not os.path.isdir(split_folder):
                    continue
                
                # Check for images folder within split
                images_folder = os.path.join(split_folder, "images")
                if os.path.isdir(images_folder):
                    search_folder = images_folder
                else:
                    search_folder = split_folder
                
                # Find images
                if recursive:
                    for root, _, files in os.walk(search_folder):
                        for file in files:
                            if any(file.lower().endswith(ext) for ext in image_extensions):
                                image_files.append(os.path.join(root, file))
                else:
                    for file in os.listdir(search_folder):
                        if any(file.lower().endswith(ext) for ext in image_extensions):
                            image_files.append(os.path.join(search_folder, file))
            
            # Sort image files
            image_files = sorted(image_files)
            
            # Update progress
            progress_bar.setValue(50)
            status_label.setText(f"Found {len(image_files)} images. Loading annotations...")
            QApplication.processEvents()
            
            # Load annotations from each split
            for split in selected_splits:
                split_folder = os.path.join(folder_path, split)
                if not os.path.isdir(split_folder):
                    continue
                
                # Check for annotations folder
                annotations_folder = os.path.join(split_folder, "annotations")
                if os.path.isdir(annotations_folder):
                    # Look for XML or JSON files
                    for file in os.listdir(annotations_folder):
                        if file.endswith(('.xml', '.json')):
                            annotation_file = os.path.join(annotations_folder, file)
                            try:
                                from utils import import_annotations
                                _, _, imported_frame_annotations = import_annotations(
                                    annotation_file, 
                                    BoundingBox, 
                                    640, 480, 
                                    class_colors
                                )
                                
                                # Update frame annotations
                                for frame_num, anns in imported_frame_annotations.items():
                                    if frame_num not in frame_annotations:
                                        frame_annotations[frame_num] = []
                                    frame_annotations[frame_num].extend(anns)
                            except Exception as e:
                                print(f"Error importing annotations from {annotation_file}: {str(e)}")
                
                # Check for labels folder (YOLO format)
                labels_folder = os.path.join(split_folder, "labels")
                if os.path.isdir(labels_folder):
                    # Look for classes.txt
                    classes_file = os.path.join(folder_path, "classes.txt")
                    if not os.path.exists(classes_file):
                        classes_file = os.path.join(split_folder, "classes.txt")
                    
                    if os.path.exists(classes_file):
                        # Read class names
                        with open(classes_file, 'r') as f:
                            class_names = [line.strip() for line in f.readlines()]
                        
                        # Add class colors if not exist
                        for class_name in class_names:
                            if class_name not in class_colors:
                                import random
                                from PyQt5.QtGui import QColor
                                class_colors[class_name] = QColor(
                                    random.randint(0, 255),
                                    random.randint(0, 255),
                                    random.randint(0, 255)
                                )
                        
                        # Process label files
                        for image_path in image_files:
                            # Get base name for matching with label
                            base_name = os.path.splitext(os.path.basename(image_path))[0]
                            label_file = os.path.join(labels_folder, f"{base_name}.txt")
                            
                            if os.path.exists(label_file):
                                try:
                                    # Get image dimensions
                                    img = cv2.imread(image_path)
                                    if img is not None:
                                        img_height, img_width = img.shape[:2]
                                    else:
                                        img_width, img_height = 640, 480
                                    
                                    # Get frame number
                                    frame_num = image_files.index(image_path)
                                    
                                    # Parse YOLO format
                                    with open(label_file, 'r') as f:
                                        lines = f.readlines()
                                    
                                    # Create annotations
                                    annotations = []
                                    for line in lines:
                                        parts = line.strip().split()
                                        if len(parts) >= 5:
                                            class_idx = int(parts[0])
                                            x_center = float(parts[1])
                                            y_center = float(parts[2])
                                            width = float(parts[3])
                                            height = float(parts[4])
                                            
                                            # Convert to pixel coordinates
                                            x = int((x_center - width/2) * img_width)
                                            y = int((y_center - height/2) * img_height)
                                            w = int(width * img_width)
                                            h = int(height * img_height)
                                            
                                            # Get class name
                                            if class_idx < len(class_names):
                                                class_name = class_names[class_idx]
                                            else:
                                                class_name = f"class_{class_idx}"
                                            
                                            # Create bounding box
                                            from PyQt5.QtCore import QRect
                                            rect = QRect(x, y, w, h)
                                                                                        # Create bounding box
                                            from PyQt5.QtCore import QRect
                                            rect = QRect(x, y, w, h)
                                            bbox = BoundingBox(
                                                rect, 
                                                class_name, 
                                                {"Size": -1, "Quality": -1}, 
                                                class_colors[class_name]
                                            )
                                            
                                            annotations.append(bbox)
                                    
                                    # Add to frame annotations
                                    if annotations:
                                        if frame_num not in frame_annotations:
                                            frame_annotations[frame_num] = []
                                        frame_annotations[frame_num].extend(annotations)
                                except Exception as e:
                                    print(f"Error importing YOLO annotation from {label_file}: {str(e)}")
            
            success_message = f"Loaded {len(image_files)} images from {len(selected_splits)} splits"
        
        elif structure.startswith("Parallel folders"):
            # Find images folder
            images_folder = os.path.join(folder_path, "images")
            if not os.path.isdir(images_folder):
                raise ValueError("Images folder not found")
            
            # Find images
            if recursive:
                for root, _, files in os.walk(images_folder):
                    for file in files:
                        if any(file.lower().endswith(ext) for ext in image_extensions):
                            image_files.append(os.path.join(root, file))
            else:
                for file in os.listdir(images_folder):
                    if any(file.lower().endswith(ext) for ext in image_extensions):
                        image_files.append(os.path.join(images_folder, file))
            
            # Sort image files
            image_files = sorted(image_files)
            
            # Update progress
            progress_bar.setValue(50)
            status_label.setText(f"Found {len(image_files)} images. Loading annotations...")
            QApplication.processEvents()
            
            # Check for annotations folder
            annotations_folder = os.path.join(folder_path, "annotations")
            if os.path.isdir(annotations_folder):
                # Look for COCO JSON file
                coco_files = [f for f in os.listdir(annotations_folder) if f.endswith('.json')]
                if coco_files:
                    coco_file = os.path.join(annotations_folder, coco_files[0])
                    try:
                        from utils import import_annotations
                        _, _, imported_frame_annotations = import_annotations(
                            coco_file, 
                            BoundingBox, 
                            640, 480, 
                            class_colors
                        )
                        
                        # Update frame annotations
                        for frame_num, anns in imported_frame_annotations.items():
                            if frame_num not in frame_annotations:
                                frame_annotations[frame_num] = []
                            frame_annotations[frame_num].extend(anns)
                    except Exception as e:
                        print(f"Error importing annotations from {coco_file}: {str(e)}")
                
                # Look for XML files
                for i, image_path in enumerate(image_files):
                    base_name = os.path.splitext(os.path.basename(image_path))[0]
                    xml_file = os.path.join(annotations_folder, f"{base_name}.xml")
                    
                    if os.path.exists(xml_file):
                        try:
                            from utils import import_annotations
                            _, _, imported_frame_annotations = import_annotations(
                                xml_file, 
                                BoundingBox, 
                                640, 480, 
                                class_colors
                            )
                            
                            # Map to correct frame number
                            if 0 in imported_frame_annotations:
                                if i not in frame_annotations:
                                    frame_annotations[i] = []
                                frame_annotations[i].extend(imported_frame_annotations[0])
                        except Exception as e:
                            print(f"Error importing annotations from {xml_file}: {str(e)}")
            
            # Check for labels folder (YOLO format)
            labels_folder = os.path.join(folder_path, "labels")
            if os.path.isdir(labels_folder):
                # Look for classes.txt
                classes_file = os.path.join(folder_path, "classes.txt")
                
                if os.path.exists(classes_file):
                    # Read class names
                    with open(classes_file, 'r') as f:
                        class_names = [line.strip() for line in f.readlines()]
                    
                    # Add class colors if not exist
                    for class_name in class_names:
                        if class_name not in class_colors:
                            import random
                            from PyQt5.QtGui import QColor
                            class_colors[class_name] = QColor(
                                random.randint(0, 255),
                                random.randint(0, 255),
                                random.randint(0, 255)
                            )
                    
                    # Process label files
                    for i, image_path in enumerate(image_files):
                        # Get base name for matching with label
                        base_name = os.path.splitext(os.path.basename(image_path))[0]
                        label_file = os.path.join(labels_folder, f"{base_name}.txt")
                        
                        if os.path.exists(label_file):
                            try:
                                # Get image dimensions
                                img = cv2.imread(image_path)
                                if img is not None:
                                    img_height, img_width = img.shape[:2]
                                else:
                                    img_width, img_height = 640, 480
                                
                                # Parse YOLO format
                                with open(label_file, 'r') as f:
                                    lines = f.readlines()
                                
                                # Create annotations
                                annotations = []
                                for line in lines:
                                    parts = line.strip().split()
                                    if len(parts) >= 5:
                                        class_idx = int(parts[0])
                                        x_center = float(parts[1])
                                        y_center = float(parts[2])
                                        width = float(parts[3])
                                        height = float(parts[4])
                                        
                                        # Convert to pixel coordinates
                                        x = int((x_center - width/2) * img_width)
                                        y = int((y_center - height/2) * img_height)
                                        w = int(width * img_width)
                                        h = int(height * img_height)
                                        
                                        # Get class name
                                        if class_idx < len(class_names):
                                            class_name = class_names[class_idx]
                                        else:
                                            class_name = f"class_{class_idx}"
                                        
                                        # Create bounding box
                                        from PyQt5.QtCore import QRect
                                        rect = QRect(x, y, w, h)
                                        bbox = BoundingBox(
                                            rect, 
                                            class_name, 
                                            {"Size": -1, "Quality": -1}, 
                                            class_colors[class_name]
                                        )
                                        
                                        annotations.append(bbox)
                                
                                # Add to frame annotations
                                if annotations:
                                    if i not in frame_annotations:
                                        frame_annotations[i] = []
                                    frame_annotations[i].extend(annotations)
                            except Exception as e:
                                print(f"Error importing YOLO annotation from {label_file}: {str(e)}")
            
            success_message = f"Loaded {len(image_files)} images from parallel folders structure"
        
        elif structure.startswith("YOLO format"):
            # Find images folder
            images_folder = os.path.join(folder_path, "images")
            if not os.path.isdir(images_folder):
                # Try to find images in the main folder
                images_folder = folder_path
            
            # Find images
            if recursive:
                for root, _, files in os.walk(images_folder):
                    for file in files:
                        if any(file.lower().endswith(ext) for ext in image_extensions):
                            image_files.append(os.path.join(root, file))
            else:
                for file in os.listdir(images_folder):
                    if any(file.lower().endswith(ext) for ext in image_extensions):
                        image_files.append(os.path.join(images_folder, file))
            
            # Sort image files
            image_files = sorted(image_files)
            
            # Update progress
            progress_bar.setValue(50)
            status_label.setText(f"Found {len(image_files)} images. Loading annotations...")
            QApplication.processEvents()
            
            # Look for classes.txt
            classes_file = os.path.join(folder_path, "classes.txt")
            
            if os.path.exists(classes_file):
                # Read class names
                with open(classes_file, 'r') as f:
                    class_names = [line.strip() for line in f.readlines()]
                
                # Add class colors if not exist
                for class_name in class_names:
                    if class_name not in class_colors:
                        import random
                        from PyQt5.QtGui import QColor
                        class_colors[class_name] = QColor(
                            random.randint(0, 255),
                            random.randint(0, 255),
                            random.randint(0, 255)
                        )
                
                # Find labels folder
                labels_folder = os.path.join(folder_path, "labels")
                if not os.path.isdir(labels_folder):
                    # Try to find labels in the main folder
                    labels_folder = folder_path
                
                # Process label files
                for i, image_path in enumerate(image_files):
                    # Get base name for matching with label
                    base_name = os.path.splitext(os.path.basename(image_path))[0]
                    label_file = os.path.join(labels_folder, f"{base_name}.txt")
                    
                    if os.path.exists(label_file):
                        try:
                            # Get image dimensions
                            img = cv2.imread(image_path)
                            if img is not None:
                                img_height, img_width = img.shape[:2]
                            else:
                                img_width, img_height = 640, 480
                            
                            # Parse YOLO format
                            with open(label_file, 'r') as f:
                                lines = f.readlines()
                            
                            # Create annotations
                            annotations = []
                            for line in lines:
                                parts = line.strip().split()
                                if len(parts) >= 5:
                                    class_idx = int(parts[0])
                                    x_center = float(parts[1])
                                    y_center = float(parts[2])
                                    width = float(parts[3])
                                    height = float(parts[4])
                                    
                                    # Convert to pixel coordinates
                                    x = int((x_center - width/2) * img_width)
                                    y = int((y_center - height/2) * img_height)
                                    w = int(width * img_width)
                                    h = int(height * img_height)
                                    
                                    # Get class name
                                    if class_idx < len(class_names):
                                        class_name = class_names[class_idx]
                                    else:
                                        class_name = f"class_{class_idx}"
                                    
                                    # Create bounding box
                                    from PyQt5.QtCore import QRect
                                    rect = QRect(x, y, w, h)
                                    bbox = BoundingBox(
                                        rect, 
                                        class_name, 
                                        {"Size": -1, "Quality": -1}, 
                                        class_colors[class_name]
                                    )
                                    
                                    annotations.append(bbox)
                            
                            # Add to frame annotations
                            if annotations:
                                if i not in frame_annotations:
                                    frame_annotations[i] = []
                                frame_annotations[i].extend(annotations)
                        except Exception as e:
                            print(f"Error importing YOLO annotation from {label_file}: {str(e)}")
            
            success_message = f"Loaded {len(image_files)} images from YOLO format"
        
        elif structure.startswith("Annotation file"):
            # Find images in the folder
            if recursive:
                for root, _, files in os.walk(folder_path):
                    for file in files:
                        if any(file.lower().endswith(ext) for ext in image_extensions):
                            image_files.append(os.path.join(root, file))
            else:
                for file in os.listdir(folder_path):
                    if any(file.lower().endswith(ext) for ext in image_extensions):
                        image_files.append(os.path.join(folder_path, file))
            
            # Sort image files
            image_files = sorted(image_files)
            
            # Update progress
            progress_bar.setValue(50)
            status_label.setText(f"Found {len(image_files)} images. Loading annotations...")
            QApplication.processEvents()
            
            # Import annotations from the specified file
            annotation_file = config.get("annotation_file")
            if annotation_file and os.path.exists(annotation_file):
                try:
                    from utils import import_annotations
                    _, _, imported_frame_annotations = import_annotations(
                        annotation_file, 
                        BoundingBox, 
                        640, 480, 
                        class_colors
                    )
                    
                    # Update frame annotations
                    for frame_num, anns in imported_frame_annotations.items():
                        if frame_num not in frame_annotations:
                            frame_annotations[frame_num] = []
                        frame_annotations[frame_num].extend(anns)
                except Exception as e:
                    print(f"Error importing annotations from {annotation_file}: {str(e)}")
            
            success_message = f"Loaded {len(image_files)} images with annotations from {os.path.basename(annotation_file)}"
        
        else:  # Simple folder
            # Find all images in the folder
            if recursive:
                for root, _, files in os.walk(folder_path):
                    for file in files:
                        if any(file.lower().endswith(ext) for ext in image_extensions):
                            image_files.append(os.path.join(root, file))
            else:
                for file in os.listdir(folder_path):
                    if any(file.lower().endswith(ext) for ext in image_extensions):
                        image_files.append(os.path.join(folder_path, file))
            
            # Sort image files
            image_files = sorted(image_files)
            success_message = f"Loaded {len(image_files)} images from folder"
        
        # Update progress
        progress_bar.setValue(100)
        status_label.setText("Dataset loaded successfully!")
        QApplication.processEvents()
        
        # Close progress dialog after a short delay
        import time
        time.sleep(0.5)
        progress.close()
        
        return image_files, success_message
    
    except Exception as e:
        progress.close()
        import traceback
        traceback.print_exc()
        QMessageBox.critical(
            parent,
            "Error Loading Dataset",
            f"Failed to load dataset: {str(e)}"
        )
        return [], "Error loading dataset"




def export_dataset_dialog(parent, image_files, frame_annotations):
    """
    Show a dialog to export a dataset with various options.

    Args:
        parent: Parent widget
        image_files (list): List of image file paths
        frame_annotations (dict): Dictionary of annotations by frame

    Returns:
        dict: Export configuration or None if cancelled
    """
    dialog = QDialog(parent)
    dialog.setWindowTitle("Export Dataset")
    dialog.setMinimumWidth(500)

    layout = QVBoxLayout(dialog)

    # Format selection
    format_group = QGroupBox("Export Format")
    format_layout = QFormLayout(format_group)

    format_combo = QComboBox()
    format_combo.addItems(["COCO JSON", "YOLO TXT", "Pascal VOC XML"])
    format_layout.addRow("Format:", format_combo)

    layout.addWidget(format_group)

    # Structure options
    structure_group = QGroupBox("Dataset Structure")
    structure_layout = QFormLayout(structure_group)

    structure_combo = QComboBox()
    structure_combo.addItems(
        [
            "Flat (all images in one folder)",
            "Split (train/val/test folders)",
            "Parallel (separate images and labels folders)",
        ]
    )
    structure_layout.addRow("Structure:", structure_combo)

    # Split options (initially hidden)
    split_options = QGroupBox("Split Options")
    split_options.setVisible(False)
    split_layout = QFormLayout(split_options)

    train_spin = QLineEdit("80")
    val_spin = QLineEdit("10")
    test_spin = QLineEdit("10")

    split_layout.addRow("Train %:", train_spin)
    split_layout.addRow("Validation %:", val_spin)
    split_layout.addRow("Test %:", test_spin)

    # Show/hide split options based on structure selection
    def update_split_visibility():
        split_options.setVisible(structure_combo.currentText().startswith("Split"))

    structure_combo.currentTextChanged.connect(update_split_visibility)

    structure_layout.addWidget(split_options)
    layout.addWidget(structure_group)

    # Output options
    output_group = QGroupBox("Output Options")
    output_layout = QFormLayout(output_group)

    output_path = QLineEdit()
    output_path.setReadOnly(True)

    browse_button = QPushButton("Browse...")

    def browse_output():
        folder = QFileDialog.getExistingDirectory(dialog, "Select Output Folder")
        if folder:
            output_path.setText(folder)

    browse_button.clicked.connect(browse_output)

    path_layout = QHBoxLayout()
    path_layout.addWidget(output_path)
    path_layout.addWidget(browse_button)

    output_layout.addRow("Output folder:", path_layout)

    copy_images = QCheckBox("Copy images to output folder")
    copy_images.setChecked(True)
    output_layout.addRow("", copy_images)

    layout.addWidget(output_group)

    # Buttons
    buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
    buttons.accepted.connect(dialog.accept)
    buttons.rejected.connect(dialog.reject)
    layout.addWidget(buttons)

    # Validate before accepting
    def validate_and_accept():
        if not output_path.text():
            QMessageBox.warning(dialog, "Error", "Please select an output folder.")
            return

        if structure_combo.currentText().startswith("Split"):
            try:
                train_pct = float(train_spin.text())
                val_pct = float(val_spin.text())
                test_pct = float(test_spin.text())

                if abs(train_pct + val_pct + test_pct - 100) > 0.01:
                    QMessageBox.warning(
                        dialog, "Error", "Split percentages must sum to 100%."
                    )
                    return
            except ValueError:
                QMessageBox.warning(
                    dialog, "Error", "Split percentages must be valid numbers."
                )
                return

        dialog.accept()

    buttons.accepted.disconnect()
    buttons.accepted.connect(validate_and_accept)

    # Show dialog
    if dialog.exec_() == QDialog.Accepted:
        config = {
            "format": format_combo.currentText(),
            "structure": structure_combo.currentText(),
            "output_path": output_path.text(),
            "copy_images": copy_images.isChecked(),
        }

        if structure_combo.currentText().startswith("Split"):
            config["train_pct"] = float(train_spin.text())
            config["val_pct"] = float(val_spin.text())
            config["test_pct"] = float(test_spin.text())

        return config

    return None


def export_dataset(parent, config, image_files, frame_annotations, class_colors):
    """
    Export a dataset based on the provided configuration.

    Args:
        parent: Parent widget for progress dialog
        config (dict): Dataset export configuration
        image_files (list): List of image file paths
        frame_annotations (dict): Dictionary of annotations by frame
        class_colors (dict): Dictionary of class colors

    Returns:
        str: Success message or None if failed
    """
    from utils import (
        export_image_dataset_pascal_voc,
        export_image_dataset_yolo,
        export_image_dataset_coco,
    )

    output_path = config["output_path"]
    export_format = config["format"]
    structure = config["structure"]
    copy_images = config["copy_images"]

    # Create progress dialog
    progress = QDialog(parent)
    progress.setWindowTitle("Exporting Dataset")
    progress.setFixedSize(400, 100)
    progress_layout = QVBoxLayout(progress)

    status_label = QLabel("Preparing export...")
    progress_layout.addWidget(status_label)

    progress_bar = QProgressBar()
    progress_bar.setRange(0, 100)
    progress_layout.addWidget(progress_bar)

    # Non-blocking progress dialog
    progress.setModal(False)
    progress.show()
    QApplication.processEvents()

    try:
        # Create necessary directories based on structure
        if structure.startswith("Flat"):
            # Flat structure - all in one folder
            os.makedirs(output_path, exist_ok=True)

            if export_format == "YOLO TXT":
                labels_dir = os.path.join(output_path, "labels")
                os.makedirs(labels_dir, exist_ok=True)
            elif export_format == "Pascal VOC XML":
                annotations_dir = os.path.join(output_path, "annotations")
                os.makedirs(annotations_dir, exist_ok=True)

            if copy_images:
                images_dir = os.path.join(output_path, "images")
                os.makedirs(images_dir, exist_ok=True)

            # Export annotations
            if export_format == "COCO JSON":
                status_label.setText("Exporting COCO annotations...")
                QApplication.processEvents()

                coco_file = os.path.join(output_path, "annotations.json")
                export_image_dataset_coco(
                    coco_file,
                    image_files,
                    frame_annotations,
                    class_colors,
                    640,
                    480,  # Default dimensions, will be overridden by actual image sizes
                )

            elif export_format == "YOLO TXT":
                status_label.setText("Exporting YOLO annotations...")
                QApplication.processEvents()

                export_image_dataset_yolo(
                    output_path, image_files, frame_annotations, class_colors
                )

            elif export_format == "Pascal VOC XML":
                status_label.setText("Exporting Pascal VOC annotations...")
                QApplication.processEvents()

                export_image_dataset_pascal_voc(
                    output_path,
                    image_files,
                    frame_annotations,
                    None,  # No pixmap needed, dimensions will be read from images
                )

            # Copy images if requested
            if copy_images:
                status_label.setText("Copying images...")
                progress_bar.setValue(50)
                QApplication.processEvents()

                for i, img_path in enumerate(image_files):
                    if i % 10 == 0:  # Update progress every 10 images
                        progress_value = 50 + (i * 50 // len(image_files))
                        progress_bar.setValue(progress_value)
                        QApplication.processEvents()

                    dest_path = os.path.join(images_dir, os.path.basename(img_path))
                    shutil.copy2(img_path, dest_path)

        elif structure.startswith("Split"):
            # Split structure - train/val/test folders
            train_pct = config.get("train_pct", 80)
            val_pct = config.get("val_pct", 10)
            test_pct = config.get("test_pct", 10)

            # Create split directories
            train_dir = os.path.join(output_path, "train")
            val_dir = os.path.join(output_path, "val")
            test_dir = os.path.join(output_path, "test")

            os.makedirs(train_dir, exist_ok=True)
            os.makedirs(val_dir, exist_ok=True)
            os.makedirs(test_dir, exist_ok=True)

            if copy_images:
                os.makedirs(os.path.join(train_dir, "images"), exist_ok=True)
                os.makedirs(os.path.join(train_dir, "images"), exist_ok=True)
                os.makedirs(os.path.join(val_dir, "images"), exist_ok=True)
                os.makedirs(os.path.join(test_dir, "images"), exist_ok=True)

            if export_format == "YOLO TXT":
                os.makedirs(os.path.join(train_dir, "labels"), exist_ok=True)
                os.makedirs(os.path.join(val_dir, "labels"), exist_ok=True)
                os.makedirs(os.path.join(test_dir, "labels"), exist_ok=True)
            elif export_format == "Pascal VOC XML":
                os.makedirs(os.path.join(train_dir, "annotations"), exist_ok=True)
                os.makedirs(os.path.join(val_dir, "annotations"), exist_ok=True)
                os.makedirs(os.path.join(test_dir, "annotations"), exist_ok=True)

            # Split the dataset
            import random

            random.seed(42)  # For reproducibility

            # Shuffle indices
            indices = list(range(len(image_files)))
            random.shuffle(indices)

            # Calculate split sizes
            train_size = int(len(indices) * train_pct / 100)
            val_size = int(len(indices) * val_pct / 100)

            # Split indices
            train_indices = indices[:train_size]
            val_indices = indices[train_size : train_size + val_size]
            test_indices = indices[train_size + val_size :]

            # Create split datasets
            splits = {
                "train": {"dir": train_dir, "indices": train_indices},
                "val": {"dir": val_dir, "indices": val_indices},
                "test": {"dir": test_dir, "indices": test_indices},
            }

            # Export each split
            for split_name, split_info in splits.items():
                split_dir = split_info["dir"]
                split_indices = split_info["indices"]

                status_label.setText(f"Processing {split_name} split...")
                QApplication.processEvents()

                # Get images and annotations for this split
                split_images = [image_files[i] for i in split_indices]
                split_annotations = {}

                for i, img_idx in enumerate(split_indices):
                    if img_idx in frame_annotations:
                        split_annotations[i] = frame_annotations[img_idx]

                # Export annotations for this split
                if export_format == "COCO JSON":
                    coco_file = os.path.join(
                        split_dir, f"{split_name}_annotations.json"
                    )
                    export_image_dataset_coco(
                        coco_file,
                        split_images,
                        split_annotations,
                        class_colors,
                        640,
                        480,
                    )
                elif export_format == "YOLO TXT":
                    export_image_dataset_yolo(
                        split_dir, split_images, split_annotations, class_colors
                    )
                elif export_format == "Pascal VOC XML":
                    export_image_dataset_pascal_voc(
                        split_dir, split_images, split_annotations, None
                    )

                # Copy images if requested
                if copy_images:
                    images_dir = os.path.join(split_dir, "images")

                    for img_path in split_images:
                        dest_path = os.path.join(images_dir, os.path.basename(img_path))
                        shutil.copy2(img_path, dest_path)

        elif structure.startswith("Parallel"):
            # Parallel structure - separate images and labels folders
            images_dir = os.path.join(output_path, "images")
            os.makedirs(images_dir, exist_ok=True)

            if export_format == "YOLO TXT":
                labels_dir = os.path.join(output_path, "labels")
                os.makedirs(labels_dir, exist_ok=True)
            elif export_format == "Pascal VOC XML":
                annotations_dir = os.path.join(output_path, "annotations")
                os.makedirs(annotations_dir, exist_ok=True)

            # Export annotations
            if export_format == "COCO JSON":
                status_label.setText("Exporting COCO annotations...")
                QApplication.processEvents()

                coco_file = os.path.join(output_path, "annotations.json")
                export_image_dataset_coco(
                    coco_file, image_files, frame_annotations, class_colors, 640, 480
                )
            elif export_format == "YOLO TXT":
                status_label.setText("Exporting YOLO annotations...")
                QApplication.processEvents()

                # Export class names
                classes = list(class_colors.keys())
                with open(os.path.join(output_path, "classes.txt"), "w") as f:
                    for cls in classes:
                        f.write(f"{cls}\n")

                # Export each annotation to a separate file
                for i, img_path in enumerate(image_files):
                    if i % 10 == 0:
                        progress_value = 30 + (i * 40 // len(image_files))
                        progress_bar.setValue(progress_value)
                        QApplication.processEvents()

                    if i not in frame_annotations:
                        continue

                    base_name = os.path.splitext(os.path.basename(img_path))[0]
                    label_file = os.path.join(labels_dir, f"{base_name}.txt")

                    # Get image dimensions
                    img = cv2.imread(img_path)
                    if img is None:
                        continue

                    img_height, img_width = img.shape[:2]

                    # Write YOLO format annotations
                    with open(label_file, "w") as f:
                        for ann in frame_annotations[i]:
                            # Get class index
                            class_idx = classes.index(ann.class_name)

                            # Convert to YOLO format (normalized)
                            x = ann.rect.x()
                            y = ann.rect.y()
                            w = ann.rect.width()
                            h = ann.rect.height()

                            # Calculate center point and normalized dimensions
                            x_center = (x + w / 2) / img_width
                            y_center = (y + h / 2) / img_height
                            width = w / img_width
                            height = h / img_height

                            f.write(
                                f"{class_idx} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n"
                            )

            elif export_format == "Pascal VOC XML":
                status_label.setText("Exporting Pascal VOC annotations...")
                QApplication.processEvents()

                export_image_dataset_pascal_voc(
                    output_path, image_files, frame_annotations, None
                )

            # Copy images if requested
            if copy_images:
                status_label.setText("Copying images...")
                progress_bar.setValue(70)
                QApplication.processEvents()

                for i, img_path in enumerate(image_files):
                    if i % 10 == 0:
                        progress_value = 70 + (i * 30 // len(image_files))
                        progress_bar.setValue(progress_value)
                        QApplication.processEvents()

                    dest_path = os.path.join(images_dir, os.path.basename(img_path))
                    shutil.copy2(img_path, dest_path)

        progress_bar.setValue(100)
        status_label.setText("Export completed successfully!")
        QApplication.processEvents()

        # Close progress dialog after a short delay
        import time

        time.sleep(0.5)
        progress.close()

        # Return success message
        return f"Dataset exported successfully to {output_path}"

    except Exception as e:
        progress.close()
        QMessageBox.critical(
            parent, "Export Error", f"An error occurred during export: {str(e)}"
        )
        return None

def create_dataset_dialog(parent, image_files, frame_annotations, class_colors):
    """
    Show a dialog to create a new dataset from the current annotations.
    
    Args:
        parent: Parent widget
        image_files (list): List of image file paths
        frame_annotations (dict): Dictionary of annotations by frame
        class_colors (dict): Dictionary of class colors
        
    Returns:
        dict: Export configuration or None if cancelled
    """
    dialog = QDialog(parent)
    dialog.setWindowTitle("Create Dataset")
    dialog.setMinimumWidth(500)
    
    layout = QVBoxLayout(dialog)
    
    # Dataset format selection
    format_group = QGroupBox("Dataset Format")
    format_layout = QFormLayout(format_group)
    
    format_combo = QComboBox()
    format_combo.addItems([
        "COCO JSON",
        "YOLO",
        "Pascal VOC",
        "Custom"
    ])
    format_layout.addRow("Format:", format_combo)
    
    # Output directory
    output_group = QGroupBox("Output Directory")
    output_layout = QFormLayout(output_group)
    
    output_path = QLineEdit()
    output_path.setReadOnly(True)
    
    browse_button = QPushButton("Browse...")
    browse_button.clicked.connect(lambda: select_output_directory(output_path))
    
    output_row = QHBoxLayout()
    output_row.addWidget(output_path)
    output_row.addWidget(browse_button)
    
    output_layout.addRow("Directory:", output_row)
    
    # Dataset structure options
    structure_group = QGroupBox("Dataset Structure")
    structure_layout = QFormLayout(structure_group)
    
    structure_combo = QComboBox()
    structure_combo.addItems([
        "Simple (all images in one folder)",
        "Split (train/val/test folders)",
        "Parallel (images and labels folders)"
    ])
    structure_layout.addRow("Structure:", structure_combo)
    
    # Split options
    split_options = QGroupBox("Train/Val/Test Split")
    split_options.setVisible(False)
    split_layout = QFormLayout(split_options)
    
    train_split = QLineEdit("70")
    val_split = QLineEdit("20")
    test_split = QLineEdit("10")
    
    split_layout.addRow("Train %:", train_split)
    split_layout.addRow("Val %:", val_split)
    split_layout.addRow("Test %:", test_split)
    
    # Show/hide split options based on structure selection
    def update_structure_visibility():
        split_options.setVisible(structure_combo.currentText().startswith("Split"))
    
    structure_combo.currentTextChanged.connect(update_structure_visibility)
    
    # Add options to layout
    layout.addWidget(format_group)
    layout.addWidget(output_group)
    layout.addWidget(structure_group)
    layout.addWidget(split_options)
    
    # Buttons
    buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
    buttons.accepted.connect(dialog.accept)
    buttons.rejected.connect(dialog.reject)
    layout.addWidget(buttons)
    
    # Helper function to select output directory
    def select_output_directory(line_edit):
        directory = QFileDialog.getExistingDirectory(
            dialog, "Select Output Directory", "", QFileDialog.ShowDirsOnly
        )
        if directory:
            line_edit.setText(directory)
    
    # Show dialog
    if dialog.exec_() == QDialog.Accepted:
        # Validate inputs
        if not output_path.text():
            QMessageBox.warning(
                parent, "Missing Output Directory", "Please select an output directory."
            )
            return None
        
        # Get split percentages if needed
        splits = {}
        if structure_combo.currentText().startswith("Split"):
            try:
                train_pct = int(train_split.text())
                val_pct = int(val_split.text())
                test_pct = int(test_split.text())
                
                if train_pct + val_pct + test_pct != 100:
                    QMessageBox.warning(
                        parent, "Invalid Split", "Split percentages must sum to 100%."
                    )
                    return None
                
                splits = {
                    "train": train_pct / 100,
                    "val": val_pct / 100,
                    "test": test_pct / 100
                }
            except ValueError:
                QMessageBox.warning(
                    parent, "Invalid Split", "Split percentages must be integers."
                )
                return None
        
        return {
            "format": format_combo.currentText(),
            "output_dir": output_path.text(),
            "structure": structure_combo.currentText(),
            "splits": splits
        }
    
    return None

def create_dataset(parent, config, image_files, frame_annotations, class_colors):
    """
    Create a new dataset from the current annotations.
    
    Args:
        parent: Parent widget for progress dialog
        config (dict): Export configuration
        image_files (list): List of image file paths
        frame_annotations (dict): Dictionary of annotations by frame
        class_colors (dict): Dictionary of class colors
        
    Returns:
        bool: True if successful, False otherwise
    """
    from PyQt5.QtWidgets import QApplication
    import os
    import shutil
    import random
    
    # Create progress dialog
    progress = QDialog(parent)
    progress.setWindowTitle("Creating Dataset")
    progress.setFixedSize(400, 100)
    progress_layout = QVBoxLayout(progress)
    
    status_label = QLabel("Preparing dataset...")
    progress_layout.addWidget(status_label)
    
    progress_bar = QProgressBar()
    progress_bar.setRange(0, len(image_files))
    progress_layout.addWidget(progress_bar)
    
    # Non-blocking progress dialog
    progress.setModal(False)
    progress.show()
    QApplication.processEvents()
    
    try:
        output_dir = config["output_dir"]
        format_type = config["format"]
        structure = config["structure"]
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Process based on structure
        if structure.startswith("Simple"):
            # Simple structure - all images in one folder
            images_dir = os.path.join(output_dir, "images")
            os.makedirs(images_dir, exist_ok=True)
            
            # Copy images
            for i, image_path in enumerate(image_files):
                progress_bar.setValue(i)
                status_label.setText(f"Copying image {i+1}/{len(image_files)}...")
                QApplication.processEvents()
                
                # Copy image to output directory
                dest_path = os.path.join(images_dir, os.path.basename(image_path))
                shutil.copy2(image_path, dest_path)
            
            # Export annotations based on format
            progress_bar.setValue(len(image_files))
            status_label.setText("Exporting annotations...")
            QApplication.processEvents()
            
            if format_type == "COCO JSON":
                from utils.file_operations import export_image_dataset_coco
                annotations_dir = os.path.join(output_dir, "annotations")
                os.makedirs(annotations_dir, exist_ok=True)
                
                export_image_dataset_coco(
                    os.path.join(annotations_dir, "instances_default.json"),
                    image_files,
                    frame_annotations,
                    class_colors,
                    640, 480  # Default dimensions
                )
            
            elif format_type == "YOLO":
                from utils.file_operations import export_image_dataset_yolo
                export_image_dataset_yolo(
                    output_dir,
                    image_files,
                    frame_annotations,
                    class_colors
                )
            
            elif format_type == "Pascal VOC":
                from utils.file_operations import export_image_dataset_pascal_voc
                export_image_dataset_pascal_voc(
                    output_dir,
                    image_files,
                    frame_annotations,
                    None  # No pixmap available
                )
        
        elif structure.startswith("Split"):
            # Split structure - train/val/test folders
            splits = config.get("splits", {"train": 0.7, "val": 0.2, "test": 0.1})
            
            # Create split directories
            for split in splits:
                os.makedirs(os.path.join(output_dir, split, "images"), exist_ok=True)
                if format_type == "Pascal VOC":
                    os.makedirs(os.path.join(output_dir, split, "annotations"), exist_ok=True)
                elif format_type == "YOLO":
                    os.makedirs(os.path.join(output_dir, split, "labels"), exist_ok=True)
            
            # Shuffle and split images
            image_indices = list(range(len(image_files)))
            random.shuffle(image_indices)
            
            train_end = int(len(image_indices) * splits.get("train", 0.7))
            val_end = train_end + int(len(image_indices) * splits.get("val", 0.2))

            split_indices = {
                "train": image_indices[:train_end],
                "val": image_indices[train_end:val_end],
                "test": image_indices[val_end:]
            }
            
            # Process each split
            for split, indices in split_indices.items():
                split_images = []
                split_frame_annotations = {}
                
                # Copy images and collect annotations
                for i, idx in enumerate(indices):
                    progress_bar.setValue(i)
                    status_label.setText(f"Processing {split} split: {i+1}/{len(indices)}...")
                    QApplication.processEvents()
                    
                    image_path = image_files[idx]
                    dest_path = os.path.join(output_dir, split, "images", os.path.basename(image_path))
                    
                    # Copy image
                    shutil.copy2(image_path, dest_path)
                    split_images.append(dest_path)
                    
                    # Copy annotations
                    if idx in frame_annotations:
                        split_frame_annotations[len(split_images) - 1] = frame_annotations[idx]
                
                # Export annotations based on format
                if format_type == "COCO JSON":
                    from utils.file_operations import export_image_dataset_coco
                    annotations_dir = os.path.join(output_dir, split, "annotations")
                    os.makedirs(annotations_dir, exist_ok=True)
                    
                    export_image_dataset_coco(
                        os.path.join(annotations_dir, f"instances_{split}.json"),
                        split_images,
                        split_frame_annotations,
                        class_colors,
                        640, 480  # Default dimensions
                    )
                
                elif format_type == "YOLO":
                    from utils.file_operations import export_image_dataset_yolo
                    export_image_dataset_yolo(
                        os.path.join(output_dir, split),
                        split_images,
                        split_frame_annotations,
                        class_colors
                    )
                
                elif format_type == "Pascal VOC":
                    from utils.file_operations import export_image_dataset_pascal_voc
                    export_image_dataset_pascal_voc(
                        os.path.join(output_dir, split),
                        split_images,
                        split_frame_annotations,
                        None  # No pixmap available
                    )
        
        elif structure.startswith("Parallel"):
            # Parallel structure - images and labels folders
            images_dir = os.path.join(output_dir, "images")
            os.makedirs(images_dir, exist_ok=True)
            
            if format_type == "YOLO":
                labels_dir = os.path.join(output_dir, "labels")
                os.makedirs(labels_dir, exist_ok=True)
            elif format_type == "Pascal VOC":
                annotations_dir = os.path.join(output_dir, "annotations")
                os.makedirs(annotations_dir, exist_ok=True)
            elif format_type == "COCO JSON":
                annotations_dir = os.path.join(output_dir, "annotations")
                os.makedirs(annotations_dir, exist_ok=True)
            
            # Copy images
            for i, image_path in enumerate(image_files):
                progress_bar.setValue(i)
                status_label.setText(f"Copying image {i+1}/{len(image_files)}...")
                QApplication.processEvents()
                
                # Copy image to output directory
                dest_path = os.path.join(images_dir, os.path.basename(image_path))
                shutil.copy2(image_path, dest_path)
            
            # Export annotations based on format
            progress_bar.setValue(len(image_files))
            status_label.setText("Exporting annotations...")
            QApplication.processEvents()
            
            if format_type == "COCO JSON":
                from utils.file_operations import export_image_dataset_coco
                export_image_dataset_coco(
                    os.path.join(annotations_dir, "instances_default.json"),
                    image_files,
                    frame_annotations,
                    class_colors,
                    640, 480  # Default dimensions
                )
            
            elif format_type == "YOLO":
                from utils.file_operations import export_image_dataset_yolo
                export_image_dataset_yolo(
                    output_dir,
                    image_files,
                    frame_annotations,
                    class_colors
                )
            
            elif format_type == "Pascal VOC":
                from utils.file_operations import export_image_dataset_pascal_voc
                export_image_dataset_pascal_voc(
                    output_dir,
                    image_files,
                    frame_annotations,
                    None  # No pixmap available
                )
        
        # Update progress
        progress_bar.setValue(len(image_files))
        status_label.setText("Dataset created successfully!")
        QApplication.processEvents()
        
        # Close progress dialog after a short delay
        import time
        time.sleep(0.5)
        progress.close()
        
        return True
    
    except Exception as e:
        progress.close()
        import traceback
        traceback.print_exc()
        QMessageBox.critical(
            parent,
            "Error Creating Dataset",
            f"Failed to create dataset: {str(e)}"
        )
        return False
