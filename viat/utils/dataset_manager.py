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
    QListWidgetItem,
    QRadioButton,
)
from PyQt5.QtGui import QColor
from PyQt5.QtCore import Qt,QRect
import random
import xml.dom.minidom as minidom
import xml.etree.ElementTree as ET

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
    """Show dialog for importing a dataset with advanced options."""
    dialog = QDialog(parent)
    dialog.setWindowTitle("Import Dataset")
    dialog.setMinimumWidth(500)
    
    layout = QVBoxLayout(dialog)
    
    # Dataset type selection
    type_group = QGroupBox("Dataset Type")
    type_layout = QVBoxLayout(type_group)
    
    coco_radio = QRadioButton("COCO JSON")
    yolo_radio = QRadioButton("YOLO")
    pascal_radio = QRadioButton("Pascal VOC")
    
    coco_radio.setChecked(True)  # Default to COCO
    
    type_layout.addWidget(coco_radio)
    type_layout.addWidget(yolo_radio)
    type_layout.addWidget(pascal_radio)
    
    # Class mapping section
    class_group = QGroupBox("Class Mapping")
    class_layout = QVBoxLayout(class_group)
    
    class_list = QListWidget()
    class_list.setSelectionMode(QListWidget.ExtendedSelection)
    
    # Add class mapping controls
    class_map_layout = QHBoxLayout()
    class_map_label = QLabel("Map selected to:")
    class_map_input = QLineEdit()
    class_map_btn = QPushButton("Map")
    
    class_map_layout.addWidget(class_map_label)
    class_map_layout.addWidget(class_map_input)
    class_map_layout.addWidget(class_map_btn)
    
    # Class selection controls
    class_select_layout = QHBoxLayout()
    select_all_btn = QPushButton("Select All")
    deselect_all_btn = QPushButton("Deselect All")
    invert_selection_btn = QPushButton("Invert Selection")
    
    class_select_layout.addWidget(select_all_btn)
    class_select_layout.addWidget(deselect_all_btn)
    class_select_layout.addWidget(invert_selection_btn)
    
    class_layout.addWidget(class_list)
    class_layout.addLayout(class_map_layout)
    class_layout.addLayout(class_select_layout)
    
    # Options section
    options_group = QGroupBox("Import Options")
    options_layout = QVBoxLayout(options_group)
    
    skip_empty_check = QCheckBox("Skip images without annotations")
    skip_empty_check.setChecked(True)
    
    options_layout.addWidget(skip_empty_check)
    
    # Buttons
    buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
    buttons.accepted.connect(dialog.accept)
    buttons.rejected.connect(dialog.reject)
    
    # Add all sections to main layout
    layout.addWidget(type_group)
    layout.addWidget(class_group)
    layout.addWidget(options_group)
    layout.addWidget(buttons)
    
    # Connect signals
    def detect_and_load_classes():
        class_list.clear()
        
        # Detect dataset type and load classes
        if coco_radio.isChecked():
            # Find COCO JSON files
            json_files = [f for f in os.listdir(folder_path) 
                         if f.endswith('.json') and os.path.isfile(os.path.join(folder_path, f))]
            
            if json_files:
                # Try to load the first JSON file that looks like COCO
                for json_file in json_files:
                    try:
                        with open(os.path.join(folder_path, json_file), 'r') as f:
                            data = json.load(f)
                            if 'categories' in data:
                                # Found COCO file, load classes
                                for category in data['categories']:
                                    item = QListWidgetItem(category['name'])
                                    item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
                                    item.setCheckState(Qt.Checked)
                                    item.setData(Qt.UserRole, category['id'])
                                    class_list.addItem(item)
                                break
                    except:
                        continue
        
        elif yolo_radio.isChecked():
            # Look for classes.txt or obj.names
            class_files = ['classes.txt', 'obj.names']
            for class_file in class_files:
                class_path = os.path.join(folder_path, class_file)
                if os.path.exists(class_path):
                    try:
                        with open(class_path, 'r') as f:
                            for i, line in enumerate(f):
                                class_name = line.strip()
                                if class_name:
                                    item = QListWidgetItem(class_name)
                                    item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
                                    item.setCheckState(Qt.Checked)
                                    item.setData(Qt.UserRole, i)
                                    class_list.addItem(item)
                        break
                    except:
                        continue
        
        elif pascal_radio.isChecked():
            # Look for a sample XML file to extract classes
            xml_files = [f for f in os.listdir(folder_path) 
                        if f.endswith('.xml') and os.path.isfile(os.path.join(folder_path, f))]
            
            classes = set()
            for xml_file in xml_files[:10]:  # Check first 10 files
                try:
                    tree = ET.parse(os.path.join(folder_path, xml_file))
                    root = tree.getroot()
                    for obj in root.findall('.//object'):
                        class_name = obj.find('name').text
                        classes.add(class_name)
                except:
                    continue
            
            for i, class_name in enumerate(sorted(classes)):
                item = QListWidgetItem(class_name)
                item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
                item.setCheckState(Qt.Checked)
                item.setData(Qt.UserRole, i)
                class_list.addItem(item)
    
    # Connect radio buttons to class detection
    coco_radio.toggled.connect(detect_and_load_classes)
    yolo_radio.toggled.connect(detect_and_load_classes)
    pascal_radio.toggled.connect(detect_and_load_classes)
    
    # Connect class selection buttons
    select_all_btn.clicked.connect(lambda: [class_list.item(i).setCheckState(Qt.Checked) 
                                           for i in range(class_list.count())])
    deselect_all_btn.clicked.connect(lambda: [class_list.item(i).setCheckState(Qt.Unchecked) 
                                             for i in range(class_list.count())])
    invert_selection_btn.clicked.connect(lambda: [class_list.item(i).setCheckState(
                                                 Qt.Unchecked if class_list.item(i).checkState() == Qt.Checked 
                                                 else Qt.Checked) 
                                                 for i in range(class_list.count())])
    
    # Class mapping functionality
    def map_selected_classes():
        new_class = class_map_input.text().strip()
        if not new_class:
            return
            
        for item in class_list.selectedItems():
            item.setText(f"{item.text()} → {new_class}")
            item.setData(Qt.UserRole + 1, new_class)  # Store mapping
    
    class_map_btn.clicked.connect(map_selected_classes)
    
    # Initial class detection
    detect_and_load_classes()
    
    # Show dialog
    if dialog.exec_() == QDialog.Accepted:
        # Create configuration
        config = {
            "dataset_type": "coco" if coco_radio.isChecked() else 
                           "yolo" if yolo_radio.isChecked() else "pascal_voc",
            "folder_path": folder_path,
            "skip_empty": skip_empty_check.isChecked(),
            "classes": {}
        }
        
        # Get selected classes and mappings
        for i in range(class_list.count()):
            item = class_list.item(i)
            if item.checkState() == Qt.Checked:
                original_class = item.text().split(' → ')[0]
                mapped_class = item.data(Qt.UserRole + 1) or original_class
                class_id = item.data(Qt.UserRole)
                config["classes"][class_id] = {
                    "original": original_class,
                    "mapped": mapped_class,
                    "enabled": True
                }
            else:
                class_id = item.data(Qt.UserRole)
                original_class = item.text().split(' → ')[0]
                config["classes"][class_id] = {
                    "original": original_class,
                    "mapped": original_class,
                    "enabled": False
                }
        
        return config
    
    return None

def load_dataset(parent, config, frame_annotations, class_colors, BoundingBox):
    """Load a dataset based on configuration."""
    dataset_type = config.get("dataset_type", "coco")
    folder_path = config.get("folder_path", "")
    skip_empty = config.get("skip_empty", True)
    classes_config = config.get("classes", {})
    
    image_files = []
    
    # Create progress dialog
    progress = QDialog(parent)
    progress.setWindowTitle("Loading Dataset")
    progress.setFixedSize(400, 100)
    progress_layout = QVBoxLayout(progress)
    
    status_label = QLabel("Loading dataset...")
    progress_layout.addWidget(status_label)
    
    progress_bar = QProgressBar()
    progress_bar.setRange(0, 100)
    progress_layout.addWidget(progress_bar)
    
    # Non-blocking progress dialog
    progress.setModal(False)
    progress.show()
    QApplication.processEvents()
    
    try:
        if dataset_type == "coco":
            # Find COCO JSON file
            json_files = [f for f in os.listdir(folder_path) 
                         if f.endswith('.json') and os.path.isfile(os.path.join(folder_path, f))]
            
            if not json_files:
                progress.close()
                return [], "No COCO JSON file found in the selected folder."
            
            # Try to load the first JSON file that looks like COCO
            coco_file = None
            for json_file in json_files:
                try:
                    with open(os.path.join(folder_path, json_file), 'r') as f:
                        data = json.load(f)
                        if 'images' in data and 'annotations' in data and 'categories' in data:
                            coco_file = json_file
                            break
                except:
                    continue
            
            if not coco_file:
                progress.close()
                return [], "No valid COCO JSON file found in the selected folder."
            
            # Load COCO dataset
            status_label.setText(f"Loading COCO dataset from {coco_file}...")
            QApplication.processEvents()
            
            with open(os.path.join(folder_path, coco_file), 'r') as f:
                data = json.load(f)
            
            # Create mapping from category ID to class name
            category_mapping = {}
            for category in data['categories']:
                cat_id = category['id']
                if cat_id in classes_config and classes_config[cat_id]['enabled']:
                    # Use mapped class name if available
                    category_mapping[cat_id] = classes_config[cat_id]['mapped']
                    
                    # Ensure the class color exists
                    if category_mapping[cat_id] not in class_colors:
                        class_colors[category_mapping[cat_id]] = QColor(
                            random.randint(0, 255),
                            random.randint(0, 255),
                            random.randint(0, 255)
                        )
            
            # Create mapping from image ID to filename
            image_mapping = {}
            for image in data['images']:
                image_mapping[image['id']] = {
                    'file_name': image['file_name'],
                    'width': image.get('width', 0),
                    'height': image.get('height', 0)
                }
            
            # Group annotations by image ID
            annotations_by_image = {}
            for annotation in data['annotations']:
                image_id = annotation['image_id']
                category_id = annotation['category_id']
                
                # Skip if category is not enabled
                if category_id not in category_mapping:
                    continue
                    
                if image_id not in annotations_by_image:
                    annotations_by_image[image_id] = []
                
                # Convert COCO bbox [x,y,width,height] to QRect
                bbox = annotation['bbox']
                rect = QRect(int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]))
                
                # Get class name from mapping
                class_name = category_mapping[category_id]
                
                # Create BoundingBox object
                bbox_obj = BoundingBox(
                    rect=rect,
                    class_name=class_name,
                    attributes={"Size": -1, "Quality": -1},  # Default attributes
                    color=class_colors[class_name]
                )
                
                annotations_by_image[image_id].append(bbox_obj)
            
            # Find all image files
            all_images = []
            for image_id, image_info in image_mapping.items():
                file_name = image_info['file_name']
                file_path = os.path.join(folder_path, file_name)
                
                # Check if file exists
                if os.path.exists(file_path):
                    has_annotations = image_id in annotations_by_image
                    
                    # Skip images without annotations if requested
                    if skip_empty and not has_annotations:
                        continue
                    
                    all_images.append((file_path, image_id))
            
            # Sort images by filename
            all_images.sort(key=lambda x: x[0])
            
            # Update progress bar
            progress_bar.setRange(0, len(all_images))
            
            # Load images and annotations
            for i, (file_path, image_id) in enumerate(all_images):
                # Update progress
                progress_bar.setValue(i)
                status_label.setText(f"Loading image {i+1}/{len(all_images)}: {os.path.basename(file_path)}")
                                # Update progress
                progress_bar.setValue(i)
                status_label.setText(f"Loading image {i+1}/{len(all_images)}: {os.path.basename(file_path)}")
                QApplication.processEvents()
                
                # Add to image files list
                image_files.append(file_path)
                
                # Add annotations to frame_annotations dictionary
                if image_id in annotations_by_image:
                    frame_annotations[i] = annotations_by_image[image_id]
            
            # Close progress dialog
            progress.close()
            
            return image_files, f"Loaded {len(image_files)} images with {sum(len(anns) for anns in frame_annotations.values())} annotations from COCO dataset"
            
        elif dataset_type == "yolo":
            # Look for classes.txt or obj.names
            class_files = ['classes.txt', 'obj.names']
            class_file = None
            class_list = []
            
            for cf in class_files:
                class_path = os.path.join(folder_path, cf)
                if os.path.exists(class_path):
                    class_file = class_path
                    with open(class_path, 'r') as f:
                        class_list = [line.strip() for line in f if line.strip()]
                    break
            
            if not class_file or not class_list:
                progress.close()
                return [], "No YOLO class file found in the selected folder."
            
            # Create class mapping
            class_mapping = {}
            for i, class_name in enumerate(class_list):
                if i in classes_config and classes_config[i]['enabled']:
                    # Use mapped class name if available
                    class_mapping[i] = classes_config[i]['mapped']
                    
                    # Ensure the class color exists
                    if class_mapping[i] not in class_colors:
                        class_colors[class_mapping[i]] = QColor(
                            random.randint(0, 255),
                            random.randint(0, 255),
                            random.randint(0, 255)
                        )
            
            # Find all image files
            image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp']
            all_images = []
            
            for file in os.listdir(folder_path):
                if any(file.lower().endswith(ext) for ext in image_extensions):
                    image_path = os.path.join(folder_path, file)
                    txt_path = os.path.join(folder_path, os.path.splitext(file)[0] + '.txt')
                    
                    has_annotations = os.path.exists(txt_path)
                    
                    # Skip images without annotations if requested
                    if skip_empty and not has_annotations:
                        continue
                    
                    all_images.append((image_path, txt_path))
            
            # Sort images by filename
            all_images.sort(key=lambda x: x[0])
            
            # Update progress bar
            progress_bar.setRange(0, len(all_images))
            
            # Load images and annotations
            for i, (image_path, txt_path) in enumerate(all_images):
                # Update progress
                progress_bar.setValue(i)
                status_label.setText(f"Loading image {i+1}/{len(all_images)}: {os.path.basename(image_path)}")
                QApplication.processEvents()
                
                # Add to image files list
                image_files.append(image_path)
                
                # Load annotations if available
                if os.path.exists(txt_path):
                    # Get image dimensions
                    img = cv2.imread(image_path)
                    if img is None:
                        continue
                    
                    img_height, img_width = img.shape[:2]
                    
                    annotations = []
                    with open(txt_path, 'r') as f:
                        for line in f:
                            parts = line.strip().split()
                            if len(parts) >= 5:
                                try:
                                    class_id = int(parts[0])
                                    
                                    # Skip if class is not enabled
                                    if class_id not in class_mapping:
                                        continue
                                    
                                    # YOLO format: class_id x_center y_center width height (normalized)
                                    x_center = float(parts[1]) * img_width
                                    y_center = float(parts[2]) * img_height
                                    width = float(parts[3]) * img_width
                                    height = float(parts[4]) * img_height
                                    
                                    # Convert to top-left coordinates
                                    x = int(x_center - width / 2)
                                    y = int(y_center - height / 2)
                                    
                                    rect = QRect(x, y, int(width), int(height))
                                    
                                    # Get class name from mapping
                                    class_name = class_mapping[class_id]
                                    
                                    # Parse additional attributes if present
                                    attributes = {"Size": -1, "Quality": -1}
                                    if len(parts) > 5 and '#' in line:
                                        attr_part = line.split('#', 1)[1].strip()
                                        for attr in attr_part.split(','):
                                            if ':' in attr:
                                                attr_name, attr_value = attr.split(':', 1)
                                                try:
                                                    attributes[attr_name.strip()] = int(attr_value.strip())
                                                except ValueError:
                                                    attributes[attr_name.strip()] = attr_value.strip()
                                    
                                    # Create BoundingBox object
                                    bbox_obj = BoundingBox(
                                        rect=rect,
                                        class_name=class_name,
                                        attributes=attributes,
                                        color=class_colors[class_name]
                                    )
                                    
                                    annotations.append(bbox_obj)
                                except (ValueError, IndexError):
                                    continue
                    
                    if annotations:
                        frame_annotations[i] = annotations
            
            # Close progress dialog
            progress.close()
            
            return image_files, f"Loaded {len(image_files)} images with {sum(len(anns) for anns in frame_annotations.values())} annotations from YOLO dataset"
            
        elif dataset_type == "pascal_voc":
            # Find all XML files
            xml_files = [f for f in os.listdir(folder_path) 
                        if f.endswith('.xml') and os.path.isfile(os.path.join(folder_path, f))]
            
            if not xml_files:
                progress.close()
                return [], "No Pascal VOC XML files found in the selected folder."
            
            # Create class mapping from config
            class_mapping = {}
            for class_id, class_info in classes_config.items():
                if class_info['enabled']:
                    original_class = class_info['original']
                    mapped_class = class_info['mapped']
                    class_mapping[original_class] = mapped_class
                    
                    # Ensure the class color exists
                    if mapped_class not in class_colors:
                        class_colors[mapped_class] = QColor(
                            random.randint(0, 255),
                            random.randint(0, 255),
                            random.randint(0, 255)
                        )
            
            # Find all image files corresponding to XML files
            all_images = []
            
            for xml_file in xml_files:
                xml_path = os.path.join(folder_path, xml_file)
                
                try:
                    tree = ET.parse(xml_path)
                    root = tree.getroot()
                    
                    # Get filename from XML
                    filename_elem = root.find('filename')
                    if filename_elem is None:
                        continue
                    
                    image_filename = filename_elem.text
                    
                    # Check for different image extensions
                    image_path = None
                    for ext in ['.jpg', '.jpeg', '.png', '.bmp']:
                        potential_path = os.path.join(folder_path, os.path.splitext(image_filename)[0] + ext)
                        if os.path.exists(potential_path):
                            image_path = potential_path
                            break
                    
                    if image_path is None:
                        continue
                    
                    # Check if there are any enabled annotations
                    has_enabled_annotations = False
                    for obj in root.findall('.//object'):
                        class_elem = obj.find('name')
                        if class_elem is not None and class_elem.text in class_mapping:
                            has_enabled_annotations = True
                            break
                    
                    # Skip images without enabled annotations if requested
                    if skip_empty and not has_enabled_annotations:
                        continue
                    
                    all_images.append((image_path, xml_path))
                    
                except Exception as e:
                    print(f"Error parsing {xml_file}: {str(e)}")
                    continue
            
            # Sort images by filename
            all_images.sort(key=lambda x: x[0])
            
            # Update progress bar
            progress_bar.setRange(0, len(all_images))
            
            # Load images and annotations
            for i, (image_path, xml_path) in enumerate(all_images):
                # Update progress
                progress_bar.setValue(i)
                status_label.setText(f"Loading image {i+1}/{len(all_images)}: {os.path.basename(image_path)}")
                QApplication.processEvents()
                
                # Add to image files list
                image_files.append(image_path)
                
                # Parse XML for annotations
                try:
                    tree = ET.parse(xml_path)
                    root = tree.getroot()
                    
                    annotations = []
                    
                    for obj in root.findall('.//object'):
                        class_elem = obj.find('name')
                        if class_elem is None:
                            continue
                        
                        original_class = class_elem.text
                        
                        # Skip if class is not enabled
                        if original_class not in class_mapping:
                            continue
                        
                        # Get mapped class name
                        class_name = class_mapping[original_class]
                        
                        # Get bounding box coordinates
                        bbox = obj.find('bndbox')
                        if bbox is None:
                            continue
                        
                        xmin = int(float(bbox.find('xmin').text))
                        ymin = int(float(bbox.find('ymin').text))
                        xmax = int(float(bbox.find('xmax').text))
                        ymax = int(float(bbox.find('ymax').text))
                        
                        width = xmax - xmin
                        height = ymax - ymin
                        
                        rect = QRect(xmin, ymin, width, height)
                        
                        # Check for attributes in the XML
                        attributes = {"Size": -1, "Quality": -1}
                        for attr_elem in obj.findall('./attribute'):
                            attr_name = attr_elem.find('name')
                            attr_value = attr_elem.find('value')
                            
                            if attr_name is not None and attr_value is not None:
                                try:
                                    attributes[attr_name.text] = int(attr_value.text)
                                except ValueError:
                                    attributes[attr_name.text] = attr_value.text
                        
                        # Create BoundingBox object
                        bbox_obj = BoundingBox(
                            rect=rect,
                            class_name=class_name,
                            attributes=attributes,
                            color=class_colors[class_name]
                        )
                        
                        annotations.append(bbox_obj)
                    
                    if annotations:
                        frame_annotations[i] = annotations
                        
                except Exception as e:
                    print(f"Error loading annotations from {xml_path}: {str(e)}")
                    continue
            
            # Close progress dialog
            progress.close()
            
            return image_files, f"Loaded {len(image_files)} images with {sum(len(anns) for anns in frame_annotations.values())} annotations from Pascal VOC dataset"
    
    except Exception as e:
        # Close progress dialog
        progress.close()
        return [], f"Error loading dataset: {str(e)}"





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
