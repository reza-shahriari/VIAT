import os
import json
import cv2
import numpy as np
from PyQt5.QtCore import QRect, QSaveFile, QIODevice
from PyQt5.QtGui import QColor
import shutil
import datetime
import random
import xml.dom.minidom as minidom
import xml.etree.ElementTree as ET
import glob

def load_project_with_backup(filename):
    """
    Loads a JSON project file, falling back to the most recent backup if needed.

    Args:
        filename (str): Path to the main project file.

    Returns:
        dict or None: The loaded project data, or None if all attempts fail.
    """
    try:
        with open(filename, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"[Warning] Failed to load main file: {e}")

        # Try to load most recent backup
        base_dir = os.path.dirname(filename)
        name, ext = os.path.splitext(os.path.basename(filename))
        backup_pattern = os.path.join(base_dir, f"{name}_backup_*{ext}")
        backups = sorted(glob.glob(backup_pattern), reverse=True)

        for backup_file in backups:
            try:
                with open(backup_file, 'r') as f:
                    print(f"[Info] Loaded backup file: {backup_file}")
                    return json.load(f)
            except Exception as e:
                print(f"[Warning] Failed to load backup {backup_file}: {e}")

        print("[Error] No valid project or backup files could be loaded.")
        return None

def backup_before_save(filename, use_timestamp=True, backup_limit=5):
    """
    Create a backup of the given file before overwriting it.

    Args:
        filename (str): Path of the file to back up
        use_timestamp (bool): Whether to append a timestamp to the backup file
        backup_limit (int): Max number of backup files to keep
    """
    if not os.path.exists(filename):
        return  # Nothing to back up

    base_dir = os.path.dirname(filename)
    base_name = os.path.basename(filename)
    name, ext = os.path.splitext(base_name)

    if use_timestamp:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_name = f"{name}_backup_{timestamp}{ext}"
    else:
        backup_name = f"{name}.bak{ext}"

    backup_path = os.path.join(base_dir, backup_name)

    try:
        shutil.copy2(filename, backup_path)  # includes metadata
    except Exception as e:
        print(f"Warning: Failed to create backup: {e}")

    # Cleanup old backups (optional)
    if backup_limit > 0:
        backups = sorted(
            [f for f in os.listdir(base_dir) if f.startswith(name + "_backup_")],
            reverse=True
        )
        for old_backup in backups[backup_limit:]:
            try:
                os.remove(os.path.join(base_dir, old_backup))
            except Exception as e:
                print(f"Warning: Could not remove old backup {old_backup}: {e}")

def save_json_atomically(filename, data):
    file = QSaveFile(filename)
    if file.open(QIODevice.WriteOnly | QIODevice.Text):
        try:
            json_str = json.dumps(data, indent=2)
            file.write(bytes(json_str, encoding='utf-8'))
        except Exception as e:
            print("Error while saving JSON:", e)
            file.cancelWriting()
            return
        if not file.commit():
            print("Failed to commit file")
        else:
            backup_before_save(filename)
    else:
        print("Could not open file for writing")


def save_project(
    filename,
    annotations,
    class_colors,
    video_path=None,
    current_frame=0,
    frame_annotations=None,
    class_attributes=None,
    current_style=None,
    auto_show_attribute_dialog=True,
    use_previous_attributes=True,
    duplicate_frames_enabled=False,
    frame_hashes=None,
    duplicate_frames_cache=None,
    image_dataset_info=None,
    tracking_mode_enabled=False,
    interpolation_mode_active=False,
    verification_mode_enabled=False,
    annotations_imported_list=None,

):
    """
    Save project to a JSON file.

    Args:
        filename (str): Path to save the project file
        annotations (list): List of annotation objects
        class_colors (dict): Dictionary mapping class names to colors
        video_path (str, optional): Path to the video file
        current_frame (int, optional): Current frame number
        frame_annotations (dict, optional): Dictionary mapping frame numbers to annotations
        class_attributes (dict, optional): Dictionary of class attribute configurations
        current_style (str, optional): Current UI style
        auto_show_attribute_dialog (bool, optional): Whether to show attribute dialog for new annotations
        use_previous_attributes (bool, optional): Whether to use previous annotation attributes as default
        duplicate_frames_enabled (bool, optional): Whether duplicate frame detection is enabled
        frame_hashes (dict, optional): Dictionary mapping frame numbers to hash values
        duplicate_frames_cache (dict, optional): Dictionary mapping hash values to lists of frame numbers
        image_dataset_info (dict, optional): Information about image dataset if applicable
        tracking_mode_enabled (bool, optional): Whether tracking mode is enabled
        interpolation_mode_active (bool, optional): Whether interpolation mode is active
        verification_mode_enabled (bool, optional): Whether verification mode is enabled
    """
    # Convert annotations to serializable format
    serialized_annotations = []
    for annotation in annotations:
        serialized_annotations.append(annotation.to_dict())

    # Convert class colors to serializable format
    serialized_colors = {}
    for class_name, color in class_colors.items():
        serialized_colors[class_name] = [color.red(), color.green(), color.blue()]

    # Convert frame annotations to serializable format
    serialized_frame_annotations = {}
    if frame_annotations:
        for frame_num, frame_anns in frame_annotations.items():
            serialized_frame_annotations[str(frame_num)] = [
                ann.to_dict() for ann in frame_anns
            ]

    # Create project data dictionary
    project_data = {
        "viat_project_identifier": "VIAT_PROJECT_FILE",
        "version": "1.0",
        "annotations": serialized_annotations,
        "class_colors": serialized_colors,
        "video_path": video_path,
        "current_frame": current_frame,
        "frame_annotations": serialized_frame_annotations,
        "class_attributes": class_attributes,
        "current_style": current_style,
        "auto_show_attribute_dialog": auto_show_attribute_dialog,
        "use_previous_attributes": use_previous_attributes,
        "duplicate_frames_enabled": duplicate_frames_enabled,
        "timestamp": datetime.datetime.now().isoformat(),
        "tracking_mode_enabled": tracking_mode_enabled,
        "interpolation_mode_active": interpolation_mode_active,
        "verification_mode_enabled": verification_mode_enabled,
        "annotations_imported_list": annotations_imported_list,
    }

    # Add frame hashes if available
    if frame_hashes:
        # Convert frame numbers from int to str for JSON serialization
        serialized_frame_hashes = {str(k): v for k, v in frame_hashes.items()}
        project_data["frame_hashes"] = serialized_frame_hashes

    # Add duplicate frames cache if available
    if duplicate_frames_cache:
        project_data["duplicate_frames_cache"] = duplicate_frames_cache

    # Add image dataset info if available
    if image_dataset_info:
        project_data["image_dataset_info"] = image_dataset_info

    # Save to file
    save_json_atomically(filename, project_data)

    # Update recent projects list
    update_recent_projects(filename)

def load_project(filename, bbox_class):
    """
    Load project from a JSON file.

    Args:
        filename (str): Path to the project file
        bbox_class (class): Class to use for bounding box objects

    Returns:
        tuple: (annotations, class_colors, video_path, current_frame, frame_annotations,
                class_attributes, current_style, auto_show_attribute_dialog, use_previous_attributes,
                duplicate_frames_enabled, frame_hashes, duplicate_frames_cache, image_dataset_info,
                tracking_mode_enabled, interpolation_mode_active, verification_mode_enabled)
    """
    with open(filename, "r") as f:
        project_data = json.load(f)

    # Check if this is a valid VIAT project file
    if "viat_project_identifier" not in project_data:
        raise ValueError("Not a valid VIAT project file")

    # Load annotations
    annotations = []
    for ann_data in project_data.get("annotations", []):
        annotation = bbox_class.from_dict(ann_data)
        annotations.append(annotation)

    # Load class colors
    class_colors = {}
    for class_name, color_values in project_data.get("class_colors", {}).items():
        class_colors[class_name] = QColor(*color_values)

    # Load video path
    video_path = project_data.get("video_path")

    # Load current frame
    current_frame = project_data.get("current_frame", 0)

    # Load frame annotations
    frame_annotations = {}
    for frame_num, frame_anns in project_data.get("frame_annotations", {}).items():
        frame_annotations[int(frame_num)] = [
            bbox_class.from_dict(ann_data) for ann_data in frame_anns
        ]
    
    # Load class attributes
    class_attributes = project_data.get("class_attributes", {})

    # Load UI style
    current_style = project_data.get("current_style", "DarkModern")

    # Load annotation settings
    auto_show_attribute_dialog = project_data.get("auto_show_attribute_dialog", True)
    use_previous_attributes = project_data.get("use_previous_attributes", True)

    # Load duplicate frame detection settings
    duplicate_frames_enabled = project_data.get("duplicate_frames_enabled", False)

    # Load frame hashes
    frame_hashes = {}
    for frame_num, hash_value in project_data.get("frame_hashes", {}).items():
        frame_hashes[int(frame_num)] = hash_value

    # Load duplicate frames cache
    duplicate_frames_cache = project_data.get("duplicate_frames_cache", {})

    # Load image dataset info
    image_dataset_info = project_data.get("image_dataset_info", None)

    # Load mode states
    tracking_mode_enabled = project_data.get("tracking_mode_enabled", False)
    interpolation_mode_active = project_data.get("interpolation_mode_active", False)
    verification_mode_enabled = project_data.get("verification_mode_enabled", False)
    annotations_imported_list = project_data.get("annotations_imported_list", [])

    # Update recent projects list
    update_recent_projects(filename)

    return (
        annotations,
        class_colors,
        video_path,
        current_frame,
        frame_annotations,
        class_attributes,
        current_style,
        auto_show_attribute_dialog,
        use_previous_attributes,
        duplicate_frames_enabled,
        frame_hashes,
        duplicate_frames_cache,
        image_dataset_info,
        tracking_mode_enabled,
        interpolation_mode_active,
        verification_mode_enabled,
        annotations_imported_list
    )

def get_recent_projects():
    """
    Get list of recent projects.

    Returns:
        list: List of recent project file paths
    """
    config_dir = get_config_directory()
    recent_projects_file = os.path.join(config_dir, "recent_projects.json")

    if os.path.exists(recent_projects_file):
        try:
            with open(recent_projects_file, "r") as f:
                recent_projects = json.load(f)

            # Filter out projects that no longer exist
            recent_projects = [p for p in recent_projects if os.path.exists(p)]
            return recent_projects
        except Exception:
            return []
    else:
        return []


def update_recent_projects(project_file, max_projects=10):
    """
    Update the list of recent projects.

    Args:
        project_file (str): Path to the project file to add
        max_projects (int): Maximum number of recent projects to keep
    """
    config_dir = get_config_directory()
    recent_projects_file = os.path.join(config_dir, "recent_projects.json")

    # Create config directory if it doesn't exist
    if not os.path.exists(config_dir):
        os.makedirs(config_dir)

    # Load existing recent projects
    recent_projects = []
    if os.path.exists(recent_projects_file):
        try:
            with open(recent_projects_file, "r") as f:
                recent_projects = json.load(f)
        except Exception:
            recent_projects = []

    # Add current project to the top of the list
    if project_file in recent_projects:
        recent_projects.remove(project_file)
    recent_projects.insert(0, project_file)

    # Limit to max_projects
    recent_projects = recent_projects[:max_projects]

    # Save updated list
    save_json_atomically(recent_projects_file,recent_projects)


def get_last_project():
    """
    Get the most recently used project.

    Returns:
        str: Path to the most recent project file, or None if no recent projects
    """
    recent_projects = get_recent_projects()
    if recent_projects:
        return recent_projects[0]
    return None


def get_config_directory():
    """
    Get the configuration directory for the application.

    Returns:
        str: Path to the configuration directory
    """
    # Use platform-specific config directory
    if os.name == "nt":  # Windows
        config_dir = os.path.join(os.environ["APPDATA"], "VideoAnnotationTool")
    else:  # macOS, Linux, etc.
        config_dir = os.path.join(
            os.path.expanduser("~"), ".config", "VideoAnnotationTool"
        )

    return config_dir


def save_last_state(state_data):
    """
    Save the last application state.

    Args:
        state_data (dict): Dictionary containing application state data
    """
    config_dir = get_config_directory()
    state_file = os.path.join(config_dir, "last_state.json")

    # Create config directory if it doesn't exist
    if not os.path.exists(config_dir):
        os.makedirs(config_dir)

    # Save state data

    save_json_atomically(state_file, state_data)



def load_last_state():
    """
    Load the last application state.

    Returns:
        dict: Dictionary containing application state data, or None if no state file
    """
    config_dir = get_config_directory()
    state_file = os.path.join(config_dir, "last_state.json")

    if os.path.exists(state_file):
        try:
            with open(state_file, "r") as f:
                return json.load(f)
        except Exception:
            return None
    else:
        return None


def export_annotations(
    filename, annotations, image_width, image_height, format_type="coco"
):
    """Export annotations to various formats"""
    if format_type == "coco":
        export_coco(filename, annotations, image_width, image_height)
    elif format_type == "yolo":
        export_yolo(filename, annotations, image_width, image_height)
    elif format_type == "pascal_voc":
        export_pascal_voc(filename, annotations, image_width, image_height)
    elif format_type == "raya":
        export_raya_annotations(filename, annotations)
    else:
        raise ValueError(f"Unsupported export format: {format_type}")


def export_coco(filename, annotations, image_width, image_height):
    """Export annotations in COCO format"""
    data = {
        "images": [
            {
                "id": 1,
                "width": image_width,
                "height": image_height,
                "file_name": os.path.basename(filename).replace(".json", ".jpg"),
            }
        ],
        "annotations": [],
        "categories": [],
    }

    # Create categories
    categories = {}
    category_id = 1

    for annotation in annotations:
        if annotation.class_name not in categories:
            categories[annotation.class_name] = category_id
            data["categories"].append(
                {
                    "id": category_id,
                    "name": annotation.class_name,
                    "supercategory": "none",
                }
            )
            category_id += 1

    # Create annotations
    annotation_id = 1
    for annotation in annotations:
        x, y, w, h = (
            annotation.rect.x(),
            annotation.rect.y(),
            annotation.rect.width(),
            annotation.rect.height(),
        )

        ann_data = {
            "id": annotation_id,
            "image_id": 1,
            "category_id": categories[annotation.class_name],
            "bbox": [x, y, w, h],
            "area": w * h,
            "segmentation": [],
            "iscrowd": 0,
        }

        # Add attributes
        if hasattr(annotation, "attributes") and annotation.attributes:
            attributes = {}
            for attr_name, attr_value in annotation.attributes.items():
                if attr_value != -1:  # Only export non-default attributes
                    attributes[attr_name] = attr_value

            if attributes:
                ann_data["attributes"] = attributes

        data["annotations"].append(ann_data)
        annotation_id += 1

    # Save to file
    update_recent_projects(filename,data)
    


def export_yolo(filename, annotations, image_width, image_height):
    """Export annotations in YOLO format"""
    # Create class mapping
    classes = sorted(set(a.class_name for a in annotations))
    class_to_id = {cls: i for i, cls in enumerate(classes)}

    # Save class mapping
    classes_file = filename.replace(".txt", "_classes.txt")
    with open(classes_file, "w") as f:
        for cls in classes:
            f.write(f"{cls}\n")

    # Save annotations
    with open(filename, "w") as f:
        for annotation in annotations:
            # Convert to YOLO format: class_id, x_center, y_center, width, height
            # All values normalized to [0, 1]
            class_id = class_to_id[annotation.class_name]
            x = annotation.rect.x()
            y = annotation.rect.y()
            w = annotation.rect.width()
            h = annotation.rect.height()

            # Convert to center coordinates and normalize
            x_center = (x + w / 2) / image_width
            y_center = (y + h / 2) / image_height
            norm_width = w / image_width
            norm_height = h / image_height

            line = f"{class_id} {x_center:.6f} {y_center:.6f} {norm_width:.6f} {norm_height:.6f}"

            # Add attributes as comments (since YOLO format doesn't support attributes directly)
            if hasattr(annotation, "attributes") and annotation.attributes:
                attr_parts = []
                for attr_name, attr_value in annotation.attributes.items():
                    if attr_value != -1:  # Only export non-default attributes
                        attr_parts.append(f"{attr_name}:{attr_value}")

                if attr_parts:
                    line += f" # {','.join(attr_parts)}"

            f.write(line + "\n")

    # Save attributes in a separate file for reference
    attributes_file = filename.replace(".txt", "_attributes.json")
    attributes_data = {}

    for i, annotation in enumerate(annotations):
        if hasattr(annotation, "attributes") and annotation.attributes:
            attrs = {}
            for attr_name, attr_value in annotation.attributes.items():
                if attr_value != -1:  # Only export non-default attributes
                    attrs[attr_name] = attr_value

            if attrs:
                attributes_data[i] = {
                    "class": annotation.class_name,
                    "attributes": attrs,
                }

    if attributes_data:
        update_recent_projects(attributes_file,attributes_data)


def export_pascal_voc(filename, annotations, image_width, image_height):
    """Export annotations in Pascal VOC XML format"""
    from xml.etree.ElementTree import Element, SubElement, tostring
    from xml.dom import minidom

    root = Element("annotation")

    # Add basic image info
    folder = SubElement(root, "folder")
    folder.text = "images"

    filename_elem = SubElement(root, "filename")
    filename_elem.text = os.path.basename(filename).replace(".xml", ".jpg")

    size = SubElement(root, "size")
    width_elem = SubElement(size, "width")
    width_elem.text = str(image_width)
    height_elem = SubElement(size, "height")
    height_elem.text = str(image_height)
    depth = SubElement(size, "depth")
    depth.text = "3"

    # Add each object (annotation)
    for annotation in annotations:
        obj = SubElement(root, "object")

        name = SubElement(obj, "name")
        name.text = annotation.class_name

        pose = SubElement(obj, "pose")
        pose.text = "Unspecified"

        truncated = SubElement(obj, "truncated")
        truncated.text = "0"

        difficult = SubElement(obj, "difficult")
        difficult.text = "0"

        bndbox = SubElement(obj, "bndbox")
        xmin = SubElement(bndbox, "xmin")
        xmin.text = str(annotation.rect.x())
        ymin = SubElement(bndbox, "ymin")
        ymin.text = str(annotation.rect.y())
        xmax = SubElement(bndbox, "xmax")
        xmax.text = str(annotation.rect.x() + annotation.rect.width())
        ymax = SubElement(bndbox, "ymax")
        ymax.text = str(annotation.rect.y() + annotation.rect.height())

        # Add attributes
        if hasattr(annotation, "attributes") and annotation.attributes:
            attributes = SubElement(obj, "attributes")
            for attr_name, attr_value in annotation.attributes.items():
                if attr_value != -1:  # Only export non-default attributes
                    attr = SubElement(attributes, "attribute")
                    attr_name_elem = SubElement(attr, "name")
                    attr_name_elem.text = attr_name
                    attr_value_elem = SubElement(attr, "value")
                    attr_value_elem.text = str(attr_value)

    # Convert to pretty XML
    rough_string = tostring(root, "utf-8")
    reparsed = minidom.parseString(rough_string)
    pretty_xml = reparsed.toprettyxml(indent="  ")

    # Save to file
    with open(filename, "w") as f:
        f.write(pretty_xml)


def export_raya_annotations(filename, annotations):
    """
    Export annotations to Raya text format.

    Format: [class,x,y,width,height,size,quality,Difficult(optional)];
    If no detection: []

    Args:
        filename (str): Path to save the Raya text file
        annotations (list): List of annotation objects
    """
    try:
        # Group annotations by frame
        annotations_by_frame = {}
        for annotation in annotations:
            frame_num = getattr(annotation, "frame", 0)
            if frame_num not in annotations_by_frame:
                annotations_by_frame[frame_num] = []
            annotations_by_frame[frame_num].append(annotation)

        # Get the maximum frame number
        max_frame = max(annotations_by_frame.keys()) if annotations_by_frame else 0

        # Create lines for each frame
        lines = ["[]"] * (max_frame + 1)

        # Fill in annotations for frames that have them
        for frame_num, frame_annotations in annotations_by_frame.items():
            if not frame_annotations:
                lines[frame_num] = "[]"
                continue

            # Format annotations for this frame
            frame_str = ""
            for annotation in frame_annotations:

                # Get annotation properties
                rect = annotation.rect
                class_id = 0  # Default to 0 for Quad class
                x = rect.x()
                y = rect.y()
                width = rect.width()
                height = rect.height()
                size = annotation.attributes.get("Size", -1)
                quality = annotation.attributes.get("Quality", -1)
                Difficult = annotation.attributes.get("Difficult", -1)

                # Format the annotation with a semicolon after each one
                if Difficult == -1:
                    frame_str += (
                        f"[{class_id},{x},{y},{width},{height},{size},{quality}];"
                    )
                else:
                    frame_str += f"[{class_id},{x},{y},{width},{height},{size},{quality},{Difficult}];"

            lines[frame_num] = frame_str

        # Write to file
        with open(filename, "w") as f:
            for line in lines:
                f.write(line + "\n")

    except Exception as e:
        raise Exception(f"Error exporting to Raya format: {str(e)}")


def export_image_dataset_pascal_voc(
    output_dir, image_files, frame_annotations, canvas_pixmap
):
    """
    Export annotations for an image dataset in Pascal VOC format.

    Args:
        output_dir (str): Directory to save Pascal VOC XML files
        image_files (list): List of image file paths
        frame_annotations (dict): Frame number to annotation list
        canvas_pixmap (QPixmap): Canvas pixmap to get image size
    """

    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Get image dimensions
    image_width = canvas_pixmap.width() if canvas_pixmap else 640
    image_height = canvas_pixmap.height() if canvas_pixmap else 480

    # Process each image
    for frame_num, image_path in enumerate(image_files):
        # Skip if no annotations for this frame
        if frame_num not in frame_annotations or not frame_annotations[frame_num]:
            continue

        # Create XML structure
        annotation = ET.Element("annotation")

        # Add folder and filename
        folder = ET.SubElement(annotation, "folder")
        folder.text = os.path.basename(os.path.dirname(image_path))

        filename_elem = ET.SubElement(annotation, "filename")
        filename_elem.text = os.path.basename(image_path)

        path_elem = ET.SubElement(annotation, "path")
        path_elem.text = image_path

        # Add source information
        source = ET.SubElement(annotation, "source")
        database = ET.SubElement(source, "database")
        database.text = "VIAT"

        # Add size information
        size = ET.SubElement(annotation, "size")
        width_elem = ET.SubElement(size, "width")
        width_elem.text = str(image_width)
        height_elem = ET.SubElement(size, "height")
        height_elem.text = str(image_height)
        depth = ET.SubElement(size, "depth")
        depth.text = "3"  # Assuming RGB images

        # Add segmented flag
        segmented = ET.SubElement(annotation, "segmented")
        segmented.text = "0"

        # Add objects (annotations)
        for annotation_obj in frame_annotations[frame_num]:
            obj = ET.SubElement(annotation, "object")

            # Add class name
            name = ET.SubElement(obj, "name")
            name.text = annotation_obj.class_name

            # Add pose
            pose = ET.SubElement(obj, "pose")
            pose.text = "Unspecified"

            # Add truncated flag
            truncated = ET.SubElement(obj, "truncated")
            truncated.text = "0"

            # Add difficult flag
            difficult = ET.SubElement(obj, "difficult")
            difficult.text = "0"

            # Add bounding box
            rect = annotation_obj.rect
            bndbox = ET.SubElement(obj, "bndbox")

            xmin = ET.SubElement(bndbox, "xmin")
            xmin.text = str(rect.x())

            ymin = ET.SubElement(bndbox, "ymin")
            ymin.text = str(rect.y())

            xmax = ET.SubElement(bndbox, "xmax")
            xmax.text = str(rect.x() + rect.width())

            ymax = ET.SubElement(bndbox, "ymax")
            ymax.text = str(rect.y() + rect.height())

            # Add attributes as custom elements
            for attr_name, attr_value in annotation_obj.attributes.items():
                if attr_name in ["Size", "Quality"] and attr_value != -1:
                    attr_elem = ET.SubElement(obj, "attribute")
                    attr_name_elem = ET.SubElement(attr_elem, "name")
                    attr_name_elem.text = attr_name
                    attr_value_elem = ET.SubElement(attr_elem, "value")
                    attr_value_elem.text = str(attr_value)

        # Create XML file
        xml_str = ET.tostring(annotation, encoding="utf-8")
        dom = minidom.parseString(xml_str)
        pretty_xml = dom.toprettyxml(indent="  ")

        # Get output filename
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        xml_filename = os.path.join(output_dir, f"{base_name}.xml")

        # Write to file
        with open(xml_filename, "w") as f:
            f.write(pretty_xml)

    # Create a README file explaining the format
    with open(os.path.join(output_dir, "README.txt"), "w") as f:
        f.write("Pascal VOC Format Export from VIAT\n")
        f.write("================================\n\n")
        f.write("This directory contains:\n")
        f.write("- One .xml file per image with annotations in Pascal VOC format\n\n")
        f.write("Pascal VOC format includes:\n")
        f.write("- Object class names\n")
        f.write("- Bounding box coordinates (xmin, ymin, xmax, ymax)\n")
        f.write("- Additional attributes like Size and Quality\n")


def export_image_dataset_yolo(output_dir, image_files, frame_annotations, class_colors):
    """
    Export annotations for an image dataset in YOLO format.

    Args:
        output_dir (str): Directory to save YOLO .txt files
        image_files (list): List of image file paths
        frame_annotations (dict): Frame number to annotation list
        class_colors (dict): Dictionary mapping class names to QColor (for class order)
    """
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Get class list and mapping
    class_list = list(class_colors.keys())
    class_to_id = {cls: i for i, cls in enumerate(class_list)}

    # Write classes.txt
    classes_file = os.path.join(output_dir, "classes.txt")
    with open(classes_file, "w") as f:
        for cls in class_list:
            f.write(f"{cls}\n")

    # Process each image
    for frame_num, image_path in enumerate(image_files):
        # Skip if no annotations for this frame
        if frame_num not in frame_annotations or not frame_annotations[frame_num]:
            continue

        # Get output .txt filename (same basename as image)
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        txt_filename = os.path.join(output_dir, f"{base_name}.txt")

        try:
            img = cv2.imread(image_path)
            if img is not None:
                image_height, image_width = img.shape[:2]
            else:

                image_width, image_height = 640, 480
        except Exception:
            image_width, image_height = 640, 480

        with open(txt_filename, "w") as f:
            for annotation in frame_annotations[frame_num]:
                class_id = class_to_id.get(annotation.class_name, 0)
                rect = annotation.rect

                x = rect.x()
                y = rect.y()
                w = rect.width()
                h = rect.height()
                x_center = (x + w / 2) / image_width
                y_center = (y + h / 2) / image_height
                norm_w = w / image_width
                norm_h = h / image_height
                f.write(
                    f"{class_id} {x_center:.6f} {y_center:.6f} {norm_w:.6f} {norm_h:.6f}\n"
                )


def export_image_dataset_coco(
    filename, image_files, frame_annotations, class_colors, image_width, image_height
):
    """
    Export annotations for an image dataset in COCO format.

    Args:
        filename (str): Path to save the COCO JSON file
        image_files (list): List of image file paths
        frame_annotations (dict): Frame number to annotation list
        class_colors (dict): Dictionary mapping class names to QColor (for class order)
        image_width (int): Width of the images
        image_height (int): Height of the images
    """
    import json
    from datetime import datetime
    import os

    # Initialize COCO format structure
    coco_data = {
        "info": {
            "description": "VIAT Exported Annotations",
            "url": "",
            "version": "1.0",
            "year": datetime.now().year,
            "contributor": "VIAT",
            "date_created": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        },
        "licenses": [
            {
                "id": 1,
                "name": "Unknown",
                "url": "",
            }
        ],
        "images": [],
        "annotations": [],
        "categories": [],
    }

    # Create category mapping
    class_list = list(class_colors.keys())
    category_id_map = {class_name: idx + 1 for idx, class_name in enumerate(class_list)}
    for class_name, cat_id in category_id_map.items():
        coco_data["categories"].append(
            {"id": cat_id, "name": class_name, "supercategory": "none"}
        )

    # Add images and annotations
    annotation_id = 1

    for image_id, image_path in enumerate(image_files, 1):
        # Add image info
        image_filename = os.path.basename(image_path)
        coco_data["images"].append(
            {
                "id": image_id,
                "license": 1,
                "file_name": image_filename,
                "height": image_height,
                "width": image_width,
                "date_captured": "",
                "frame_id": image_id - 1,  # Store frame number for compatibility
            }
        )

        # Add annotations for this image
        frame_num = image_id - 1
        if frame_num in frame_annotations:
            for annotation in frame_annotations[frame_num]:
                # Get category id
                category_id = category_id_map.get(annotation.class_name, 1)

                # Get bounding box in COCO format [x, y, width, height]
                rect = annotation.rect
                bbox = [rect.x(), rect.y(), rect.width(), rect.height()]

                # Calculate area
                area = rect.width() * rect.height()

                # Create annotation entry
                coco_annotation = {
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": category_id,
                    "bbox": bbox,
                    "area": area,
                    "segmentation": [],
                    "iscrowd": 0,
                }

                # Add attributes if available
                for attr_name, attr_value in annotation.attributes.items():
                    coco_annotation[attr_name.lower()] = attr_value

                coco_data["annotations"].append(coco_annotation)
                annotation_id += 1

    # Write to file
    save_json_atomically(filename,coco_data)
    


def export_standard_annotations(
    filename,
    frame_annotations,
    canvas_annotations,
    export_format,
    image_width,
    image_height,
):
    """
    Export annotations using the standard export function.

    Args:
        filename (str): Output file path
        frame_annotations (dict): Frame number to annotation list
        canvas_annotations (list): Current frame annotation list (used if frame_annotations is empty)
        export_format (str): Format type ("coco", "yolo", "pascal_voc", "raya")
        image_width (int): Image width
        image_height (int): Image height
    """
    # Collect all annotations from all frames
    all_annotations = []
    for frame_num, annotations in frame_annotations.items():
        for annotation in annotations:
            annotation_copy = annotation
            annotation_copy.frame = frame_num
            all_annotations.append(annotation_copy)

    if not all_annotations and canvas_annotations:
        all_annotations = canvas_annotations

    export_annotations(
        filename, all_annotations, image_width, image_height, export_format
    )


def import_coco_annotations(filename, bbox_class):
    """
    Import annotations from a COCO JSON file.

    Args:
        filename (str): Path to the COCO JSON file
        bbox_class (class): Class to use for bounding box objects

    Returns:
        list: List of annotation objects
    """
    with open(filename, "r") as f:
        data = json.load(f)

    # Build category id to name mapping
    categories = {cat["id"]: cat["name"] for cat in data.get("categories", [])}

    # Build image id to file name mapping (not always needed)
    images = {img["id"]: img for img in data.get("images", [])}

    annotations = []
    for ann in data.get("annotations", []):
        image_id = ann.get("image_id")
        category_id = ann.get("category_id")
        bbox = ann.get("bbox", [0, 0, 0, 0])
        class_name = categories.get(category_id, "unknown")
        score = ann.get("score", None)

        x, y, w, h = bbox
        # Create annotation object
        annotation = bbox_class(
            x=int(x),
            y=int(y),
            width=int(w),
            height=int(h),
            class_name=class_name,
            attributes=ann.get("attributes", {}),
            source="detected",
            score=score,
        )
        # Optionally, set frame/image info if needed
        if "frame_id" in images.get(image_id, {}):
            annotation.frame = images[image_id]["frame_id"]
        annotations.append(annotation)

    return annotations


def detect_annotation_format(filename):
    """
    Detect the annotation format based on file extension and content.

    Returns:
        str: Detected format ("COCO", "YOLO", "Pascal VOC", "Raya", "RayaYOLO") or None if not detected
    """
    # Check file extension
    ext = os.path.splitext(filename)[1].lower()

    # Read file content
    try:
        with open(filename, "r") as f:
            content = f.read()
    except (IOError, OSError):
        return None

    # Detect format based on extension and content
    if ext == ".json":
        if (
            '"images"' in content
            and '"annotations"' in content
            and '"categories"' in content
        ):
            return "COCO"
    elif ext == ".xml":
        if "<annotation>" in content and "<object>" in content:
            return "Pascal VOC"
    elif ext == ".txt":
        lines = content.strip().split("\n")

        # More flexible Raya format detection
        if all("[]" in line or ('[' and '];' in line) for line in lines):
            return "Raya"

        # Check for RayaYOLO format
        if lines and all(
            line.strip() == "[]"
            or (line.strip().startswith("[[") and line.strip().endswith("]]"))
            for line in lines
            if line.strip()
        ):
            return "RayaYOLO"

        # YOLO format typically has space-separated numbers (class x y w h)
        if lines and all(
            len(line.split()) == 5 and line.split()[0].isdigit()
            for line in lines
            if line.strip()
        ):
            return "YOLO"

    # If no format detected, try more detailed analysis
    if ext == ".json":
        try:
            import json

            data = json.loads(content)
            if isinstance(data, dict):
                if "annotations" in data and "images" in data:
                    return "COCO"
        except json.JSONDecodeError:
            pass

    return None


def import_yolo_annotations(
    filename, image_width, image_height, bbox_class, class_colors=None
):
    """
    Import annotations from YOLO format.

    Args:
        filename (str): Path to the YOLO txt file
        image_width (int): Width of the image/video frame
        image_height (int): Height of the image/video frame
        bbox_class (class): Class to use for bounding box objects
        class_colors (dict, optional): Dictionary mapping class names to colors

    Returns:
        list: List of annotation objects for the current frame
    """
    # YOLO format: class_id x_center y_center width height
    # All values are normalized [0-1]

    if class_colors is None:
        class_colors = {}

    # First, try to find a classes.txt file in the same directory
    classes_file = os.path.join(os.path.dirname(filename), "classes.txt")
    class_names = []

    if os.path.exists(classes_file):
        with open(classes_file, "r") as f:
            class_names = [line.strip() for line in f.readlines()]

    # Read annotations
    with open(filename, "r") as f:
        lines = f.readlines()

    # Process each line
    annotations = []
    for line in lines:
        parts = line.strip().split()
        if len(parts) != 5:
            continue

        try:
            class_id = int(parts[0])
            x_center = float(parts[1]) * image_width
            y_center = float(parts[2]) * image_height
            width = float(parts[3]) * image_width
            height = float(parts[4]) * image_height
            score = None
            if len(parts) > 5:
                try:
                    score = float(parts[5])
                except ValueError:
                    pass

            # Calculate top-left corner from center
            x = x_center - (width / 2)
            y = y_center - (height / 2)

            # Create QRect
            rect = QRect(int(x), int(y), int(width), int(height))

            # Get class name
            if class_id < len(class_names):
                class_name = class_names[class_id]
            else:
                class_name = f"class_{class_id}"

            # Get or create color for this class
            if class_name not in class_colors:
                class_colors[class_name] = QColor(
                    random.randint(0, 255),
                    random.randint(0, 255),
                    random.randint(0, 255),
                )
            color = class_colors[class_name]

            # Create attributes dictionary
            attributes = {"Size": -1, "Quality": -1}

            # Create bounding box
            bbox_obj = bbox_class(
                rect, class_name, attributes, color, source="detected", score=score
            )
            annotations.append(bbox_obj)

        except (ValueError, IndexError) as e:
            print(f"Error parsing YOLO line: {line}. Error: {e}")

    return annotations


def import_pascal_voc_annotations(
    filename, image_width, image_height, bbox_class, class_colors=None
):
    """
    Import annotations from Pascal VOC XML format.

    Args:
        filename (str): Path to the Pascal VOC XML file
        image_width (int): Width of the image/video frame
        image_height (int): Height of the image/video frame
        bbox_class (class): Class to use for bounding box objects
        class_colors (dict, optional): Dictionary mapping class names to colors

    Returns:
        list: List of annotation objects for the current frame
    """
    import xml.etree.ElementTree as ET

    if class_colors is None:
        class_colors = {}

    annotations = []
    try:
        tree = ET.parse(filename)
        root = tree.getroot()

        # Process each object
        for obj in root.findall("./object"):
            class_name = obj.find("name").text

            # Get bounding box
            bndbox = obj.find("bndbox")
            xmin = int(float(bndbox.find("xmin").text))
            ymin = int(float(bndbox.find("ymin").text))
            xmax = int(float(bndbox.find("xmax").text))
            ymax = int(float(bndbox.find("ymax").text))
            score = None
            confidence = obj.find("confidence")
            if confidence is not None:
                try:
                    score = float(confidence.text)
                except (ValueError, TypeError):
                    pass
            # Create QRect
            rect = QRect(xmin, ymin, xmax - xmin, ymax - ymin)

            # Get or create color for this class
            if class_name not in class_colors:
                class_colors[class_name] = QColor(
                    random.randint(0, 255),
                    random.randint(0, 255),
                    random.randint(0, 255),
                )
            color = class_colors[class_name]

            # Create attributes dictionary
            attributes = {"Size": -1, "Quality": -1}

            # Check for additional attributes
            for attr in obj.findall("./attribute"):
                name = attr.find("name")
                value = attr.find("value")
                if name is not None and value is not None:
                    attributes[name.text] = value.text

            # Create bounding box
            bbox_obj = bbox_class(
                rect, class_name, attributes, color, source="detected", score=score
            )
            annotations.append(bbox_obj)

    except Exception as e:
        raise Exception(f"Error parsing Pascal VOC XML: {str(e)}")

    return annotations


def import_raya_annotations(filename, bbox_class, class_colors=None):
    """
    Import annotations from Raya text format.

    Format: [class,x,y,width,height,size,quality,Difficult(optional)];
    If no detection: []

    Args:
        filename (str): Path to the Raya text file
        bbox_class (class): Class to use for bounding box objects
        class_colors (dict, optional): Dictionary mapping class names to colors

    Returns:
        dict: Dictionary mapping frame numbers to lists of annotation objects
    """
    from PyQt5.QtCore import QRect
    from PyQt5.QtGui import QColor
    import random

    if class_colors is None:
        class_colors = {}

    # Ensure "Quad" class has a color
    if "Quad" not in class_colors:
        class_colors["Quad"] = QColor(255, 0, 0)  # Red color for Quad class

    frame_annotations = {}

    # Track unique class numbers to determine if we have a single class
    unique_class_nums = set()

    try:
        # First pass: scan the file to identify all unique class numbers
        with open(filename, "r") as f:
            lines = f.readlines()

        for line in lines:
            line = line.strip()
            # Skip empty frames or frames with no detections
            if not line or line == "[]":
                continue
            if not ("[" in line and "]" in line):
                continue

            start_idx = line.find("[")
            end_idx = line.rfind("]")
            if start_idx == -1 or end_idx == -1 or start_idx >= end_idx:
                continue

            content = line[start_idx + 1 : end_idx]
            annotations = content.split(";")
            for annotation in annotations:
                if not annotation.strip():
                    continue
                # Parse the annotation values
                parts = annotation.split(",")
                # Ensure we have at least the minimum required fields
                if len(parts) < 5:
                    continue

                try:
                    # Remove any remaining brackets
                    parts = [p.strip("[]") for p in parts]
                    class_num = int(parts[0])
                    unique_class_nums.add(class_num)
                except (ValueError, IndexError):
                    continue

        # Determine if we have a single class
        single_class_file = len(unique_class_nums) == 1

        # Second pass: actually import the annotations
        for frame_num, line in enumerate(lines):
            frame_annots = []
            line = line.strip()
            # Skip empty frames or frames with no detections
            if not line or line == "[]":
                continue
            if not ("[" in line and "]" in line):
                continue

            start_idx = line.find("[")
            end_idx = line.rfind("]")
            if start_idx == -1 or end_idx == -1 or start_idx >= end_idx:
                continue

            content = line[start_idx + 1 : end_idx]
            annotations = content.split(";")

            for annotation in annotations:
                if not annotation.strip():
                    continue
                # Parse the annotation values
                parts = annotation.split(",")

                # Ensure we have at least the minimum required fields
                if len(parts) < 5:
                    continue
                try:
                    # Remove any remaining brackets
                    parts = [p.strip("[]") for p in parts]

                    class_num = int(parts[0])
                    x = float(parts[1])
                    y = float(parts[2])
                    width = float(parts[3])
                    height = float(parts[4])
                    size = float(parts[5]) if len(parts) > 5 else 100.0
                    quality = float(parts[6]) if len(parts) > 6 else 100.0
                    Difficult = float(parts[7]) if len(parts) > 7 else 0.0

                    # Create class name based on class ID and whether this is a single-class file
                    if single_class_file:
                        class_name = "Quad"  # Use Quad for single-class files
                    else:
                        class_name = "Quad" if class_num == 0 else f"class_{class_num}"

                    # Create QRect
                    rect = QRect(int(x), int(y), int(width), int(height))

                    # Get or create color for this class
                    if class_name not in class_colors:
                        class_colors[class_name] = QColor(
                            random.randint(0, 255),
                            random.randint(0, 255),
                            random.randint(0, 255),
                        )
                    color = class_colors[class_name]

                    # Create attributes dictionary
                    attributes = {
                        "Size": int(size),
                        "Quality": int(quality),
                    }
                    if Difficult != 0.0:
                        attributes["Difficult"] = int(Difficult)
                    # Create bounding box
                    bbox_obj = bbox_class(rect, class_name, attributes, color)
                    frame_annots.append(bbox_obj)
                except (ValueError, IndexError) as e:
                    print(f"Error parsing Raya annotation: {annotation}. Error: {e}")

            if frame_annots:
                frame_annotations[frame_num] = frame_annots

        return frame_annotations

    except Exception as e:
        print(f"Error parsing Raya text file: {str(e)}")
        raise Exception(f"Error parsing Raya text file: {str(e)}")


def import_raya_yolo_annotations(
    filename, image_width, image_height, bbox_class, class_colors=None
):
    """
    Import annotations from Raya YOLO format.

    Format:
    - Empty frames: []
    - Frames with predictions: [[cls,x1,y1,x2,y2,s], [cls,x1,y1,x2,y2,s], ...]

    Where:
    - cls: class ID
    - x1, y1: top-left corner coordinates
    - x2, y2: bottom-right corner coordinates
    - s: prediction score (ignored for annotation purposes)

    Args:
        filename (str): Path to the Raya YOLO text file
        image_width (int): Width of the image/video frame
        image_height (int): Height of the image/video frame
        bbox_class (class): Class to use for bounding box objects
        class_colors (dict, optional): Dictionary mapping class names to colors

    Returns:
        dict: Dictionary mapping frame numbers to lists of annotation objects
    """
    if class_colors is None:
        class_colors = {}

    # Ensure "Quad" class has a color
    if "Quad" not in class_colors:
        class_colors["Quad"] = QColor(255, 0, 0)  # Red color for Quad class

    frame_annotations = {}

    try:
        with open(filename, "r") as f:
            lines = f.readlines()

        # Process each line (each line represents a frame)
        for frame_num, line in enumerate(lines):
            line = line.strip()

            # Skip empty frames or frames with no detections
            if not line or line == "[]":
                continue

            # Extract content between the outermost brackets
            if not (line.startswith("[[") and line.endswith("]]")):
                continue

            content = line[2:-2]  # Remove outer [[ and ]]

            # Split by "], [" for multiple annotations
            if "], [" in content:
                annotations = content.split("], [")
            else:
                # Single annotation case
                annotations = [content]

            frame_annotations_list = []

            for annotation in annotations:
                # Clean up any remaining brackets
                annotation = annotation.strip("[]")

                # Parse the annotation values
                parts = annotation.split(",")

                # Ensure we have at least the minimum required fields (cls, x1, y1, x2, y2)
                if len(parts) < 5:
                    continue

                try:
                    # Parse values
                    class_id = int(float(parts[0]))
                    x1 = float(parts[1])
                    y1 = float(parts[2])
                    x2 = float(parts[3])
                    y2 = float(parts[4])
                    score = None
                    try:
                        score = float(parts[5])
                    except:
                        pass

                    # Calculate width and height
                    width = x2 - x1
                    height = y2 - y1

                    # Create class name - always use "Quad" for consistency
                    class_name = "Quad"

                    # Create QRect (ensure coordinates are valid)
                    if width <= 0 or height <= 0:
                        continue

                    rect = QRect(int(x1), int(y1), int(width), int(height))

                    # Create attributes dictionary
                    attributes = {"Size": -1, "Quality": -1}

                    # Create bounding box
                    bbox_obj = bbox_class(
                        rect,
                        class_name,
                        attributes,
                        class_colors[class_name],
                        source="detected",
                        score=score,
                    )
                    frame_annotations_list.append(bbox_obj)

                except (ValueError, IndexError) as e:
                    print(
                        f"Error parsing Raya YOLO annotation: {annotation}. Error: {e}"
                    )

            # Add to frame annotations
            if frame_annotations_list:
                frame_annotations[frame_num] = frame_annotations_list

    except Exception as e:
        raise Exception(f"Error parsing Raya YOLO text file: {str(e)}")

    return frame_annotations



def export_image_dataset_coco(
    filename, image_files, frame_annotations, class_colors, image_width, image_height
):
    """
    Export annotations for an image dataset in COCO format.

    Args:
        filename (str): Path to save the COCO JSON file
        image_files (list): List of image file paths
        frame_annotations (dict): Dictionary of annotations by frame
        class_colors (dict): Dictionary mapping class names to QColor (for class order)
        image_width (int): Width of the images
        image_height (int): Height of the images
    """
    import json
    from datetime import datetime
    import os
    import cv2

    # Initialize COCO format structure
    coco_data = {
        "info": {
            "description": "VIAT Exported Annotations",
            "url": "",
            "version": "1.0",
            "year": datetime.now().year,
            "contributor": "VIAT",
            "date_created": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        },
        "licenses": [
            {
                "id": 1,
                "name": "Unknown",
                "url": "",
            }
        ],
        "images": [],
        "annotations": [],
        "categories": [],
    }

    # Create category mapping
    class_list = list(class_colors.keys())
    category_id_map = {class_name: idx + 1 for idx, class_name in enumerate(class_list)}
    for class_name, cat_id in category_id_map.items():
        coco_data["categories"].append(
            {"id": cat_id, "name": class_name, "supercategory": "none"}
        )

    # Add images and annotations
    annotation_id = 1

    for image_id, image_path in enumerate(image_files, 1):
        # Try to get actual image dimensions
        try:
            img = cv2.imread(image_path)
            if img is not None:
                img_height, img_width = img.shape[:2]
            else:
                img_height, img_width = image_height, image_width
        except:
            img_height, img_width = image_height, image_width

        # Add image info
        image_filename = os.path.basename(image_path)
        coco_data["images"].append(
            {
                "id": image_id,
                "license": 1,
                "file_name": image_filename,
                "height": img_height,
                "width": img_width,
                "date_captured": "",
                "frame_id": image_id - 1,  # Store frame number for compatibility
            }
        )

        # Add annotations for this image
        frame_num = image_id - 1
        if frame_num in frame_annotations:
            for annotation in frame_annotations[frame_num]:
                # Get category id
                category_id = category_id_map.get(annotation.class_name, 1)

                # Get bounding box in COCO format [x, y, width, height]
                rect = annotation.rect
                bbox = [rect.x(), rect.y(), rect.width(), rect.height()]

                # Calculate area
                area = rect.width() * rect.height()

                # Create annotation entry
                coco_annotation = {
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": category_id,
                    "bbox": bbox,
                    "area": area,
                    "segmentation": [],
                    "iscrowd": 0,
                }

                # Add attributes if available
                if hasattr(annotation, "attributes") and annotation.attributes:
                    for attr_name, attr_value in annotation.attributes.items():
                        if attr_value != -1:  # Only export non-default attributes
                            coco_annotation[attr_name.lower()] = attr_value

                coco_data["annotations"].append(coco_annotation)
                annotation_id += 1

    # Write to file
    save_json_atomically(filename,coco_data)


def export_image_dataset_yolo(output_dir, image_files, frame_annotations, class_colors):
    """
    Export annotations for an image dataset in YOLO format.

    Args:
        output_dir (str): Directory to save YOLO .txt files
        image_files (list): List of image file paths
        frame_annotations (dict): Dictionary mapping frame numbers to annotations
        class_colors (dict): Dictionary mapping class names to QColor (for class order)
    """
    import os
    import cv2

    # Create output directories
    labels_dir = os.path.join(output_dir, "labels")
    os.makedirs(labels_dir, exist_ok=True)

    # Get class list and mapping
    class_list = list(class_colors.keys())
    class_to_id = {cls: i for i, cls in enumerate(class_list)}

    # Write classes.txt
    classes_file = os.path.join(output_dir, "classes.txt")
    with open(classes_file, "w") as f:
        for cls in class_list:
            f.write(f"{cls}\n")

    # Process each image
    for frame_num, image_path in enumerate(image_files):
        # Skip if no annotations for this frame
        if frame_num not in frame_annotations or not frame_annotations[frame_num]:
            continue

        # Get output .txt filename (same basename as image)
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        txt_filename = os.path.join(labels_dir, f"{base_name}.txt")

        # Try to get actual image dimensions
        try:
            img = cv2.imread(image_path)
            if img is not None:
                image_height, image_width = img.shape[:2]
            else:
                image_width, image_height = 640, 480
        except Exception:
            image_width, image_height = 640, 480

        with open(txt_filename, "w") as f:
            for annotation in frame_annotations[frame_num]:
                class_id = class_to_id.get(annotation.class_name, 0)
                rect = annotation.rect

                x = rect.x()
                y = rect.y()
                w = rect.width()
                h = rect.height()

                # Ensure values are within image bounds
                if (
                    x < 0
                    or y < 0
                    or w <= 0
                    or h <= 0
                    or x + w > image_width
                    or y + h > image_height
                ):
                    continue

                x_center = (x + w / 2) / image_width
                y_center = (y + h / 2) / image_height
                norm_w = w / image_width
                norm_h = h / image_height

                # Ensure normalized values are between 0 and 1
                x_center = max(0, min(1, x_center))
                y_center = max(0, min(1, y_center))
                norm_w = max(0, min(1, norm_w))
                norm_h = max(0, min(1, norm_h))

                f.write(
                    f"{class_id} {x_center:.6f} {y_center:.6f} {norm_w:.6f} {norm_h:.6f}\n"
                )

def import_annotations(
    filename, bbox_class, image_width=640, image_height=480, class_colors=None,
    existing_annotations=None
):
    """
    Import annotations from various formats (YOLO, Pascal VOC, COCO, Raya, Raya YOLO).
    The format is automatically detected based on file extension and content.

    Args:
        filename (str): Path to the annotation file
        bbox_class (class): Class to use for bounding box objects
        image_width (int, optional): Width of the image/video frame
        image_height (int, optional): Height of the image/video frame
        class_colors (dict, optional): Dictionary mapping class names to colors
        existing_annotations (dict, optional): Dictionary mapping frame numbers to existing annotations

    Returns:
        tuple: (format_type, annotations, frame_annotations)
            - format_type (str): Detected format type
            - annotations (list): List of annotations for current frame (if applicable)
            - frame_annotations (dict): Dictionary mapping frame numbers to lists of annotations
    """
    if class_colors is None:
        class_colors = {}

    # Detect format based on file extension and content
    format_type = detect_annotation_format(filename)

    if not format_type:
        raise ValueError(
            "Could not automatically detect the annotation format. Please ensure the file is in YOLO, Pascal VOC, COCO, Raya, or Raya YOLO format."
        )

    # Initialize return values
    annotations = []
    frame_annotations = {}

    # Import annotations based on format
    if format_type == "COCO":
        coco_annotations = import_coco_annotations(filename, bbox_class)
        for ann in coco_annotations:
            # Set class to "Quad" if we want a single class
            ann.class_name = "Quad"

            frame_num = getattr(ann, "frame", 0)
            if frame_num not in frame_annotations:
                frame_annotations[frame_num] = []
            frame_annotations[frame_num].append(ann)
            if frame_num == 0:  # Assume current frame is 0 for simplicity
                annotations.append(ann)

    elif format_type == "YOLO":
        annotations = import_yolo_annotations(
            filename, image_width, image_height, bbox_class, class_colors
        )
        # Set all annotations to "Quad" class
        for ann in annotations:
            ann.class_name = "Quad"
        frame_annotations[0] = annotations  # Assume frame 0 for YOLO

    elif format_type == "Pascal VOC":
        annotations = import_pascal_voc_annotations(
            filename, image_width, image_height, bbox_class, class_colors
        )
        # Set all annotations to "Quad" class
        for ann in annotations:
            ann.class_name = "Quad"
        frame_annotations[0] = annotations  # Assume frame 0 for Pascal VOC

    elif format_type == "Raya":
        frame_annotations = import_raya_annotations(filename, bbox_class, class_colors)
        # Raya format already uses "Quad" class by default
        if 0 in frame_annotations:
            annotations = frame_annotations[0]

    elif format_type == "RayaYOLO":
        frame_annotations = import_raya_yolo_annotations(
            filename, image_width, image_height, bbox_class, class_colors
        )
        # Set all annotations to "Quad" class
        for frame_num, frame_anns in frame_annotations.items():
            for ann in frame_anns:
                ann.class_name = "Quad"
        if 0 in frame_annotations:
            annotations = frame_annotations[0]

    # Ensure "Quad" class has a color
    if "Quad" not in class_colors:
        class_colors["Quad"] = QColor(255, 0, 0)  # Red color for Quad class
    
    _scale_annotation_scores([list(frame_annotations.values())])
   
    # Filter out overlapping annotations among the new ones
    frame_annotations = _filter_overlapping_annotations(frame_annotations)
    
    # Filter out annotations that significantly overlap with existing annotations
    if existing_annotations:
        frame_annotations = _filter_against_existing_annotations(frame_annotations, existing_annotations)
        
    # Update annotations list for current frame (frame 0)
    if 0 in frame_annotations:
        annotations = frame_annotations[0]

    return format_type, annotations, frame_annotations

def _filter_against_existing_annotations(new_annotations, existing_annotations, iou_threshold=0.8):
    """
    Filter out new annotations that have significant overlap (IoU > threshold) with existing annotations.
    
    Args:
        new_annotations (dict): Dictionary mapping frame numbers to lists of new annotations
        existing_annotations (dict): Dictionary mapping frame numbers to lists of existing annotations
        iou_threshold (float): IoU threshold above which annotations are considered duplicates (default: 0.8)
        
    Returns:
        dict: Filtered frame annotations
    """
    filtered_annotations = {}
    
    for frame_num, new_anns in new_annotations.items():
        # Get existing annotations for this frame if available
        existing_anns = existing_annotations.get(frame_num, [])
        
        # Filter out annotations with high IoU with existing ones
        kept_annotations = []
        for new_ann in new_anns:
            # Check if this annotation overlaps with any existing annotation
            is_duplicate = False
            for existing_ann in existing_anns:
                iou = _calculate_iou(new_ann.rect, existing_ann.rect)
                if iou > iou_threshold:
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                kept_annotations.append(new_ann)
        
        if kept_annotations:
            filtered_annotations[frame_num] = kept_annotations
        elif frame_num in new_annotations:
            # Keep the frame key in the dictionary even if all annotations were filtered out
            filtered_annotations[frame_num] = []
    
    return filtered_annotations

def _calculate_iou(box1, box2):
    """
    Calculate IoU (Intersection over Union) between two bounding boxes.
    
    Args:
        box1: First bounding box (QRect)
        box2: Second bounding box (QRect)
        
    Returns:
        float: IoU value between 0 and 1
    """
    # Calculate intersection area
    x1 = max(box1.x(), box2.x())
    y1 = max(box1.y(), box2.y())
    x2 = min(box1.x() + box1.width(), box2.x() + box2.width())
    y2 = min(box1.y() + box1.height(), box2.y() + box2.height())
    
    if x2 < x1 or y2 < y1:
        return 0.0  # No intersection
    
    intersection_area = (x2 - x1) * (y2 - y1)
    
    # Calculate union area
    box1_area = box1.width() * box1.height()
    box2_area = box2.width() * box2.height()
    union_area = box1_area + box2_area - intersection_area
    
    # Calculate IoU
    iou = intersection_area / union_area if union_area > 0 else 0.0
    return iou

def _filter_overlapping_annotations(frame_annotations, iou_threshold=0.5):
    """
    Filter out overlapping annotations based on IoU threshold.
    Keep only the annotation with the higher score when IoU exceeds threshold.
    
    Args:
        frame_annotations (dict): Dictionary mapping frame numbers to lists of annotations
        iou_threshold (float): IoU threshold for considering annotations as overlapping
        
    Returns:
        dict: Filtered frame annotations
    """
    filtered_annotations = {}
    
    for frame_num, annotations in frame_annotations.items():
        # Sort annotations by score (higher scores first)
        sorted_annotations = sorted(
            annotations, 
            key=lambda ann: getattr(ann, 'score', 0) if hasattr(ann, 'score') and ann.score is not None else 0,
            reverse=True
        )
        
        # Filter out overlapping annotations
        kept_annotations = []
        for ann in sorted_annotations:
            # Check if this annotation overlaps with any kept annotation
            should_keep = True
            for kept_ann in kept_annotations:
                iou = _calculate_iou(ann.rect, kept_ann.rect)
                if iou > iou_threshold:
                    should_keep = False
                    break
            
            if should_keep:
                kept_annotations.append(ann)
        
        filtered_annotations[frame_num] = kept_annotations
    
    return filtered_annotations

def _scale_annotation_scores(frame_annotations_list):
    """
    Scale the scores of annotations to a range of 0.2 to 1.0.
    
    Args:
        frame_annotations_list (list): List of frame annotation lists
    """
    # First, collect all valid scores from all annotations
    all_annotations = []
    valid_scores = []
    
    # Flatten the list of frame annotations and collect scores
    for frame_anns in frame_annotations_list:
        if isinstance(frame_anns, list):
            # If it's already a list of annotations
            for ann in frame_anns:
                all_annotations.append(ann)
                if hasattr(ann, 'score') and ann.score is not None:
                    valid_scores.append(ann.score)
        elif isinstance(frame_anns, dict):
            # If it's a dictionary of frame numbers to annotation lists
            for frame_num, anns in frame_anns.items():
                for ann in anns:
                    all_annotations.append(ann)
                    if hasattr(ann, 'score') and ann.score is not None:
                        valid_scores.append(ann.score)
    
    # If no valid scores, return
    if not valid_scores:
        return
    
    # Find min and max scores
    min_score = min(valid_scores)
    max_score = max(valid_scores)
    
    # If all scores are the same, set them all to 1.0
    if min_score == max_score:
        for ann in all_annotations:
            if hasattr(ann, 'score') and ann.score is not None:
                ann.score = 1.0
        return
    
    # Scale scores to range 0.2 to 1.0
    for ann in all_annotations:
        if hasattr(ann, 'score') and ann.score is not None:
            # Linear scaling formula: new_value = new_min + (value - old_min) * (new_max - new_min) / (old_max - old_min)
            ann.score = 0.2 + (ann.score - min_score) * 0.8 / (max_score - min_score)

def export_image_dataset_pascal_voc(
    output_dir, image_files, frame_annotations, pixmap=None
):
    """
    Export annotations for an image dataset in Pascal VOC XML format.

    Args:
        output_dir (str): Directory to save XML files
        image_files (list): List of image file paths
        frame_annotations (dict): Dictionary mapping frame numbers to annotations
        pixmap (QPixmap, optional): Pixmap for getting image dimensions if not available
    """
    import os
    import cv2
    from datetime import datetime

    # Create output directories
    annotations_dir = os.path.join(output_dir, "annotations")
    os.makedirs(annotations_dir, exist_ok=True)

    # Process each image
    for frame_num, image_path in enumerate(image_files):
        # Skip if no annotations for this frame
        if frame_num not in frame_annotations or not frame_annotations[frame_num]:
            continue

        # Get output XML filename (same basename as image)
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        xml_filename = os.path.join(annotations_dir, f"{base_name}.xml")

        # Try to get actual image dimensions
        try:
            img = cv2.imread(image_path)
            if img is not None:
                image_height, image_width = img.shape[:2]
            elif pixmap:
                image_width = pixmap.width()
                image_height = pixmap.height()
            else:
                image_width, image_height = 640, 480
        except Exception:
            if pixmap:
                image_width = pixmap.width()
                image_height = pixmap.height()
            else:
                image_width, image_height = 640, 480

        # # Create XML content
        # xml_content = create_pascal_voc_xml(
        #     os.path.basename(image_path),
        #     image_width,
        #     image_height,
        #     frame_annotations[frame_num]
        # )

        # # Write to file
        # with open(xml_filename, "w") as f:
        #     f.write(xml_content)
