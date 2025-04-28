import os
import json
import cv2
import numpy as np
from PyQt5.QtCore import QRect
from PyQt5.QtGui import QColor
import datetime

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
    with open(filename, "w") as f:
        json.dump(project_data, f, indent=2)

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
                duplicate_frames_enabled, frame_hashes, duplicate_frames_cache, image_dataset_info)
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
            bbox_class.from_dict(ann_data) for ann_data in frame_anns]
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
    with open(recent_projects_file, "w") as f:
        json.dump(recent_projects, f)


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
    with open(state_file, "w") as f:
        json.dump(state_data, f, indent=2)


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
    with open(filename, "w") as f:
        json.dump(data, f, indent=2)


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
        with open(attributes_file, "w") as f:
            json.dump(attributes_data, f, indent=2)


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
