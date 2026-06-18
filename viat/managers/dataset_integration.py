import os
import hashlib
import cv2
import json
import shutil
from utils.dataset_manager import scan_dataset
from utils.dataset_merger import merge_dataset_into_target

class DatasetIntegrationManager:
    """Handles the business logic for the Dataset Integration Roadmap steps."""

    def __init__(self, app):
        self.app = app

    def run_preflight_check(self, dataset_folder):
        """Scans the folder for 0-byte or corrupted image files and removes them."""
        # Simple scan
        valid_exts = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
        removed_dir = os.path.join(dataset_folder, "removed", "corrupted")
        
        for root, dirs, files in os.walk(dataset_folder):
            if "removed" in root:
                continue
            for file in files:
                ext = os.path.splitext(file)[1].lower()
                if ext in valid_exts:
                    path = os.path.join(root, file)
                    try:
                        # Check size
                        if os.path.getsize(path) == 0:
                            self._move_file(path, removed_dir)
                            continue
                        # Check if cv2 can open it
                        img = cv2.imread(path)
                        if img is None:
                            self._move_file(path, removed_dir)
                    except Exception:
                        self._move_file(path, removed_dir)

    def standardize_format(self, dataset_folder, target_ext=".jpg"):
        """Converts all images to the specified format, updating their names."""
        valid_exts = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
        target_ext = target_ext.lower()
        
        for root, dirs, files in os.walk(dataset_folder):
            if "removed" in root or "review" in root:
                continue
            for file in files:
                ext = os.path.splitext(file)[1].lower()
                if ext in valid_exts and ext != target_ext:
                    path = os.path.join(root, file)
                    img = cv2.imread(path)
                    if img is not None:
                        new_name = os.path.splitext(file)[0] + target_ext
                        new_path = os.path.join(root, new_name)
                        cv2.imwrite(new_path, img)
                        os.remove(path)
                        
                        # also rename corresponding label file if exists
                        old_label = os.path.join(root, os.path.splitext(file)[0] + ".txt")
                        new_label = os.path.join(root, os.path.splitext(new_name)[0] + ".txt")
                        if os.path.exists(old_label) and old_label != new_label:
                            shutil.move(old_label, new_label)

    def normalize_resolution(self, dataset_folder, max_size=(1920, 1080)):
        """Resizes images exceeding max_size down to fit within max_size while maintaining aspect ratio."""
        valid_exts = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
        max_w, max_h = max_size
        
        for root, dirs, files in os.walk(dataset_folder):
            if "removed" in root or "review" in root:
                continue
            for file in files:
                ext = os.path.splitext(file)[1].lower()
                if ext in valid_exts:
                    path = os.path.join(root, file)
                    img = cv2.imread(path)
                    if img is not None:
                        h, w = img.shape[:2]
                        if w > max_w or h > max_h:
                            scale = min(max_w / w, max_h / h)
                            new_w, new_h = int(w * scale), int(h * scale)
                            resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
                            cv2.imwrite(path, resized)

    def convert_to_grayscale(self, dataset_folder):
        """Converts all images in the dataset to grayscale (overwrites)."""
        valid_exts = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
        for root, dirs, files in os.walk(dataset_folder):
            if "removed" in root or "review" in root:
                continue
            for file in files:
                ext = os.path.splitext(file)[1].lower()
                if ext in valid_exts:
                    path = os.path.join(root, file)
                    img = cv2.imread(path)
                    if img is not None:
                        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                        cv2.imwrite(path, gray)

    def remove_duplicates(self, dataset_folder):
        """Removes exact duplicates using file hashing."""
        valid_exts = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
        seen_hashes = set()
        removed_dir = os.path.join(dataset_folder, "removed", "duplicates")
        
        for root, dirs, files in os.walk(dataset_folder):
            if "removed" in root or "review" in root:
                continue
            for file in files:
                ext = os.path.splitext(file)[1].lower()
                if ext in valid_exts:
                    path = os.path.join(root, file)
                    file_hash = self._get_file_hash(path)
                    if file_hash in seen_hashes:
                        self._move_file(path, removed_dir)
                    else:
                        seen_hashes.add(file_hash)

    def apply_auto_import(self, source_folder, target_main_folder, json_path):
        """
        Reads detections JSON. If detections contain classes NOT present in
        the main dataset's classes, those frames are moved to a 'Review' folder
        in the target main dataset.
        """
        target_info = scan_dataset(target_main_folder)
        target_classes = target_info.classes
        
        if not target_classes:
            return  # No target classes defined yet, can't check
            
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        metadata = data.get("_metadata", {})
        frames = data.get("frames", data)
        image_files_list = metadata.get("image_files", [])
        
        review_dir = os.path.join(target_main_folder, "Review")
        
        # Scan source for mapping filename to path
        source_paths = {}
        valid_exts = {'.jpg', '.jpeg', '.png'}
        for root, dirs, files in os.walk(source_folder):
            for file in files:
                if os.path.splitext(file)[1].lower() in valid_exts:
                    source_paths[os.path.basename(file)] = os.path.join(root, file)

        for frame_key, frame_data in frames.items():
            try:
                frame_idx = int(frame_key)
                if frame_idx < len(image_files_list):
                    img_name = os.path.basename(image_files_list[frame_idx])
                else:
                    continue
                    
                actors = frame_data.get("actors", {}) if isinstance(frame_data, dict) else {}
                unmapped_found = False
                for actor in actors.values():
                    cls = actor.get("class")
                    if cls and cls not in target_classes:
                        unmapped_found = True
                        break
                        
                if unmapped_found and img_name in source_paths:
                    # Move this image and its label to Review
                    img_path = source_paths[img_name]
                    self._move_file_and_label(img_path, review_dir)
                    
            except (ValueError, TypeError):
                continue
                
    def merge_dataset(self, source_folder, target_folder, auto_rename=True):
        """Merges using dataset_merger.py logic."""
        # If auto_rename is True, we pass a prefix (dataset_name) 
        # based on the source folder.
        prefix = os.path.basename(source_folder) if auto_rename else "img"
        
        result = merge_dataset_into_target(
            self.app, 
            source_folder, 
            target_folder,
            dataset_name=prefix,
            split_mode="keep"
        )
        return result

    def _get_file_hash(self, filepath):
        hasher = hashlib.md5()
        with open(filepath, 'rb') as f:
            buf = f.read()
            hasher.update(buf)
        return hasher.hexdigest()

    def _move_file(self, filepath, dest_dir):
        os.makedirs(dest_dir, exist_ok=True)
        dest_path = os.path.join(dest_dir, os.path.basename(filepath))
        try:
            shutil.move(filepath, dest_path)
        except Exception:
            pass

    def _move_file_and_label(self, img_path, dest_dir):
        """Moves an image and its corresponding label files to dest_dir."""
        os.makedirs(dest_dir, exist_ok=True)
        img_dest = os.path.join(dest_dir, "images")
        lbl_dest = os.path.join(dest_dir, "labels")
        os.makedirs(img_dest, exist_ok=True)
        os.makedirs(lbl_dest, exist_ok=True)
        
        # Move image
        try:
            shutil.move(img_path, os.path.join(img_dest, os.path.basename(img_path)))
        except Exception:
            pass
            
        # Try to find and move label (assuming YOLO txt in same dir or adjacent labels dir)
        base = os.path.splitext(os.path.basename(img_path))[0]
        # Look in same dir
        txt_path = os.path.join(os.path.dirname(img_path), f"{base}.txt")
        if os.path.exists(txt_path):
            shutil.move(txt_path, os.path.join(lbl_dest, f"{base}.txt"))
        else:
            # Look in ../labels
            lbl_dir = os.path.join(os.path.dirname(os.path.dirname(img_path)), "labels")
            txt_path = os.path.join(lbl_dir, f"{base}.txt")
            if os.path.exists(txt_path):
                shutil.move(txt_path, os.path.join(lbl_dest, f"{base}.txt"))
