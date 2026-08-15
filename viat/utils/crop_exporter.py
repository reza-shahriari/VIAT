import os
import cv2
import json
import numpy as np
from PyQt5.QtCore import QRect, QPoint
from PyQt5.QtWidgets import QProgressDialog
from PyQt5.QtCore import Qt

class CropExporter:
    """Engine for exporting cropped datasets (images or video) and transforming annotations."""
    
    def __init__(self, main_window, video_path, frame_annotations, base_crop_rect, track_id=None, class_colors=None):
        self.main_window = main_window
        self.video_path = video_path
        self.frame_annotations = frame_annotations
        self.base_crop_rect = base_crop_rect
        self.track_id = track_id
        self.class_colors = class_colors or {}
        
    def export(self, output_dir, format_type="mp4"):
        """Run the export process."""
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            return False, "Could not open video file."
            
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        crop_w = self.base_crop_rect.width()
        crop_h = self.base_crop_rect.height()
        
        progress = QProgressDialog("Exporting Cropped Dataset...", "Cancel", 0, total_frames, self.main_window)
        progress.setWindowModality(Qt.WindowModal)
        
        # Prepare output directories
        os.makedirs(output_dir, exist_ok=True)
        labels_dir = os.path.join(output_dir, "labels")
        os.makedirs(labels_dir, exist_ok=True)
        
        # Build class mapping
        class_names = list(self.class_colors.keys())
        class_to_id = {name: i for i, name in enumerate(class_names)}
        
        with open(os.path.join(output_dir, "classes.txt"), "w") as f:
            for name in class_names:
                f.write(f"{name}\n")
                
        # Setup Video Writer if mp4
        writer = None
        images_dir = None
        if format_type == "mp4":
            mp4_path = os.path.join(output_dir, "cropped_video.mp4")
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(mp4_path, fourcc, fps, (crop_w, crop_h))
        else:
            images_dir = os.path.join(output_dir, "images")
            os.makedirs(images_dir, exist_ok=True)
            
        current_crop = QRect(self.base_crop_rect)
        
        for frame_idx in range(total_frames):
            if progress.wasCanceled():
                break
                
            ret, frame = cap.read()
            if not ret:
                break
                
            progress.setValue(frame_idx)
            
            anns = self.frame_annotations.get(frame_idx, [])
            
            # Update tracking crop box
            if self.track_id is not None:
                # Find the tracked object in this frame
                tracked_ann = None
                for ann in anns:
                    aid = (getattr(ann, 'attributes', None) or {}).get('actor_id') or (getattr(ann, 'attributes', None) or {}).get('track_id')
                    # If we don't have track_id assigned properly, we might need to rely on tracking metadata
                    # Fallback to selected_annotation logic if the track_id matches
                    if aid == self.track_id or ann.id == self.track_id:
                        tracked_ann = ann
                        break
                        
                if tracked_ann is not None:
                    # Move center
                    center = tracked_ann.rect.center()
                    current_crop.moveCenter(center)
                    
            # Enforce boundary checks (Shift, not pad!)
            if current_crop.left() < 0: current_crop.moveLeft(0)
            if current_crop.top() < 0: current_crop.moveTop(0)
            if current_crop.right() > orig_w: current_crop.moveRight(orig_w)
            if current_crop.bottom() > orig_h: current_crop.moveBottom(orig_h)
            
            # Extract crop
            x1 = max(0, current_crop.x())
            y1 = max(0, current_crop.y())
            x2 = min(orig_w, current_crop.right())
            y2 = min(orig_h, current_crop.bottom())
            
            # If for some reason the crop is out of bounds completely
            if x2 <= x1 or y2 <= y1:
                continue
                
            cropped_img = frame[y1:y2, x1:x2]
            
            # If shift made it perfectly match crop_w and crop_h, great.
            # Otherwise we might need to pad (only if the video is smaller than crop_w/crop_h)
            # but user requested shift. If video < crop size, we have to pad.
            if cropped_img.shape[1] != crop_w or cropped_img.shape[0] != crop_h:
                # Video is smaller than crop size! Pad it as a fallback.
                pad_bottom = max(0, crop_h - cropped_img.shape[0])
                pad_right = max(0, crop_w - cropped_img.shape[1])
                cropped_img = cv2.copyMakeBorder(cropped_img, 0, pad_bottom, 0, pad_right, cv2.BORDER_CONSTANT, value=(0,0,0))
                
            if writer:
                writer.write(cropped_img)
            elif images_dir:
                cv2.imwrite(os.path.join(images_dir, f"frame_{frame_idx:06d}.jpg"), cropped_img)
                
            # Transform Annotations
            yolo_lines = []
            for ann in anns:
                # Apply transformation: new_x = old_x - crop_x
                new_x = ann.rect.x() - current_crop.x()
                new_y = ann.rect.y() - current_crop.y()
                new_w = ann.rect.width()
                new_h = ann.rect.height()
                
                # Clip box to crop region
                left = max(0, new_x)
                top = max(0, new_y)
                right = min(crop_w, new_x + new_w)
                bottom = min(crop_h, new_y + new_h)
                
                if right <= left or bottom <= top:
                    continue # Out of bounds
                    
                # Convert to YOLO (normalized)
                cx = (left + right) / 2.0 / crop_w
                cy = (top + bottom) / 2.0 / crop_h
                w = (right - left) / float(crop_w)
                h = (bottom - top) / float(crop_h)
                
                c_id = class_to_id.get(ann.class_name, 0)
                
                if hasattr(ann, 'polygon') and ann.polygon and len(ann.polygon) > 0:
                    # Transform polygon
                    poly_str = f"{c_id}"
                    valid_poly = False
                    for pt in ann.polygon:
                        px = pt.x() - current_crop.x()
                        py = pt.y() - current_crop.y()
                        # Clip polygon points (simple clamp)
                        px = max(0, min(crop_w, px))
                        py = max(0, min(crop_h, py))
                        norm_x = px / crop_w
                        norm_y = py / crop_h
                        poly_str += f" {norm_x:.6f} {norm_y:.6f}"
                        if 0 < norm_x < 1 and 0 < norm_y < 1:
                            valid_poly = True
                    
                    if valid_poly:
                        yolo_lines.append(poly_str)
                    else:
                        yolo_lines.append(f"{c_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
                else:
                    yolo_lines.append(f"{c_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
                    
            if yolo_lines:
                label_path = os.path.join(labels_dir, f"frame_{frame_idx:06d}.txt")
                with open(label_path, "w") as f:
                    f.write("\n".join(yolo_lines))
                    
        if writer:
            writer.release()
        cap.release()
        progress.setValue(total_frames)
        
        return True, f"Successfully exported to {output_dir}"
