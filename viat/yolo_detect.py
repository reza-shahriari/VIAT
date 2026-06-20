#!/usr/bin/env python3
"""
YOLO inference script for VIAT dataset auto-labeling.

Runs a YOLO model on every image in a dataset folder, detects specified
classes (e.g. 'human'), and saves the results in a VIAT-compatible JSON
file. VIAT can then auto-import this JSON and move flagged images to
review_label/ for manual review.

WHY STANDALONE (not inside VIAT):
  YOLO + PyTorch is a huge dependency. Keeping it out of VIAT keeps the
  annotation tool lightweight. This script is run separately (e.g. on a
  GPU machine or overnight), producing a small JSON that VIAT consumes.

USAGE:
  python yolo_detect.py \\
      --dataset /path/to/dataset \\
      --model yolov8n.pt \\
      --detect-classes person \\
      --confidence 0.5 \\
      --output /path/to/dataset/_viat_detections.json

  # Then in VIAT: Dataset > Auto-Import Detections (move to review_label)

REQUIREMENTS:
  pip install ultralytics  # YOLOv8 (also works with yolov5)

OUTPUT FORMAT (VIAT-compatible JSON):
  {
    "0000": {
      "actors": {
        "yolo_person_0": {
          "class": "person",
          "accepted": false,
          "bbox": [100, 200, 50, 80],
          "score": 0.92,
          "source": "yolo"
        }
      }
    }
  }

  Frames with at least one detection are flagged. When VIAT imports this:
    - Images WITH detections -> moved to review_label/ (need human review)
    - Images WITHOUT detections -> stay in place (model says "nothing to add")

  The 'accepted' field is false for all YOLO detections (they're unverified
  until a human confirms them).
"""

import argparse
import json
import os
import sys
from pathlib import Path

# Image extensions VIAT recognizes
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp")


def run_yolo_detection(
    dataset_path: str,
    model_path: str,
    detect_classes: list,
    confidence: float = 0.5,
    output_path: str = None,
    recursive: bool = True,
    model_instance = None,
):
    """Run YOLO detection on all images in a dataset folder.

    Args:
        dataset_path: root folder of the dataset (images can be in subfolders
            like images/, train/, valid/, etc.)
        model_path: path to YOLOv8 .pt model (or yolov5 .pt)
        detect_classes: list of COCO class names to detect (e.g. ['person']).
            Only detections of these classes are saved.
        confidence: minimum confidence threshold (0-1).
        output_path: where to save the JSON. Default: <dataset>/_viat_detections.json
        recursive: search subfolders for images.

    Returns:
        dict with stats: {images_processed, images_with_detections,
                          total_detections, output_path}
    """
    try:
        from ultralytics import YOLO
    except ImportError:
        print("ERROR: ultralytics not installed. Run: pip install ultralytics")
        sys.exit(1)

    # Load model
    if model_instance is not None:
        model = model_instance
    else:
        print(f"Loading YOLO model: {model_path}")
        model = YOLO(model_path)

    # COCO class names (YOLOv8 default). The model's names dict maps index->name.
    model_names = model.names if hasattr(model, "names") else {}
    # Build name->index map
    name_to_idx = {v.lower(): k for k, v in model_names.items()}

    # Resolve target class indices
    target_indices = set()
    for cls in detect_classes:
        cls_lower = cls.lower()
        if cls_lower in name_to_idx:
            target_indices.add(name_to_idx[cls_lower])
        else:
            print(f"WARNING: class '{cls}' not found in model. Available: {list(model_names.values())}")

    if not target_indices:
        print("ERROR: none of the requested classes found in the model.")
        sys.exit(1)

    print(f"Detecting classes: {detect_classes} (indices: {target_indices})")
    print(f"Confidence threshold: {confidence}")

    # Find all images
    images = _find_images(dataset_path, recursive)
    print(f"Found {len(images)} images")

    if not images:
        print("No images found.")
        return {"images_processed": 0, "images_with_detections": 0, "total_detections": 0}

    # Run inference
    results_json = {}
    images_with_detections = 0
    total_detections = 0

    for i, img_path in enumerate(images):
        results = model(img_path, verbose=False)

        frame_actors = {}
        det_count = 0

        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue

            for box in boxes:
                cls_idx = int(box.cls[0])
                if cls_idx not in target_indices:
                    continue
                conf = float(box.conf[0])
                if conf < confidence:
                    continue

                # bbox: xyxy -> xywh
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                x, y = int(x1), int(y1)
                w, h = int(x2 - x1), int(y2 - y1)

                cls_name = model_names.get(cls_idx, f"class_{cls_idx}")
                actor_id = f"yolo_{cls_name}_{det_count}"

                frame_actors[actor_id] = {
                    "class": cls_name,
                    "accepted": False,  # unverified -- needs human review
                    "bbox": [x, y, w, h],
                    "score": round(conf, 4),
                    "source": "yolo",
                }
                det_count += 1
                total_detections += 1

        if frame_actors:
            frame_key = str(i).zfill(4)
            results_json[frame_key] = {"actors": frame_actors}
            images_with_detections += 1

        if (i + 1) % 100 == 0:
            print(f"  Processed {i+1}/{len(images)} images, {images_with_detections} with detections")

    # Save JSON
    if output_path is None:
        output_path = os.path.join(dataset_path, "_viat_detections.json")

    # Add metadata at the top (VIAT will ignore unknown top-level keys)
    output = {
        "_metadata": {
            "model": model_path,
            "detect_classes": detect_classes,
            "confidence": confidence,
            "total_images": len(images),
            "images_with_detections": images_with_detections,
            "total_detections": total_detections,
            "image_files": images,  # so VIAT can map frame keys -> file paths
        },
        "frames": results_json,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    print(f"\nDone!")
    print(f"  Images processed: {len(images)}")
    print(f"  Images with detections: {images_with_detections}")
    print(f"  Total detections: {total_detections}")
    print(f"  Output: {output_path}")
    print(f"\nIn VIAT: Dataset > Auto-Import Detections > {output_path}")

    return {
        "images_processed": len(images),
        "images_with_detections": images_with_detections,
        "total_detections": total_detections,
        "output_path": output_path,
    }


def _find_images(root: str, recursive: bool = True) -> list:
    """Find all image files in root, sorted."""
    images = []
    if recursive:
        for dirpath, dirnames, filenames in os.walk(root):
            # skip the review_label/ and removed/ folders
            dirnames[:] = [d for d in dirnames if d not in ("review_label", "removed", "discarded")]
            for f in filenames:
                if f.lower().endswith(IMAGE_EXTENSIONS):
                    images.append(os.path.join(dirpath, f))
    else:
        for f in os.listdir(root):
            if f.lower().endswith(IMAGE_EXTENSIONS):
                images.append(os.path.join(root, f))
    images.sort()
    return images


def get_label_path(image_path: str) -> str:
    """Find the corresponding YOLO label file for an image, if it exists."""
    p = Path(image_path)
    # Check 1: same directory, change suffix to .txt
    same_dir = p.with_suffix('.txt')
    if same_dir.exists():
        return str(same_dir)

    # Check 2: swap 'images' or 'image' directory with 'labels' or 'label'
    parts = list(p.parts)
    for i in range(len(parts) - 1, -1, -1):
        part_lower = parts[i].lower()
        if part_lower == "images":
            parts[i] = "labels"
            new_p = Path(*parts).with_suffix('.txt')
            if new_p.exists():
                return str(new_p)
        elif part_lower == "image":
            parts[i] = "label"
            new_p = Path(*parts).with_suffix('.txt')
            if new_p.exists():
                return str(new_p)
    return None


def parse_dataset_yaml(dataset_path: str) -> dict:
    """Find and parse names of classes from dataset yaml files."""
    p = Path(dataset_path)
    yaml_files = list(p.glob("*.yaml")) + list(p.glob("*.yml"))
    names_dict = {}
    for yf in yaml_files:
        try:
            with open(yf, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Try parsing with PyYAML if available
            try:
                import yaml
                data = yaml.safe_load(content)
                if isinstance(data, dict) and 'names' in data:
                    names = data['names']
                    if isinstance(names, dict):
                        return {int(k): str(v) for k, v in names.items()}
                    elif isinstance(names, list):
                        return {i: str(v) for i, v in enumerate(names)}
            except ImportError:
                # Fallback manual YAML parser for simple list/dictionary name formats
                import re
                names_match = re.search(r'names:\s*(.*)', content)
                if names_match:
                    inline_list = names_match.group(1).strip()
                    if inline_list.startswith('[') and inline_list.endswith(']'):
                        names_list = re.findall(r"['\"]([^'\"]+)['\"]", inline_list)
                        if names_list:
                            return {i: name for i, name in enumerate(names_list)}
                    
                    rest = content[names_match.end():]
                    lines = rest.split('\n')
                    curr_dict = {}
                    for line in lines:
                        line_strip = line.strip()
                        if not line_strip:
                            continue
                        dict_match = re.match(r'^(\d+)\s*:\s*[\'"]?([^\'"#]+)[\'"]?', line_strip)
                        if dict_match:
                            curr_dict[int(dict_match.group(1))] = dict_match.group(2).strip()
                        elif line_strip.startswith('-'):
                            item = line_strip.lstrip('- ').strip().strip("'\"")
                            curr_dict[len(curr_dict)] = item
                        elif ':' in line_strip and not re.match(r'^\s*\d+\s*:', line_strip):
                            break
                    if curr_dict:
                        return curr_dict
        except Exception as e:
            print(f"Warning parsing YAML {yf}: {e}")
    return names_dict


def letterbox_image(image, target_size=(400, 400)):
    """Resize image to fit target size while maintaining aspect ratio, adding dark grey margins."""
    import cv2
    import numpy as np
    
    ih, iw = image.shape[:2]
    th, tw = target_size
    scale = min(tw / iw, th / ih)
    nw = int(iw * scale)
    nh = int(ih * scale)
    resized = cv2.resize(image, (nw, nh))
    
    canvas = np.zeros((th, tw, 3), dtype=np.uint8) + 40  # dark grey padding
    dx = (tw - nw) // 2
    dy = (th - nh) // 2
    canvas[dy:dy+nh, dx:dx+nw] = resized
    return canvas


def create_image_grid(images, cols=3):
    """Combine a list of images into a single grid image."""
    import numpy as np
    
    if not images:
        return None
        
    cols = min(len(images), cols) if len(images) > 0 else 1
    # Pad images list with blanks if necessary
    while len(images) % cols != 0:
        blank = np.zeros_like(images[0]) + 20  # very dark grey padding
        images.append(blank)
        
    rows = len(images) // cols
    row_images = []
    for r in range(rows):
        row_img = np.hstack(images[r*cols : (r+1)*cols])
        row_images.append(row_img)
    grid = np.vstack(row_images)
    return grid


def curate_and_select_classes(datasets_root: str, model_path: str, confidence: float = 0.5):
    """Interactive CLI and OpenCV curation workflow to select model classes for each dataset."""
    try:
        from ultralytics import YOLO
    except ImportError:
        print("ERROR: ultralytics not installed. Run: pip install ultralytics")
        sys.exit(1)
        
    root_path = Path(datasets_root)
    if not root_path.exists() or not root_path.is_dir():
        print(f"ERROR: Datasets root path '{datasets_root}' does not exist or is not a directory.")
        sys.exit(1)
        
    # Scan subdirectories for images
    print(f"Scanning '{datasets_root}' for sub-datasets...")
    datasets = []
    for entry in os.scandir(datasets_root):
        if entry.is_dir():
            images = _find_images(entry.path, recursive=True)
            if images:
                datasets.append((entry.name, entry.path, images))
    datasets.sort(key=lambda x: x[0])
    
    if not datasets:
        print(f"No subdirectories with images found in '{datasets_root}'.")
        sys.exit(1)
        
    print(f"Found {len(datasets)} sub-dataset(s) to curate.")
    
    # Load model
    print(f"Loading YOLO model to inspect classes: {model_path}")
    model = YOLO(model_path)
    model_names = model.names if hasattr(model, "names") else {}
    name_to_idx = {v.lower(): k for k, v in model_names.items()}
    
    # Print model classes compactly
    print("\nAvailable model classes for prediction:")
    class_items = [f"{k}: {v}" for k, v in model_names.items()]
    for i in range(0, len(class_items), 5):
        print("  " + ", ".join(class_items[i:i+5]))
        
    dataset_selections = {}
    last_selection = None
    apply_to_all_remaining = False
    
    for idx, (name, path, images) in enumerate(datasets):
        print(f"\n[{idx+1}/{len(datasets)}] Dataset: {name}")
        print(f"  Path: {path}")
        print(f"  Images: {len(images)}")
        
        if apply_to_all_remaining:
            if last_selection:
                dataset_selections[path] = last_selection
                print(f"  Automatically selected (same as previous): {last_selection}")
            continue
            
        # Parse YAML names
        yaml_names = parse_dataset_yaml(path)
        
        # Scan label files for class samples
        class_samples = {}
        for img_path in images:
            lbl_path = get_label_path(img_path)
            if not lbl_path:
                continue
            try:
                with open(lbl_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) >= 5:
                            class_id = int(parts[0])
                            bbox = [float(x) for x in parts[1:5]]
                            if class_id not in class_samples:
                                class_samples[class_id] = []
                            class_samples[class_id].append((img_path, bbox))
            except Exception:
                pass
                
        # Generate grid image
        grid_images = []
        try:
            import cv2
            import numpy as np
            
            for class_id in sorted(class_samples.keys()):
                # Pick sample with largest bounding box for clarity
                samples = class_samples[class_id]
                best_sample = None
                max_area = -1
                for img_p, bbox in samples[:10]:
                    w, h = bbox[2], bbox[3]
                    area = w * h
                    if area > max_area:
                        max_area = area
                        best_sample = (img_p, bbox)
                if not best_sample:
                    best_sample = samples[0]
                    
                img = cv2.imread(best_sample[0])
                if img is None:
                    continue
                    
                h_img, w_img = img.shape[:2]
                x_c, y_c, w_b, h_b = best_sample[1]
                xmin = int((x_c - w_b/2) * w_img)
                ymin = int((y_c - h_b/2) * h_img)
                xmax = int((x_c + w_b/2) * w_img)
                ymax = int((y_c + h_b/2) * h_img)
                
                xmin, ymin = max(0, xmin), max(0, ymin)
                xmax, ymax = min(w_img, xmax), min(h_img, ymax)
                
                cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 255, 0), 3)
                
                cls_name = yaml_names.get(class_id, f"Class {class_id}")
                label_text = f"{class_id}: {cls_name}"
                (text_w, text_h), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                text_y = max(ymin, text_h + 15)
                cv2.rectangle(img, (xmin, text_y - text_h - 15), (xmin + text_w + 10, text_y), (0, 255, 0), cv2.FILLED)
                cv2.putText(img, label_text, (xmin + 5, text_y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
                
                grid_images.append(letterbox_image(img, (400, 400)))
        except ImportError:
            pass
            
        grid = None
        if grid_images:
            grid = create_image_grid(grid_images, cols=3)
            
        # Display window if OpenCV is loaded and grid is generated
        window_name = f"Dataset: {name}"
        has_display = False
        if grid is not None:
            try:
                cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
                cv2.imshow(window_name, grid)
                cv2.waitKey(200)  # Render window
                has_display = True
            except Exception as e:
                print(f"  Warning: Display window could not be initialized ({e}).")
                
        # Print info in CLI
        if class_samples:
            print("  Classes found in dataset labels:")
            for cid in sorted(class_samples.keys()):
                lbl = yaml_names.get(cid, f"Class {cid}")
                print(f"    - ID {cid}: {lbl} ({len(class_samples[cid])} annotations)")
        else:
            print("  No annotations found in label files.")
            
        # Prompt user
        while True:
            prompt_str = (
                "  Select classes to predict for this dataset.\n"
                "  Options:\n"
                "    - Comma-separated names/indices (e.g. 'car, truck' or '2, 7')\n"
                "    - 'all' for all model classes\n"
                "    - 'skip' (or press Enter) to skip this dataset\n"
            )
            if last_selection:
                prompt_str += f"    - 'same' to use last selection: {last_selection}\n"
            prompt_str += "  Selection: "
            
            user_input = input(prompt_str).strip().lower()
            
            if not user_input or user_input == 'skip':
                selected = []
                break
            if user_input == 'same' and last_selection:
                selected = last_selection
                break
            if user_input == 'all':
                selected = list(model_names.values())
                break
                
            # Parse custom list
            selected = []
            valid = True
            for part in user_input.split(','):
                part = part.strip()
                if not part:
                    continue
                if part.isdigit():
                    idx = int(part)
                    if idx in model_names:
                        selected.append(model_names[idx])
                    else:
                        print(f"    ERROR: index {idx} not in model classes.")
                        valid = False
                else:
                    if part in name_to_idx:
                        selected.append(model_names[name_to_idx[part]])
                    else:
                        # Substring search
                        matches = [v for v in model_names.values() if part in v.lower()]
                        if len(matches) == 1:
                            selected.append(matches[0])
                        elif len(matches) > 1:
                            print(f"    Ambiguous name '{part}'. Matches: {matches}. Using the first one: {matches[0]}")
                            selected.append(matches[0])
                        else:
                            print(f"    ERROR: class name '{part}' not found in model.")
                            valid = False
            if valid:
                break
                
        if has_display:
            try:
                cv2.destroyWindow(window_name)
                # Extra waitKey to allow window to close completely
                cv2.waitKey(1)
            except Exception:
                pass
                
        if selected:
            print(f"  Selected for prediction: {selected}")
            dataset_selections[path] = selected
            last_selection = selected
            
            if not apply_to_all_remaining and idx < len(datasets) - 1:
                ans = input("  Apply this selection to all remaining datasets? (y/n) [n]: ").strip().lower()
                if ans in ('y', 'yes'):
                    apply_to_all_remaining = True
        else:
            print("  Skipping prediction for this dataset.")
            
    if not dataset_selections:
        print("\nNo datasets selected for predictions. Exiting.")
        return
        
    print("\n" + "=" * 50)
    print("YOLO PREDICTION PLAN SUMMARY:")
    print("=" * 50)
    for path, target_classes in dataset_selections.items():
        print(f"  - {os.path.basename(path)}: {target_classes}")
    print("=" * 50)
    
    confirm = input("Proceed with running YOLO predictions on these datasets? (y/n) [y]: ").strip().lower()
    if confirm not in ('', 'y', 'yes'):
        print("Cancelled.")
        return
        
    # Execute batch predictions
    model_stem = Path(model_path).stem
    print(f"\nStarting batch prediction using preloaded model...")
    for path, target_classes in dataset_selections.items():
        print(f"\nProcessing: {path}")
        output_json = os.path.join(path, f"{model_stem}_detections.json")
        run_yolo_detection(
            dataset_path=path,
            model_path=model_path,
            detect_classes=target_classes,
            confidence=confidence,
            output_path=output_json,
            recursive=True,
            model_instance=model,
        )


def main():
    parser = argparse.ArgumentParser(
        description="Run YOLO detection on a dataset, output VIAT-compatible JSON."
    )
    parser.add_argument(
        "--dataset", default=None,
        help="Path to a single dataset folder"
    )
    parser.add_argument(
        "--datasets-root", default=None,
        help="Path to a directory containing multiple sub-dataset folders"
    )
    parser.add_argument(
        "--dataset-queue", default=None,
        help="Path to a .txt file containing a list of dataset folders (generated by VIAT Batch Prediction Queue Builder)"
    )
    parser.add_argument(
        "--model", required=True,
        help="Path to YOLOv8 .pt model (e.g. yolov8n.pt)"
    )
    parser.add_argument(
        "--detect-classes", nargs="+", default=["soldier","tank"],
        help="COCO class names to detect (e.g. person car truck). Default: person"
    )
    parser.add_argument(
        "--confidence", type=float, default=0.5,
        help="Minimum confidence threshold (0-1). Default: 0.5"
    )
    parser.add_argument(
        "--output", default=None,
        help="Output JSON path. Default: <dataset>/_viat_detections.json"
    )
    parser.add_argument(
        "--no-recursive", action="store_true",
        help="Don't search subfolders for images"
    )

    args = parser.parse_args()

    if args.datasets_root:
        curate_and_select_classes(
            datasets_root=args.datasets_root,
            model_path=args.model,
            confidence=args.confidence,
        )
    elif args.dataset:
        run_yolo_detection(
            dataset_path=args.dataset,
            model_path=args.model,
            detect_classes=args.detect_classes,
            confidence=args.confidence,
            output_path=args.output,
            recursive=not args.no_recursive,
        )
    elif args.dataset_queue:
        if not os.path.isfile(args.dataset_queue):
            print(f"Queue file not found: {args.dataset_queue}")
            sys.exit(1)
            
        with open(args.dataset_queue, "r", encoding="utf-8") as f:
            paths = [line.strip() for line in f if line.strip()]
            
        print(f"Found {len(paths)} datasets in queue.")
        
        try:
            from ultralytics import YOLO
            model = YOLO(args.model)
        except ImportError:
            print("ultralytics not installed.")
            sys.exit(1)
            
        for path in paths:
            if not os.path.isdir(path):
                print(f"Skipping invalid path: {path}")
                continue
            
            # If no output path specified, save it in the dataset folder
            output_json = args.output
            if not output_json:
                model_stem = Path(args.model).stem
                output_json = os.path.join(path, f"{model_stem}_detections.json")
                
            print(f"\nProcessing queued dataset: {path}")
            run_yolo_detection(
                dataset_path=path,
                model_path=args.model,
                detect_classes=args.detect_classes,
                confidence=args.confidence,
                output_path=output_json,
                recursive=not args.no_recursive,
                model_instance=model,
            )
    else:
        print("ERROR: You must specify --dataset, --datasets-root, or --dataset-queue.")
        sys.exit(1)


if __name__ == "__main__":
    main()
