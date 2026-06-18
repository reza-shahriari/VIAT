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


def main():
    parser = argparse.ArgumentParser(
        description="Run YOLO detection on a dataset, output VIAT-compatible JSON."
    )
    parser.add_argument(
        "--dataset", required=True,
        help="Path to the dataset folder (images can be in subfolders)"
    )
    parser.add_argument(
        "--model", required=True,
        help="Path to YOLOv8 .pt model (e.g. yolov8n.pt)"
    )
    parser.add_argument(
        "--detect-classes", nargs="+", default=["person"],
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

    run_yolo_detection(
        dataset_path=args.dataset,
        model_path=args.model,
        detect_classes=args.detect_classes,
        confidence=args.confidence,
        output_path=args.output,
        recursive=not args.no_recursive,
    )


if __name__ == "__main__":
    main()
