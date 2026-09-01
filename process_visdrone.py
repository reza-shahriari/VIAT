#!/usr/bin/env python3
"""
VisDrone Dataset Preprocessor for VIAT.

Filtering & Blurring Rules:
1. Blur all annotations with:
   - occlusion > 0
   - truncation > 0
   - width < 10 or height < 10 pixels (small boxes)
   - class 0 (ignored_region) or class 11 (others)
2. Cascading Blurring:
   - Any clean bounding box that intersects an already blurred region is also blurred,
     and its label is removed.
3. 20% Image Area Rejection:
   - If total blurred area > 20% of image area (W * H), the entire image and its labels are discarded.
4. Export:
   - Applies Gaussian blur to blurred regions in retained images.
   - Saves clean surviving annotations in both VisDrone format (.txt in annotations/) and YOLO format (.txt in labels/).
   - Generates data.yaml for instant model training and VIAT loading.
   - Generates a comprehensive summary report.
"""

import os
import sys
import glob
import json
import argparse
import numpy as np
import cv2
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, List, Tuple, Optional

VISDRONE_CLASSES = [
    "ignored_region",   # 0
    "pedestrian",       # 1
    "person",           # 2
    "bicycle",          # 3
    "car",              # 4
    "van",              # 5
    "truck",            # 6
    "tricycle",         # 7
    "awning-tricycle",  # 8
    "bus",              # 9
    "motor",            # 10
    "others",           # 11
]

# 10 Target classes for object detection (mapping 1-10 to 0-9 in YOLO)
YOLO_TARGET_CLASSES = [
    "pedestrian",       # 0 (VisDrone 1)
    "person",           # 1 (VisDrone 2)
    "bicycle",          # 2 (VisDrone 3)
    "car",              # 3 (VisDrone 4)
    "van",              # 4 (VisDrone 5)
    "truck",            # 5 (VisDrone 6)
    "tricycle",         # 6 (VisDrone 7)
    "awning-tricycle",  # 7 (VisDrone 8)
    "bus",              # 8 (VisDrone 9)
    "motor",            # 9 (VisDrone 10)
]

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def parse_visdrone_line(line: str) -> Optional[dict]:
    line = line.strip()
    if not line or line.startswith("#"):
        return None
    if "," in line:
        parts = [p.strip() for p in line.split(",")]
    else:
        parts = line.split()
    if len(parts) < 8:
        return None
    try:
        x = int(round(float(parts[0])))
        y = int(round(float(parts[1])))
        w = int(round(float(parts[2])))
        h = int(round(float(parts[3])))
        score = float(parts[4])
        cat = int(float(parts[5]))
        trunc = int(float(parts[6]))
        occ = int(float(parts[7]))
    except (ValueError, IndexError):
        return None

    if w <= 0 or h <= 0:
        return None

    return {
        "x": x,
        "y": y,
        "w": w,
        "h": h,
        "score": score,
        "cat": cat,
        "trunc": trunc,
        "occ": occ,
    }


def should_blur_box(b: dict, min_size: int = 10) -> Tuple[bool, str]:
    """Check if box triggers initial blur condition."""
    if b["cat"] == 0:
        return True, "ignored_region"
    if b["cat"] == 11:
        return True, "others_class"
    if b["occ"] > 0:
        return True, f"occlusion_{b['occ']}"
    if b["trunc"] > 0:
        return True, f"truncation_{b['trunc']}"
    if b["w"] < min_size or b["h"] < min_size:
        return True, f"small_box_{b['w']}x{b['h']}"
    return False, ""


def process_single_image(
    img_path: str,
    ann_path: Optional[str],
    out_img_dir: str,
    out_ann_dir: str,
    out_lbl_dir: str,
    min_size: int = 10,
    max_blur_ratio: float = 0.20,
    blur_kernel_size: int = 51,
) -> dict:
    """Process a single image and annotation file according to all rules."""
    stats = {
        "filename": os.path.basename(img_path),
        "status": "kept",  # kept | rejected_blurred | no_image | no_ann
        "rejection_reason": "",
        "orig_boxes": 0,
        "kept_boxes": 0,
        "blurred_boxes": 0,
        "blur_reasons": {},
        "blur_ratio": 0.0,
    }

    if not os.path.isfile(img_path):
        stats["status"] = "no_image"
        stats["rejection_reason"] = "Image file not found"
        return stats

    img = cv2.imread(img_path)
    if img is None:
        stats["status"] = "corrupt_image"
        stats["rejection_reason"] = "Could not decode image"
        return stats

    img_h, img_w = img.shape[:2]
    total_pixels = img_w * img_h

    # Parse annotations if they exist
    boxes = []
    if ann_path and os.path.isfile(ann_path):
        with open(ann_path, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                b = parse_visdrone_line(line)
                if b:
                    boxes.append(b)

    stats["orig_boxes"] = len(boxes)

    # Separate initial blur boxes and candidate clean boxes
    blur_boxes = []
    clean_boxes = []
    for b in boxes:
        to_blur, reason = should_blur_box(b, min_size=min_size)
        if to_blur:
            blur_boxes.append(b)
            stats["blur_reasons"][reason] = stats["blur_reasons"].get(reason, 0) + 1
        else:
            clean_boxes.append(b)

    # Build initial blur mask
    blur_mask = np.zeros((img_h, img_w), dtype=np.uint8)
    for b in blur_boxes:
        x1 = max(0, min(img_w, b["x"]))
        y1 = max(0, min(img_h, b["y"]))
        x2 = max(0, min(img_w, b["x"] + b["w"]))
        y2 = max(0, min(img_h, b["y"] + b["h"]))
        if x2 > x1 and y2 > y1:
            blur_mask[y1:y2, x1:x2] = 1

    # Cascading blur: iteratively check if any clean box intersects the blur mask
    while True:
        cascaded = []
        remaining_clean = []
        for b in clean_boxes:
            x1 = max(0, min(img_w, b["x"]))
            y1 = max(0, min(img_h, b["y"]))
            x2 = max(0, min(img_w, b["x"] + b["w"]))
            y2 = max(0, min(img_h, b["y"] + b["h"]))
            if x2 > x1 and y2 > y1:
                patch = blur_mask[y1:y2, x1:x2]
                if np.any(patch > 0):
                    cascaded.append(b)
                    blur_mask[y1:y2, x1:x2] = 1
                else:
                    remaining_clean.append(b)
            else:
                # Invalid bounds, blur and drop
                cascaded.append(b)

        if not cascaded:
            break

        for b in cascaded:
            blur_boxes.append(b)
            stats["blur_reasons"]["cascade_overlap"] = stats["blur_reasons"].get("cascade_overlap", 0) + 1
        clean_boxes = remaining_clean

    # Calculate blur ratio
    blurred_pixels = int(np.count_nonzero(blur_mask))
    blur_ratio = blurred_pixels / total_pixels if total_pixels > 0 else 0.0
    stats["blur_ratio"] = round(blur_ratio, 4)
    stats["blurred_boxes"] = len(blur_boxes)
    stats["kept_boxes"] = len(clean_boxes)

    # Check 20% rejection rule
    if blur_ratio > max_blur_ratio:
        stats["status"] = "rejected_blurred"
        stats["rejection_reason"] = f"Blur ratio {blur_ratio:.1%} exceeds maximum threshold {max_blur_ratio:.1%}"
        return stats

    # Apply Gaussian blur to the image for all blur boxes
    if blur_boxes:
        for b in blur_boxes:
            x1 = max(0, min(img_w, b["x"]))
            y1 = max(0, min(img_h, b["y"]))
            x2 = max(0, min(img_w, b["x"] + b["w"]))
            y2 = max(0, min(img_h, b["y"] + b["h"]))
            if x2 <= x1 or y2 <= y1:
                continue
            bw = x2 - x1
            bh = y2 - y1
            # Compute effective kernel size based on box dimensions
            k = max(15, min(blur_kernel_size, (min(bw, bh) // 2) | 1))
            k = max(3, k | 1)
            patch = img[y1:y2, x1:x2]
            blurred_patch = cv2.GaussianBlur(patch, (k, k), 0)
            img[y1:y2, x1:x2] = blurred_patch

    # Save output image
    stem = os.path.splitext(os.path.basename(img_path))[0]
    out_img_path = os.path.join(out_img_dir, os.path.basename(img_path))
    cv2.imwrite(out_img_path, img)

    # Save output annotations (VisDrone format)
    out_ann_path = os.path.join(out_ann_dir, f"{stem}.txt")
    with open(out_ann_path, "w", encoding="utf-8") as f:
        for b in clean_boxes:
            score_val = int(b["score"]) if b["score"] == int(b["score"]) else round(b["score"], 3)
            f.write(f"{b['x']},{b['y']},{b['w']},{b['h']},{score_val},{b['cat']},{b['trunc']},{b['occ']}\n")

    # Save output labels (YOLO format: class cx cy w h)
    out_lbl_path = os.path.join(out_lbl_dir, f"{stem}.txt")
    with open(out_lbl_path, "w", encoding="utf-8") as f:
        for b in clean_boxes:
            # Map VisDrone category (1..10) to YOLO target class (0..9)
            cat = b["cat"]
            if 1 <= cat <= 10:
                yolo_cls = cat - 1
                cx = (b["x"] + b["w"] / 2.0) / img_w
                cy = (b["y"] + b["h"] / 2.0) / img_h
                nw = b["w"] / img_w
                nh = b["h"] / img_h
                # Clamp normalized values [0, 1]
                cx = max(0.0, min(1.0, cx))
                cy = max(0.0, min(1.0, cy))
                nw = max(0.0, min(1.0, nw))
                nh = max(0.0, min(1.0, nh))
                f.write(f"{yolo_cls} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}\n")

    return stats


def process_split(
    split_name: str,
    split_dir: str,
    output_split_dir: str,
    min_size: int = 10,
    max_blur_ratio: float = 0.20,
    blur_kernel_size: int = 51,
    num_workers: int = 4,
) -> dict:
    """Process all images in a VisDrone split."""
    print(f"\n=======================================================")
    print(f" Processing split: {split_name}")
    print(f" Source: {split_dir}")
    print(f" Output: {output_split_dir}")
    print(f"=======================================================")

    img_dir = os.path.join(split_dir, "images")
    if not os.path.isdir(img_dir):
        img_dir = split_dir

    ann_dir = os.path.join(split_dir, "annotations")
    if not os.path.isdir(ann_dir):
        ann_dir = os.path.join(split_dir, "labels")

    # Find all images
    img_paths = []
    for ext in IMAGE_EXTS:
        img_paths.extend(glob.glob(os.path.join(img_dir, f"*{ext}")))
        img_paths.extend(glob.glob(os.path.join(img_dir, f"*{ext.upper()}")))
    img_paths = sorted(list(set(img_paths)))

    print(f"Found {len(img_paths)} images in {split_name}.")

    out_img_dir = os.path.join(output_split_dir, "images")
    out_ann_dir = os.path.join(output_split_dir, "annotations")
    out_lbl_dir = os.path.join(output_split_dir, "labels")

    os.makedirs(out_img_dir, exist_ok=True)
    os.makedirs(out_ann_dir, exist_ok=True)
    os.makedirs(out_lbl_dir, exist_ok=True)

    results = []
    total_imgs = len(img_paths)

    # Process images with ProcessPoolExecutor
    tasks = []
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        for idx, img_p in enumerate(img_paths):
            stem = os.path.splitext(os.path.basename(img_p))[0]
            ann_p = None
            if os.path.isdir(ann_dir):
                candidate = os.path.join(ann_dir, f"{stem}.txt")
                if os.path.isfile(candidate):
                    ann_p = candidate

            future = executor.submit(
                process_single_image,
                img_p,
                ann_p,
                out_img_dir,
                out_ann_dir,
                out_lbl_dir,
                min_size=min_size,
                max_blur_ratio=max_blur_ratio,
                blur_kernel_size=blur_kernel_size,
            )
            tasks.append(future)

        completed = 0
        for f in as_completed(tasks):
            res = f.result()
            results.append(res)
            completed += 1
            if completed % 100 == 0 or completed == total_imgs:
                sys.stdout.write(f"\rProgress: [{completed}/{total_imgs}] images processed ({(completed/total_imgs)*100:.1f}%)")
                sys.stdout.flush()

    print("\n")

    # Aggregate statistics
    kept_count = sum(1 for r in results if r["status"] == "kept")
    rejected_count = sum(1 for r in results if r["status"] == "rejected_blurred")
    total_orig_boxes = sum(r["orig_boxes"] for r in results)
    total_kept_boxes = sum(r["kept_boxes"] for r in results if r["status"] == "kept")
    total_blurred_boxes = sum(r["blurred_boxes"] for r in results)

    reasons_agg = {}
    for r in results:
        for k, v in r["blur_reasons"].items():
            reasons_agg[k] = reasons_agg.get(k, 0) + v

    split_summary = {
        "split_name": split_name,
        "total_images": total_imgs,
        "kept_images": kept_count,
        "rejected_images": rejected_count,
        "rejection_rate": f"{(rejected_count/total_imgs)*100:.1f}%" if total_imgs > 0 else "0%",
        "total_orig_boxes": total_orig_boxes,
        "total_kept_boxes": total_kept_boxes,
        "total_blurred_boxes": total_blurred_boxes,
        "blur_reasons": reasons_agg,
        "results": results,
    }

    print(f"--- Summary for {split_name} ---")
    print(f"Kept Images:      {kept_count} / {total_imgs} ({(kept_count/total_imgs)*100:.1f}%)")
    print(f"Rejected (>20%):  {rejected_count} / {total_imgs} ({(rejected_count/total_imgs)*100:.1f}%)")
    print(f"Original Boxes:   {total_orig_boxes}")
    print(f"Kept Clean Boxes: {total_kept_boxes} (surviving)")
    print(f"Blurred Boxes:    {total_blurred_boxes}")
    print(f"Blur breakdown:   {reasons_agg}")

    return split_summary


def generate_data_yaml(output_dir: str, splits_present: List[str]):
    """Generate YOLO data.yaml configuration."""
    yaml_lines = [
        f"# VisDrone Cleaned Dataset for YOLO & VIAT",
        f"path: {os.path.abspath(output_dir)}",
    ]
    for sp in splits_present:
        sp_lower = sp.lower()
        if "train" in sp_lower:
            yaml_lines.append(f"train: {sp}/images")
        elif "val" in sp_lower:
            yaml_lines.append(f"val: {sp}/images")
        elif "test" in sp_lower:
            yaml_lines.append(f"test: {sp}/images")

    yaml_lines.extend([
        "",
        f"nc: {len(YOLO_TARGET_CLASSES)}",
        f"names: {YOLO_TARGET_CLASSES}",
    ])

    yaml_path = os.path.join(output_dir, "data.yaml")
    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write("\n".join(yaml_lines) + "\n")
    print(f"Generated dataset config: {yaml_path}")


def generate_report_markdown(summary: dict, output_path: str):
    """Write an extensive markdown report of the preprocessing."""
    lines = [
        "# VisDrone Dataset Processing & Blurring Report",
        "",
        "This report summarizes the dataset filtering, privacy blurring, and validation results.",
        "",
        "## Overall Summary",
        "",
        "| Split | Total Images | Kept Images | Rejected (>20% Blur) | Total Boxes | Kept Clean Boxes |",
        "|---|---|---|---|---|---|",
    ]

    for sp_name, sp_data in summary["splits"].items():
        lines.append(
            f"| **{sp_name}** | {sp_data['total_images']} | "
            f"{sp_data['kept_images']} ({sp_data['kept_images']/(sp_data['total_images'] or 1)*100:.1f}%) | "
            f"{sp_data['rejected_images']} ({sp_data['rejection_rate']}) | "
            f"{sp_data['total_orig_boxes']} | {sp_data['total_kept_boxes']} |"
        )

    lines.extend([
        "",
        "## Blur Triggers & Breakdown",
        "",
    ])

    for sp_name, sp_data in summary["splits"].items():
        lines.append(f"### Split: {sp_name}")
        lines.append("| Trigger Reason | Count |")
        lines.append("|---|---|")
        for reason, count in sorted(sp_data["blur_reasons"].items(), key=lambda x: -x[1]):
            lines.append(f"| `{reason}` | {count} |")
        lines.append("")

    lines.extend([
        "## Applied Rules",
        "1. **Occlusion & Truncation**: All bounding boxes with `occlusion > 0` or `truncation > 0` are masked and blurred.",
        "2. **Small Objects**: All bounding boxes smaller than 10x10 pixels are masked and blurred.",
        "3. **Ignored & Other Regions**: Class 0 (`ignored_region`) and class 11 (`others`) are masked and blurred.",
        "4. **Cascading / Overlap**: Any clean bounding box intersecting an existing blur mask is also blurred and removed.",
        "5. **Image Rejection Threshold**: Any image with >20% blurred area is completely discarded from the final dataset.",
        "",
        "## Target Object Detection Classes (10 classes)",
        "0: `pedestrian`, 1: `person`, 2: `bicycle`, 3: `car`, 4: `van`, 5: `truck`, 6: `tricycle`, 7: `awning-tricycle`, 8: `bus`, 9: `motor`",
    ])

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Generated report: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="VisDrone Preprocessor & Filter for VIAT")
    parser.add_argument(
        "--input_dir",
        type=str,
        default="/media/reza/New Volume/VIAT/VISDRONE",
        help="Path to VISDRONE directory",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/media/reza/New Volume/VIAT/VISDRONE_CLEANED",
        help="Destination directory for processed dataset",
    )
    parser.add_argument(
        "--min_size",
        type=int,
        default=10,
        help="Minimum width/height in pixels. Boxes smaller than this are blurred (default: 10).",
    )
    parser.add_argument(
        "--max_blur_ratio",
        type=float,
        default=0.20,
        help="Maximum blur ratio allowed before dropping an image (default: 0.20).",
    )
    parser.add_argument(
        "--kernel_size",
        type=int,
        default=51,
        help="Gaussian blur maximum kernel size (default: 51).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=max(1, os.cpu_count() - 1),
        help="Number of parallel worker processes.",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=None,
        help="Specific splits to process (e.g. VisDrone2019-DET-train VisDrone2019-DET-val VisDrone2019-DET-test-dev)",
    )

    args = parser.parse_args()

    input_dir = os.path.abspath(args.input_dir)
    output_dir = os.path.abspath(args.output_dir)

    if not os.path.isdir(input_dir):
        print(f"Error: Input directory not found: {input_dir}")
        sys.exit(1)

    os.makedirs(output_dir, exist_ok=True)

    # Detect splits
    detected_splits = []
    if args.splits:
        for s in args.splits:
            sp_path = os.path.join(input_dir, s) if not os.path.isabs(s) else s
            if os.path.isdir(sp_path):
                detected_splits.append((os.path.basename(sp_path), sp_path))
    else:
        entries = sorted(os.listdir(input_dir))
        for e in entries:
            sp_path = os.path.join(input_dir, e)
            if os.path.isdir(sp_path) and (
                "images" in os.listdir(sp_path) or any(k in e.lower() for k in ("train", "val", "test", "dev"))
            ):
                detected_splits.append((e, sp_path))

    if not detected_splits:
        # Check if root itself is a split
        if "images" in os.listdir(input_dir):
            detected_splits.append(("root", input_dir))
        else:
            print(f"Error: No VisDrone splits found in {input_dir}")
            sys.exit(1)

    print(f"Starting VisDrone preprocessing...")
    print(f"Input Directory:  {input_dir}")
    print(f"Output Directory: {output_dir}")
    print(f"Splits found:     {[s[0] for s in detected_splits]}")
    print(f"Min Box Size:     {args.min_size}x{args.min_size}")
    print(f"Max Blur Ratio:   {args.max_blur_ratio:.0%}")
    print(f"Parallel Workers: {args.workers}")

    all_summary = {"splits": {}}
    splits_processed = []

    for split_name, split_path in detected_splits:
        out_split_dir = os.path.join(output_dir, split_name) if split_name != "root" else output_dir
        res = process_split(
            split_name,
            split_path,
            out_split_dir,
            min_size=args.min_size,
            max_blur_ratio=args.max_blur_ratio,
            blur_kernel_size=args.kernel_size,
            num_workers=args.workers,
        )
        all_summary["splits"][split_name] = res
        splits_processed.append(split_name)

    # Generate data.yaml and markdown report
    generate_data_yaml(output_dir, splits_processed)
    report_path = os.path.join(output_dir, "processing_report.md")
    generate_report_markdown(all_summary, report_path)

    # Save full JSON summary
    json_path = os.path.join(output_dir, "processing_stats.json")
    # Clean results from JSON to keep file compact
    compact_summary = {
        "splits": {
            k: {
                sub_k: sub_v for sub_k, sub_v in v.items() if sub_k != "results"
            }
            for k, v in all_summary["splits"].items()
        }
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(compact_summary, f, indent=2)

    print("\n=======================================================")
    print(" Processing Complete!")
    print(f" Cleaned Dataset saved to: {output_dir}")
    print(f" Markdown Report saved to: {report_path}")
    print(f" JSON Summary saved to:    {json_path}")
    print("=======================================================\n")


if __name__ == "__main__":
    main()
