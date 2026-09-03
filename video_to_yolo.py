#!/usr/bin/env python3
"""
video_to_yolo.py

Convert a folder of videos + custom per-frame annotation .txt files (Raya format)
into a standard YOLO-formatted image dataset (images/ + labels/ + data.yaml).

------------------------------------------------------------------------
FEATURES
------------------------------------------------------------------------
- O(N) Two-Pass Uniform Class Balancing: Samples over-represented classes
  evenly across all videos from beginning to end (no video is cut off early).
- Smart Object-Focused Multi-Cropping: Generates multiple high-res sub-images
  (e.g. 640x640) from 4K/1080p frames around object clusters and distant objects.
- Letterbox/Pillarbox Padding Removal: Auto-detects and crops black borders per-frame.
- Background Frame Thinning: Randomly drops empty frames ([]) to control background ratio.
- Horizontal Flip Augmentation: Randomly creates mirrored images with inverted YOLO coords.
- Flexible Split Export: Single flat folder, preserve subfolder splits, or auto-split.

Usage:
  Direct: Edit PARAMS below and run `python3 video_to_yolo.py`
  CLI:    `python3 video_to_yolo.py --source-dir /path/to/videos --output-dir /path/to/yolo`
"""

import sys
import argparse
from pathlib import Path

# Add project root to sys.path
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from viat.converters.video_to_yolo import convert_video_dataset_to_yolo

# ======================================================================== #
# PARAMS  -- edit these for direct script execution without CLI flags
# ======================================================================== #
SOURCE_DIR = "/media/reza/01DCFBE81EB84710/MOD_DATASET/render_19"
OUTPUT_DIR = "/home/reza/Vision/datasets/fix_wings/artificialImages/images/train"
YAML_PATH = None  # e.g. "/path/to/data.yaml" or None to auto-create

DIST = 1           # sample every Nth frame (1 = every frame)
IMG_EXT = ".jpg"   # saved image extension (.jpg, .png, .webp)

# ---- class merging & balancing ----
MANUAL_CLASS_MAP = {}         # e.g. {"motorbike": "bike", "person": "human"}
AUTO_CREATE_NEW_CLASSES = True

# Max instances per class across whole dataset (e.g. {"car": 25000, "bus": 20000})
# Uses O(N) uniform stride across all videos so no videos are dropped early.
MAX_INSTANCES_PER_CLASS = None

# ---- padding removal ----
REMOVE_PADDING = False
PADDING_BLACK_THRESH = 16

# ---- background frame thinning ----
BACKGROUND_REMOVE_PERCENT = 0  # 0-100% chance to drop empty frames ([])

# ---- smart object-focused multi-cropping ----
ENABLE_SMART_CROP = False
CROP_WIDTH = 640
CROP_HEIGHT = 640
MAX_CROPS_PER_FRAME = 3
MIN_VISIBILITY = 0.4
CONTEXT_PADDING = 0.2

# ---- pre-augmentation & filtering ----
FLIP_AUGMENT_PERCENT = 0       # 0-100% chance to add horizontal flip
MIN_BOX_SIZE_PX = 2.0          # minimum box width/height in pixels

# ---- export mode ----
SPLIT_MODE = "single"          # "single" (default) | "preserve" | "auto"
SPLIT_RATIOS = (0.8, 0.2, 0.0) # (train, val, test) if SPLIT_MODE == "auto"

RANDOM_SEED = 42
# ======================================================================== #


def parse_args():
    parser = argparse.ArgumentParser(description="Convert video dataset into YOLO image dataset")
    parser.add_argument("--source-dir", type=str, default=SOURCE_DIR, help="Source folder containing videos + .txt files")
    parser.add_argument("--output-dir", type=str, default=OUTPUT_DIR, help="Output folder for YOLO images/ and labels/")
    parser.add_argument("--yaml", type=str, default=YAML_PATH, help="Existing YOLO data.yaml to merge into")
    parser.add_argument("--dist", type=int, default=DIST, help="Sample every Nth frame (default: 1)")
    parser.add_argument("--img-ext", type=str, default=IMG_EXT, help="Image extension (.jpg, .png, .webp)")
    parser.add_argument("--remove-padding", action="store_true", default=REMOVE_PADDING, help="Auto-remove black letterbox/pillarbox padding")
    parser.add_argument("--black-thresh", type=int, default=PADDING_BLACK_THRESH, help="Black pixel threshold for padding detection")
    parser.add_argument("--bg-remove-pct", type=float, default=BACKGROUND_REMOVE_PERCENT, help="Percentage to drop empty background frames")
    parser.add_argument("--smart-crop", action="store_true", default=ENABLE_SMART_CROP, help="Enable object-focused multi-cropping")
    parser.add_argument("--crop-size", type=int, nargs=2, default=[CROP_WIDTH, CROP_HEIGHT], help="Target crop width and height")
    parser.add_argument("--max-crops", type=int, default=MAX_CROPS_PER_FRAME, help="Max sub-crops per frame")
    parser.add_argument("--min-visibility", type=float, default=MIN_VISIBILITY, help="Minimum visible box area ratio to keep cropped boxes")
    parser.add_argument("--flip-pct", type=float, default=FLIP_AUGMENT_PERCENT, help="Horizontal flip augmentation percentage")
    parser.add_argument("--split-mode", type=str, default=SPLIT_MODE, choices=["single", "preserve", "auto"], help="Export layout mode")
    return parser.parse_args()


def main():
    if len(sys.argv) > 1:
        args = parse_args()
        s_dir = Path(args.source_dir)
        o_dir = Path(args.output_dir)
        y_path = Path(args.yaml) if args.yaml else None
        dist = args.dist
        img_ext = args.img_ext
        rem_pad = args.remove_padding
        b_thresh = args.black_thresh
        bg_pct = args.bg_remove_pct
        s_crop = args.smart_crop
        c_size = (args.crop_size[0], args.crop_size[1])
        m_crops = args.max_crops
        min_vis = args.min_visibility
        flip_pct = args.flip_pct
        sp_mode = args.split_mode
    else:
        s_dir = Path(SOURCE_DIR)
        o_dir = Path(OUTPUT_DIR)
        y_path = Path(YAML_PATH) if YAML_PATH else None
        dist = DIST
        img_ext = IMG_EXT
        rem_pad = REMOVE_PADDING
        b_thresh = PADDING_BLACK_THRESH
        bg_pct = BACKGROUND_REMOVE_PERCENT
        s_crop = ENABLE_SMART_CROP
        c_size = (CROP_WIDTH, CROP_HEIGHT)
        m_crops = MAX_CROPS_PER_FRAME
        min_vis = MIN_VISIBILITY
        flip_pct = FLIP_AUGMENT_PERCENT
        sp_mode = SPLIT_MODE

    print("================================================================")
    print("🚀 Video Dataset to YOLO Dataset Converter")
    print(f"  Source:     {s_dir}")
    print(f"  Output:     {o_dir}")
    print(f"  Stride:     {dist}")
    print(f"  Smart Crop: {s_crop} {c_size if s_crop else ''}")
    print(f"  Split Mode: {sp_mode}")
    print("================================================================")

    last_pct = -1
    for progress_pct, msg in convert_video_dataset_to_yolo(
        source_dir=s_dir,
        output_dir=o_dir,
        yaml_path=y_path,
        dist=dist,
        img_ext=img_ext,
        remove_padding=rem_pad,
        black_thresh=b_thresh,
        bg_remove_percent=bg_pct,
        enable_smart_crop=s_crop,
        crop_size=c_size,
        max_crops_per_frame=m_crops,
        min_visibility=min_vis,
        context_padding=CONTEXT_PADDING,
        max_instances_per_class=MAX_INSTANCES_PER_CLASS,
        flip_augment_percent=flip_pct,
        min_box_size_px=MIN_BOX_SIZE_PX,
        manual_class_map=MANUAL_CLASS_MAP,
        auto_create_classes=AUTO_CREATE_NEW_CLASSES,
        split_mode=sp_mode,
        split_ratios=SPLIT_RATIOS,
        random_seed=RANDOM_SEED,
    ):
        if progress_pct != last_pct:
            print(f"[{progress_pct:3d}%] {msg}")
            last_pct = progress_pct

    print("\n🎉 Done! YOLO dataset created successfully.")


if __name__ == "__main__":
    main()
