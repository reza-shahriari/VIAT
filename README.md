# VIAT — Video & Image Annotation Tool

VIAT is a powerful, GPU-accelerated annotation tool for computer vision tasks. It supports both video and image annotation workflows with an intuitive interface, a rich AI-assisted toolset, and a comprehensive evaluation framework.

---

## 🚀 Feature Overview

### 🎞️ Video Playback & Navigation
- Smooth frame-by-frame playback with play/pause controls
- Timeline slider with precise frame seeking
- Support for MP4, AVI, MKV, and other common video formats
- Scene-change detection with frame marking (`Ctrl+X`)
- Clip-cuts dock for managing video segments

### 🖼️ Annotation Capabilities
- **Bounding Box Annotation** — drag or two-click creation methods
- **Polygon / Segmentation Masks** — draw or import polygons
- **Smart Edge Movement** — edges snap to image gradients for sub-pixel precision
- **Keyframe Interpolation** — auto-fill intermediate frames between two annotated keyframes, configurable step size
- **Per-Object Visibility Ranges** — define disjoint frame segments where each actor is visible, managed entirely from the toolbar
- **Object Attributes** — attach custom key-value attributes (Size, Quality, etc.) to any annotation
- **Context Menu** — right-click on any annotation for quick edit, copy, or delete

### 🤖 AI-Assisted Annotation

#### SAM Suite (Segment Anything Models)
| Backend | Model | Notes |
|---|---|---|
| `SamManager` | SAM 2.1 (Ultralytics) | Auto-downloads weights; interactive point/box prompting |
| `Sam3NativeManager` | SAM3 (Meta / facebookresearch) | Native SAM3 package; image & video prediction |
| `Sam2TrtManager` | SAM2 TRT C++ | TensorRT engine for maximum GPU throughput |

The interactive SAM dock lets you click positive/negative points or draw boxes to generate precise segmentation masks and bounding boxes in real time.

#### Fast Tracker Module
A lightweight tracking pipeline that propagates annotations at high speed, designed as a companion to (or replacement for) SAM on moderate hardware.

| Tracker | Type | Speed |
|---|---|---|
| **E.T.Track** | Transformer SOT | High accuracy, GPU |
| **OSTrack** | One-Stream Transformer | ~100 FPS on modern GPU; ONNX export supported |
| **LightTrack / CSRT / KCF** | Siamese / classical | CPU-friendly fallbacks |

**Hybrid workflow**: SAM generates a precise initial mask → fast tracker propagates across frames → user corrects drifts with SAM.

#### Auto-Annotation (YOLO / YOLOWorld)
- Batch auto-detect objects using a built-in or custom YOLO model
- Configurable confidence threshold and NMS
- Supports `yolov8s-worldv2.pt` (open-vocabulary, zero-shot categories)

#### Zero-Shot Classification
- Classify crops of existing annotations without any training using CLIP-family models
- Bundled **MobileCLIP** TorchScript models (`mobileclip2_b.ts`, `mobileclip_blt.ts`)
- HuggingFace Transformers-based models also supported
- Outputs predicted class labels directly into the annotation dock

#### Segmentation-Video Labeler
- Load a **segmentation render** (e.g., Blender mask pass) as a color-coded video
- Click on a colored region to pick an object; the tool tracks that color across the frame range using HSV thresholding + contour detection
- Adjustable tolerance, minimum area, and polygon-simplification epsilon
- Preview mask before committing; assign class names and actor IDs

---

### 🎨 Class & Dataset Management

#### Class Management
- Unlimited classes with customizable per-class colors
- Color-coded overlays for fast visual identification
- Class Frames Dock — navigate frames by class
- Class Info Dialog — per-class annotation statistics

#### Dataset Integration Wizard
- **Multi-layout YOLO dataset detection**: single-folder, `images/labels/`, and `train/valid/test` splits
- Class-name resolution from `data.yaml`, `classes.txt`, `obj.names`, or index fallback
- Interactive **Dataset Merge Wizard** — import a new dataset into an existing one with class mapping
- Dataset Cleaner — remove duplicates and low-quality samples
- Single-Class Extractor — split a multi-class dataset by class

#### Background Remover
- Strip images with no foreground annotations from a dataset

#### Crop Exporter
- Export annotated crops per class, configurable padding and size

---

### 🔒 Privacy & Blur Tools
- Draw rectangular or polygon blur regions on any frame
- Regions are stored per-frame in the project file
- Analytics report shows total blurred frames and blur-type breakdown

---

### 📊 Evaluation Framework

A built-in evaluation engine (`viat/evaluation/`) supporting:

| Feature | Details |
|---|---|
| **Formats** | VIAT JSON (Raya), YOLO data.yaml, image dataset folders, COCO |
| **Metrics** | mAP@50, mAP@50-95, per-class AP, per-size (S/M/L) breakdown |
| **Class Mapping** | Interactive drag-and-drop class assignment between GT and predictions |
| **Merge Simulator** | Compare mAP before vs. after merging classes |
| **Visual Inspector** | Frame-level GT / TP / FP / FN overlay; synced with the video timeline |
| **Charts** | Matplotlib precision-recall curves and bar charts embedded in the UI |
| **Model Runner** | Run inference from a YOLO model directly inside VIAT for instant evaluation |

---

### 📝 Export Formats

| Format | Extension |
|---|---|
| **VIAT JSON** (native, full fidelity) | `.json` |
| **YOLO** | `.txt` + `data.yaml` |
| **COCO JSON** | `.json` |
| **Pascal VOC** | `.xml` |
| **CreateML** | `.json` |

The format system is pluggable (`viat/utils/label_formats/`) — new formats can be added without touching the core.

---

### 💾 Data & File Management
- **Auto-save** — prevents data loss during long sessions
- **Project Save / Load** — full project state in a single VIAT JSON file
- **Undo / Redo** — full annotation history
- **Labeler Analytics Report** — Markdown report with per-tool usage, annotation source breakdown, blur stats, and prompt logs
- **Compare Raya** — diff two annotation sets side-by-side
- **Import Masks** — convert segmentation mask images into polygon annotations
- **Scene Splitter** — split a video at detected scene boundaries

---

### 🔍 View & UI Controls
- Zoom in/out with scroll wheel or keyboard shortcuts
- Pan with middle-click drag
- Proper aspect-ratio preservation at all zoom levels
- Coordinate transform between display space and image space
- Dockable panels: Annotation Dock, Class Dock, Class Frames, Empty Frames, Uncertain Frames, Video Manager, SAM Interactive, Crop Settings, Clip Cuts

---

## 📦 Installation

### Prerequisites
- Python 3.9+
- CUDA-capable GPU recommended (required for SAM / TensorRT backends)

### From Source

```bash
# Clone the repository
git clone https://github.com/yourusername/VIAT.git
cd VIAT

# Option 1: pip
pip install -r requirements.txt
python setup.py install

# Option 2: conda
conda env create -f environment.yml
conda activate viat
```

### Optional AI Backends

Install only the backends you need:

```bash
# SAM 2.1 / SAM3 / Zero-shot detection (Ultralytics)
pip install ultralytics

# SAM3 native (Meta Research)
pip install -r sam3_src/requirements.txt

# SAM2 TensorRT C++ backend
# See sam2-trt/README.md for TensorRT engine build instructions

# DINO-based detectors
pip install -r dino_req.txt

# Florence model
pip install -r florence_req.txt

# LocateAnything
pip install -r locate_anything_req.txt

# Zero-shot classification via HuggingFace Transformers
pip install transformers
```

### Run

```bash
python -m viat.run
# or
python viat/run.py
```

---

## 📂 Project Structure

```
VIAT/
├── viat/                          # Main application package
│   ├── main.py                    # Main window & application logic
│   ├── canvas.py                  # Video/image canvas widget
│   ├── annotation.py              # Annotation data classes (BoundingBox, etc.)
│   ├── interpolation.py           # Keyframe interpolation engine
│   ├── smart_edge.py              # Smart edge snapping
│   ├── canvas_edge_movement.py    # Edge movement helpers
│   ├── yolo_detect.py             # YOLO detection integration
│   ├── config.py                  # Default settings
│   ├── logger.py                  # App-level logger
│   ├── run.py                     # Entry point
│   ├── managers/                  # Feature managers
│   │   ├── video_manager.py       # Multi-video management
│   │   └── dataset_integration.py # Dataset integration workflow
│   ├── utils/                     # Core utilities
│   │   ├── sam_manager.py         # SAM 2.1 (Ultralytics)
│   │   ├── sam3_native_manager.py # SAM3 (Meta)
│   │   ├── sam2_trt_manager.py    # SAM2 TensorRT backend
│   │   ├── fast_tracker_manager.py # Fast tracker adapter
│   │   ├── zero_shot_manager.py   # CLIP / zero-shot classification
│   │   ├── zero_shot_classifier.py
│   │   ├── dataset_manager.py     # YOLO dataset detection & loading
│   │   ├── dataset_ops.py         # Dataset operations
│   │   ├── dataset_merger.py      # Dataset merge logic
│   │   ├── background_remover.py  # Remove unannotated images
│   │   ├── blur_manager.py        # Privacy blur regions
│   │   ├── object_visibility.py   # Per-object visibility ranges
│   │   ├── seg_video_labeler.py   # Segmentation-video labeling
│   │   ├── analytics_report.py    # Labeler analytics Markdown report
│   │   ├── file_operations.py     # Save/load/export operations
│   │   ├── crop_exporter.py       # Crop export
│   │   ├── single_class_extractor.py
│   │   ├── import_masks.py        # Mask-to-polygon importer
│   │   ├── compare_raya.py        # Annotation set comparison
│   │   ├── scene_splitter.py      # Scene-cut splitting
│   │   ├── performance.py         # Performance monitoring
│   │   └── label_formats/         # Pluggable export formats
│   │       ├── coco.py
│   │       ├── yolo.py
│   │       ├── pascal_voc.py
│   │       ├── createml.py
│   │       └── viat_json.py
│   ├── widgets/                   # UI dialogs & docks
│   │   ├── annotation_dock.py
│   │   ├── auto_annotate_dialog.py
│   │   ├── auto_blur_dialog.py
│   │   ├── background_remover_dialog.py
│   │   ├── batch_prediction_dialog.py
│   │   ├── class_dialog.py
│   │   ├── class_dock.py
│   │   ├── class_frames_dock.py
│   │   ├── class_info_dialog.py
│   │   ├── clip_cuts_dock.py
│   │   ├── compare_raya_dialog.py
│   │   ├── crop_settings_dock.py
│   │   ├── dataset_cleaner_dialog.py
│   │   ├── dataset_wizard_dialog.py
│   │   ├── empty_frames_dock.py
│   │   ├── evaluation_dialog.py
│   │   ├── import_masks_dialog.py
│   │   ├── sam_interactive_dock.py
│   │   ├── scene_detect_dialog.py
│   │   ├── single_class_extractor_dialog.py
│   │   ├── toolbar.py
│   │   ├── tracking_dialog.py
│   │   ├── uncertain_frames_dock.py
│   │   ├── video_manager_dock.py
│   │   └── zero_shot_classification_dialog.py
│   ├── tracking/                  # Tracking backends
│   │   ├── manager.py             # Tracker factory
│   │   ├── ettrack.py             # E.T.Track
│   │   ├── ostrack_tracker.py     # OSTrack
│   │   ├── nossort.py             # NoSSORt
│   │   ├── ettrack_repo/          # E.T.Track source
│   │   ├── ostrack_repo/          # OSTrack source
│   │   └── track_trt/             # TensorRT tracker builds
│   └── evaluation/                # Evaluation framework
│       ├── engine.py
│       ├── bounding_box.py
│       ├── evaluators/
│       ├── inference/
│       ├── trackEval/
│       ├── utils/
│       └── visualization/
├── sam2-trt/                      # SAM2 TensorRT C++ integration
├── sam3_src/                      # SAM3 (Meta) source & scripts
├── installation_tools/            # Packaging / installer scripts
├── requirements.txt               # Core Python dependencies
├── dino_req.txt                   # DINO dependencies
├── florence_req.txt               # Florence model dependencies
├── locate_anything_req.txt        # LocateAnything dependencies
├── environment.yml                # Conda environment
├── setup.py                       # pip-installable setup
├── benchmark_ostrack.py           # OSTrack speed benchmark
├── export_ostrack_onnx.py         # Export OSTrack to ONNX
├── many2single.py                 # Multi-class to single-class converter
├── split_raya_to_frames.py        # Split Raya project by frame
└── FAST_TRACKER.MD                # Fast tracker research notes
```

---

## 💡 Usage

### Getting Started

1. **Open a Video or Image**
   - `File > Open Video` / `File > Open Image Folder`
   - Drag-and-drop files onto the canvas

2. **Create Annotations**
   - Select a class in the Class Dock
   - Choose a tool: bounding box (drag or two-click), polygon, or use an AI tool
   - Draw on the canvas; annotations appear instantly in the Annotation Dock

3. **Use AI Assistance**
   - Open the **SAM Interactive Dock** to click positive/negative points — SAM generates a mask
   - Use **Auto-Annotate** for batch YOLO detection across all frames
   - Use **Zero-Shot Classification** to auto-label existing boxes with CLIP

4. **Track Across Frames**
   - Draw an initial box, then launch the **Fast Tracker** to propagate it through the video
   - Switch to **SAM tracking** for mask-level propagation on GPU

5. **Use Interpolation**
   - Enable **Interpolation Mode**, annotate keyframes, and VIAT fills the gaps automatically

6. **Export Annotations**
   - `File > Export` — choose YOLO, COCO, Pascal VOC, CreateML, or VIAT JSON
   - Or use the Dataset Wizard for a guided merge-and-export workflow

---

## Keyboard Shortcuts

| Shortcut | Action |
|---|---|
| `Space` | Play / Pause video |
| Left / Right Arrow | Previous / Next frame |
| `Ctrl+S` | Save project |
| `Ctrl+Z` | Undo |
| `Ctrl+Y` | Redo |
| `Delete` | Remove selected annotation |
| `+` / `-` | Zoom in / out |
| `Ctrl+X` | Mark / remove frame |
| `Ctrl+A` | Select all annotations on frame |
| `Escape` | Cancel current drawing |

---

## Contributing

Contributions are welcome! Please open an issue to discuss what you would like to change, then submit a Pull Request.

## License

This project is licensed under the [MIT License](LICENSE).

## Contact

For bug reports, feature requests, or questions, please open an issue on the GitHub repository.

---

*VIAT — making annotation faster and smarter for computer vision.*
