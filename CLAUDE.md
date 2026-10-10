# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

VIAT (Video & Image Annotation Tool) is a PyQt5 desktop application for annotating video/image datasets for
computer vision, with AI-assisted tooling (SAM 2.1 / SAM3 / SAM2-TRT, YOLO/YOLO-World auto-detect, zero-shot
CLIP classification, fast SOT trackers) and a built-in mAP evaluation framework. The installable package is
`viat/`; `sam3_src/` and `sam2-trt/` are separate, largely self-contained integrations pulled in as optional
backends (each has its own `pyproject.toml`/build).

## Running the app

```bash
python -m viat.run
# or
python viat/run.py
```

There is no build step for the Python app itself. `installation_tools/` contains PyInstaller-based packaging
scripts for producing distributable builds — only touch these when asked to work on packaging/distribution.

## Installing dependencies

```bash
pip install -r requirements.txt      # core app + evaluation deps
python setup.py install              # or: pip install -e .
```

Optional AI backends are deliberately kept out of `requirements.txt` and installed a la carte:
- `pip install ultralytics` — SAM 2.1, SAM3 (Ultralytics wrapper), YOLO/YOLO-World auto-annotate
- `pip install -r sam3_src/requirements.txt` — native SAM3 (Meta/facebookresearch) package
- SAM2 TensorRT C++ backend — see `sam2-trt/README.md`; built separately, not pip-installable
- `pip install -r dino_req.txt`, `florence_req.txt`, `locate_anything_req.txt` — other detector backends
- `pip install transformers` — HuggingFace-based zero-shot classification

Because these backends are optional, code that imports them (SAM managers, zero-shot classifier, tracking
backends) must fail soft — follow the existing `try/except ImportError` pattern used throughout `viat/utils/`
and `viat/evaluation/engine.py` rather than adding a hard dependency.

## Tests

There is no unified test suite/runner for the `viat/` package itself (no pytest config at the repo root).
Test files that do exist live inside vendored third-party trackers/eval toolkits:
- `viat/evaluation/trackEval/tests/` — TrackEval metric tests, runnable with `pytest` from that directory
- `viat/tracking/ostrack_repo/`, `viat/tracking/ettrack_repo/` — vendored SOT repos with their own test scripts
  (e.g. `tracking/test_exp.py`); these are upstream code, not VIAT's own logic — avoid editing them unless the
  task specifically targets tracker integration.

`sam3_src/` and `sam2-trt/` each have their own `pyproject.toml` and test directories (`sam3_src/test/`,
`sam2-trt/tests/python/`) — run pytest from within those subprojects when working on them specifically.

`viat/test_remove.py` is an ad-hoc script, not a pytest suite.

When adding new logic to `viat/`, there's no established convention to follow for test placement — check with
the user before introducing a test framework/directory structure.

## Architecture

### Main window as coordinator, feature logic in managers/utils

`viat/main.py` defines `VideoAnnotationTool(QMainWindow)`, which is a large coordinator that wires together UI
(docks/dialogs from `viat/widgets/`), canvas state, and dozens of feature managers imported from `viat/utils/`
and `viat/managers/`. Actual feature logic (dataset ops, SAM inference, tracking, export, blur, analytics,
etc.) lives in standalone modules/classes under `viat/utils/*.py`, not in `main.py` itself — `main.py` mostly
delegates to `_viat_*`-aliased functions and manager instances (`self.sam_manager`, `self.dataset_manager`,
etc.) constructed in `init_managers()`. When adding a feature, prefer adding a module in `viat/utils/` (or a
manager under `viat/managers/`) and wiring a thin call from `main.py`, rather than growing `main.py` directly.

### AI backends are pluggable managers behind a common calling convention

Segmentation/tracking backends (`SamManager`, `Sam3NativeManager`, `Sam2TrtManager`, and the trackers under
`viat/tracking/`) are independent, optionally-installed managers selected at runtime based on what's
available/configured — there's no shared abstract base class enforced across all of them, so check the
relevant manager's existing method names (`init`, `update`/`predict`, etc.) before adding a new backend, and
mirror the fail-soft import pattern (see `viat/tracking/manager.py`'s `BaseTracker`/`TrackerManager` for the
tracker side, which does define a small `BaseTracker` interface with `init`/`update`).

### Label formats are a plugin system

`viat/utils/label_formats/` implements export/import formats (YOLO, COCO, Pascal VOC, CreateML, VisDrone,
VIAT's own JSON) as subclasses of `LabelFormat` (`base.py`), each declaring `name`, `extensions`, `per_image`,
and implementing `find_label_file` / `load` / `dump`. New formats are added purely by creating a new subclass
and registering it in `label_formats/__init__.py` — no changes needed elsewhere in the dataset/export code.

### Evaluation framework

`viat/evaluation/` is a fairly self-contained mAP evaluation engine (`engine.py`) with its own evaluators
(`evaluators/`), format converters (`utils/convert_ourformat2json.py`, `convert_ourformat2mot.py`), a vendored
`trackEval/` for MOT-style tracking metrics, and `visualization/` for PR-curve/bar-chart plotting. It supports
VIAT JSON, YOLO `data.yaml`, plain image-dataset folders, and COCO as input formats, and is invoked both from
the evaluation dialog UI (`viat/widgets/evaluation_dialog.py`) and standalone. Optional deps (`pandas`, `rich`)
are imported with fallbacks (see the top of `engine.py`) — keep that pattern for new optional deps here too.

### Tracking module

`viat/tracking/` wraps several Single Object Tracking backends behind `manager.py`'s `TrackerManager`/
`BaseTracker`: classical OpenCV trackers (CSRT/KCF/MIL) plus vendored transformer trackers `ettrack_repo/`
(E.T.Track) and `ostrack_repo/` (OSTrack, with ONNX export support via `export_ostrack_onnx.py` and
TensorRT engine builds under `track_trt/` / `setup_ostrack_trt.sh`). `FAST_TRACKER.MD` documents the design
rationale for this "fast tracker as a companion to SAM" hybrid workflow (SAM for a precise initial
mask/box → fast tracker propagates → user corrects drift with SAM) — read it before changing tracker selection
logic or adding a new tracker.

### Root-level standalone scripts

Several top-level scripts are standalone utilities, not part of the importable `viat` package: `benchmark_ostrack.py`,
`export_ostrack_onnx.py`, `many2single.py` (multi-class → single-class dataset conversion), `process_visdrone.py`,
`split_raya_to_frames.py`, `video_to_yolo.py`. These are run directly with `python <script>.py`, each with its
own CLI args (check each script's `argparse` setup / `__main__` block).
