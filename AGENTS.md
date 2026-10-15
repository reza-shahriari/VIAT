# Agent instructions

## Project shape

- VIAT is a PyQt5 desktop application. The real application package and source entrypoint are `viat/` and `viat/run.py`; run it with `python -m viat.run` (or `python viat/run.py`).
- `viat/main.py` is the `VideoAnnotationTool(QMainWindow)` coordinator. Keep feature logic in `viat/utils/` or `viat/managers/` and wire thin calls from `main.py` rather than expanding the coordinator.
- `viat/widgets/` contains UI docks/dialogs; `viat/evaluation/` is the evaluation engine; `viat/tracking/` contains the tracker manager plus vendored E.T.Track/OSTrack code.
- `viat/utils/label_formats/` is a registry-based plugin system. A new format needs a `LabelFormat` implementation and registration in `label_formats/__init__.py`.
- `sam2-trt/` is a separate C++/CUDA/TensorRT project with its own `pyproject.toml`, CMake build, and Python tests. Do not treat it as ordinary `viat/` Python code.
- `installation_tools/` is packaging/PyInstaller code; only change it for packaging/distribution work. Preserve vendored tracker code unless the task explicitly targets that integration.
- Read `FAST_TRACKER.MD` before changing SAM/tracker selection or the hybrid tracking workflow.

## Setup and run

The repository has two non-equivalent dependency specifications. The documented source setup is:

```bash
pip install -r requirements.txt
pip install -e .                 # preferred for development
# or: python setup.py install
```

`environment.yml` creates a Python 3.9 environment but installs only a small subset of the requirements. Optional model backends are installed separately when needed (`ultralytics`, `transformers`, or the `dino_req.txt`, `florence_req.txt`, and `locate_anything_req.txt` files). The native SAM2 TensorRT backend is built according to `sam2-trt/README.md` and requires its own CUDA/TensorRT toolchain.

Run the desktop application from the repository root:

```bash
python -m viat.run
```

The installed console entry point in `setup.py` is currently `vat=vat.run:main`, but this repository contains `viat/`, not `vat/`; do not rely on `vat` until packaging is corrected. `setup.py` also disagrees with `requirements.txt` on some versions (notably PyQt5), so use the requirements file for the current source environment and verify dependency changes against both files.

Optional AI imports must fail soft when unavailable. Follow the existing `try/except ImportError` patterns instead of making optional backends hard dependencies.

## Verification

- There is no root pytest configuration or unified test suite for the main `viat/` application, and no configured lint/typecheck command. For a lightweight syntax smoke check, run `python -m compileall viat`.
- TrackEval tests are vendored evaluation tests; run them from their directory: `cd viat/evaluation/trackEval && pytest`.
- SAM2-TRT tests belong to that subproject: `cd sam2-trt && pytest tests/python` (only when working on SAM2-TRT).
- `viat/test_remove.py` and tests under the tracker repositories are ad-hoc/upstream checks, not a VIAT-wide regression suite.
- The full application requires GUI dependencies and may require a display, GPU, model weights, or optional native backends; a headless import/compile check does not validate the UI or AI integrations.

## Important repository facts

- The README and `CLAUDE.md` mention `sam3_src/`, but that directory is not present in the current checkout. Confirm paths before using those instructions.
- Root utility scripts such as `many2single.py`, `video_to_yolo.py`, `process_visdrone.py`, `split_raya_to_frames.py`, `benchmark_ostrack.py`, and `export_ostrack_onnx.py` are standalone scripts, not package entrypoints; inspect each script's CLI before running it.
- `CLAUDE.md` contains the longer repository guidance. Consult it when this compact file is insufficient, especially for backend-specific setup and architecture details.
