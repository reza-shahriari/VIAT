"""Restricts the process to one physical GPU before any CUDA-capable library initializes.

CUDA only honors `CUDA_VISIBLE_DEVICES` when it's set before the driver's first
initialization in the process (the first `import torch` / `torch.cuda.*` call, or
equivalent in any other CUDA library). Changing it afterwards is silently ignored,
which is why GPU selection can't be a simple runtime toggle: whichever GPU is
selected in the UI is saved to disk and only takes effect the next time VIAT starts.

This module is intentionally stdlib-only (no relative imports beyond that) so
importing it here, at the very top of the entry point, doesn't itself trigger a
torch/PyQt5 import that would defeat the purpose.
"""
import os
import json


def _settings_path():
    if os.name == "nt":
        config_dir = os.path.join(os.environ["APPDATA"], "VideoAnnotationTool")
    else:
        config_dir = os.path.join(os.path.expanduser("~"), ".config", "VideoAnnotationTool")
    return os.path.join(config_dir, "gpu_settings.json")


def restrict_to_saved_gpu():
    """Read the persisted GPU choice (if any) and restrict this process to it.

    Must run before `torch` (or anything importing it) is imported. No-op if
    CUDA_VISIBLE_DEVICES is already set (e.g. by the shell/launcher) or if no
    preference has been saved yet, leaving all GPUs visible either way.
    """
    if "CUDA_VISIBLE_DEVICES" in os.environ:
        return None

    path = _settings_path()
    if not os.path.exists(path):
        return None

    try:
        with open(path, "r") as f:
            index = json.load(f).get("gpu_index")
    except Exception:
        return None

    if index is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(index)
    return index
