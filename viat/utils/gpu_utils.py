"""GPU device selection and persistence for VIAT.

Picking a GPU here doesn't hot-swap the device mid-session: CUDA only honors
`CUDA_VISIBLE_DEVICES` when it's set before the driver initializes (see
`viat/gpu_env.py`, applied at process startup), and by the time the main window
exists, PyTorch/Ultralytics have already locked onto whatever was visible at
launch. So `save_gpu_index()` just persists the choice for the *next* launch.

`list_gpus()` deliberately shells out to `nvidia-smi` instead of using
`torch.cuda`, so it keeps listing every physical GPU even once this process has
restricted itself to one of them via CUDA_VISIBLE_DEVICES.
"""
import os
import json
import subprocess

from .file_operations import get_config_directory, save_json_atomically

_GPU_SETTINGS_FILENAME = "gpu_settings.json"


def list_gpus():
    """Return all physical GPUs as dicts: {"index", "name", "total_gb", "free_gb"}.

    Uses `nvidia-smi` so the full physical GPU list is visible regardless of any
    CUDA_VISIBLE_DEVICES restriction already in effect for this process. Returns
    an empty list if `nvidia-smi` isn't available (no NVIDIA driver installed).
    """
    try:
        output = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,name,memory.total,memory.used",
             "--format=csv,noheader,nounits"],
            stderr=subprocess.DEVNULL, timeout=5,
        ).decode("utf-8", "ignore")
    except Exception:
        return []

    gpus = []
    for line in output.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != 4:
            continue
        try:
            index = int(parts[0])
            total_mib = float(parts[2])
            used_mib = float(parts[3])
        except ValueError:
            continue
        gpus.append({
            "index": index,
            "name": parts[1],
            "total_gb": total_mib / 1024,
            "free_gb": (total_mib - used_mib) / 1024,
        })
    return gpus


def get_active_gpu_name():
    """Return the name of the GPU this running process actually initialized CUDA
    on (reflects CUDA_VISIBLE_DEVICES as restricted at launch), or None if
    unavailable. Best-effort, torch is optional."""
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.get_device_name(0)
    except Exception:
        pass
    return None


def _settings_path():
    return os.path.join(get_config_directory(), _GPU_SETTINGS_FILENAME)


def get_saved_gpu_index():
    """Return the persisted GPU index preference, or None if unset."""
    path = _settings_path()
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r") as f:
            return json.load(f).get("gpu_index")
    except Exception:
        return None


def save_gpu_index(index):
    """Persist the chosen physical GPU index. Applied on the next app launch."""
    config_dir = get_config_directory()
    if not os.path.exists(config_dir):
        os.makedirs(config_dir)
    save_json_atomically(_settings_path(), {"gpu_index": index})
