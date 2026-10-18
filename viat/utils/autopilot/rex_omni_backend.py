"""Isolated Rex-Omni detector adapter for Autopilot.

Rex-Omni is an optional dependency.  This module deliberately does not import
``rex_omni`` at module import time so the regular VIAT application remains
importable on installations that do not have the model package installed.

The adapter follows the small interface used by ``CandidateDetector`` and the
backends in :mod:`viat.utils.autopilot.models`: it accepts a BGR NumPy frame and
returns ``(box_xyxy, score, label)`` tuples.  Rex-Omni does not expose a
calibrated confidence for each generated box, so every valid box receives the
same constant score.
"""

from __future__ import annotations

import importlib
import importlib.util
import math
import os
import sys
from collections.abc import Mapping, Sequence

import numpy as np

from .weights import missing_message, resolve_hf


_REX_MODULE = "rex_omni"
_AWQ_SUFFIX = "-awq"


def _is_awq_key(key):
    """Return whether *key* names the Rex-Omni AWQ checkpoint variant."""

    value = str(key).replace("\\", "/").rstrip("/")
    return os.path.basename(value).casefold().endswith(_AWQ_SUFFIX)


def _vllm_available():
    """Check for vLLM without importing the optional, heavyweight package."""

    try:
        return importlib.util.find_spec("vllm") is not None
    except (ImportError, ValueError):
        # A test double (or an already-imported module without a spec) can be
        # present in sys.modules even though find_spec cannot inspect it.
        return "vllm" in sys.modules


def _flash_attention_available():
    """Return whether Rex-Omni's optional FlashAttention dependency is present."""

    try:
        return importlib.util.find_spec("flash_attn") is not None
    except (ImportError, ValueError):
        return "flash_attn" in sys.modules


def _wrapper_class():
    """Import and return RexOmniWrapper only when a model is being loaded."""

    module = importlib.import_module(_REX_MODULE)
    wrapper = getattr(module, "RexOmniWrapper", None)
    if wrapper is None:
        raise ImportError("rex_omni does not expose RexOmniWrapper")
    return wrapper


def _result_mapping(response):
    """Find the per-image result in Rex-Omni's list-or-dict response.

    The documented API returns a list containing one result dictionary for each
    input image.  Accepting a direct dictionary as well makes the adapter
    tolerant of simple wrappers and test doubles, while the ``result`` and
    ``results`` unwraps handle common thin proxy layers without changing the
    documented path.
    """

    current = response
    for _ in range(3):
        if isinstance(current, (list, tuple)):
            if not current:
                return None
            current = current[0]
            continue
        if not isinstance(current, Mapping):
            return None
        if current.get("success") is False:
            return None
        if "extracted_predictions" in current:
            return current
        nested = None
        for key in ("result", "results"):
            candidate = current.get(key)
            if isinstance(candidate, (Mapping, list, tuple)):
                nested = candidate
                break
        if nested is None:
            return current
        current = nested
    return current if isinstance(current, Mapping) else None


def _prediction_groups(response):
    """Return grouped predictions from a Rex-Omni response, if present."""

    result = _result_mapping(response)
    if not isinstance(result, Mapping):
        return None
    predictions = result.get("extracted_predictions", result)
    return predictions if isinstance(predictions, Mapping) else None


def _annotation_records(value):
    """Normalize one category's annotations to an iterable of records."""

    if isinstance(value, Mapping):
        return (value,)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return value
    return ()


def _parse_predictions(response, width, height, score):
    """Convert grouped Rex-Omni box predictions to Autopilot tuples."""

    groups = _prediction_groups(response)
    if groups is None:
        return []

    image_width = float(width)
    image_height = float(height)
    out = []
    for category, annotations in groups.items():
        if category is None:
            continue
        label = category if isinstance(category, str) else str(category)
        if not label:
            continue
        for annotation in _annotation_records(annotations):
            if not isinstance(annotation, Mapping):
                continue
            annotation_type = annotation.get("type", "box")
            if not isinstance(annotation_type, str) or annotation_type.casefold() != "box":
                continue
            coords = annotation.get("coords")
            if isinstance(coords, (str, bytes, bytearray)):
                continue
            try:
                if len(coords) != 4:
                    continue
                x1, y1, x2, y2 = (float(value) for value in coords)
            except (TypeError, ValueError, OverflowError):
                continue
            if not all(math.isfinite(value) for value in (x1, y1, x2, y2)):
                continue

            # Rex reports absolute pixel coordinates.  Clip boxes to the
            # current frame, then reject empty or inverted boxes.
            x1 = min(image_width, max(0.0, x1))
            y1 = min(image_height, max(0.0, y1))
            x2 = min(image_width, max(0.0, x2))
            y2 = min(image_height, max(0.0, y2))
            if x2 <= x1 or y2 <= y1:
                continue
            out.append(([x1, y1, x2, y2], float(score), label))
    return out


def _bgr_to_pil(image):
    """Convert a BGR NumPy image to the RGB PIL image Rex-Omni expects."""

    try:
        from PIL import Image
    except ImportError as exc:  # pragma: no cover - depends on the host install
        raise RuntimeError("Rex-Omni requires Pillow to convert VIAT frames") from exc

    array = np.asarray(image)
    if array.ndim != 3 or array.shape[2] < 3:
        raise ValueError("Rex-Omni expects a BGR image with three channels")
    rgb = np.ascontiguousarray(array[..., :3][..., ::-1])
    if rgb.dtype != np.uint8:
        # OpenCV frames are normally uint8.  Converting other numeric frame
        # types keeps the adapter predictable for callers and test doubles.
        rgb = np.clip(rgb, 0, 255).astype(np.uint8)
    return Image.fromarray(rgb, mode="RGB")


class RexOmniBackend:
    """Duck-typed Autopilot backend for the optional Rex-Omni package."""

    constant = True
    ref_score = 1.0

    def __init__(self, key, log=print, cfg=None):
        self.key = key
        self.log = log
        self.cfg = cfg
        self.prompts = []
        self.wrapper = None

    def load(self):
        """Load a local Rex-Omni checkpoint without triggering any download."""

        checkpoint = resolve_hf(self.key)
        if checkpoint is None:
            return False, missing_message(self.key)

        try:
            wrapper = _wrapper_class()
        except Exception as exc:
            return False, f"Rex-Omni package unavailable: {exc}"

        awq = _is_awq_key(self.key)
        if awq and not _vllm_available():
            return False, (
                "Rex-Omni AWQ requires vLLM; install a compatible vLLM build "
                "before selecting the AWQ checkpoint"
            )
        use_vllm = awq
        options = {
            "model_path": checkpoint,
            "backend": "vllm" if use_vllm else "transformers",
            # Deterministic detection settings from the upstream quick-start;
            # the shorter generation budget avoids wasting time on verbose
            # responses when Autopilot only needs structured boxes.
            "max_tokens": 2048,
            "temperature": 0.0,
            "top_p": 0.05,
            "top_k": 1,
            "repetition_penalty": 1.05,
        }
        if use_vllm:
            options["quantization"] = "awq"
        else:
            # RexOmniWrapper forwards this option to Transformers, preventing
            # a local checkpoint miss from falling back to an online lookup.
            options["local_files_only"] = True
            # The upstream wrapper defaults to FlashAttention 2, which is not
            # available on every supported VIAT platform (notably many Windows
            # installs). Use PyTorch SDPA when the optional accelerator is not
            # installed; CUDA users with FlashAttention still get the faster
            # implementation.
            if not _flash_attention_available():
                options["attn_implementation"] = "sdpa"

        try:
            self.wrapper = wrapper(**options)
        except Exception as exc:
            self.wrapper = None
            return False, f"Rex-Omni failed to load from {checkpoint}: {exc}"
        return True, "ok"

    def set_prompts(self, prompts):
        """Set the category phrases sent to Rex-Omni at inference time."""

        self.prompts = [str(prompt) for prompt in (prompts or []) if prompt is not None]

    def set_exemplars(self, exemplars):
        """Rex-Omni's detection API has no visual-exemplar input."""

        return None

    def predict(self, image):
        """Run detection on one BGR NumPy frame."""

        if self.wrapper is None:
            raise RuntimeError("Rex-Omni backend has not been loaded")
        if not self.prompts:
            return []

        array = np.asarray(image)
        if array.ndim != 3 or array.shape[2] < 3:
            return []
        pil_image = _bgr_to_pil(array)
        response = self.wrapper.inference(
            images=pil_image,
            task="detection",
            categories=list(self.prompts),
        )
        return _parse_predictions(response, array.shape[1], array.shape[0], self.ref_score)


__all__ = ["RexOmniBackend"]
