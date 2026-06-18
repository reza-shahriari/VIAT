"""
Dataset workflow logging for VIAT.

Maintains a ``DATASET_LOG.md`` file at the root of each dataset folder,
recording:

  * **Provenance** -- source URL, version, license (from Roboflow data.yaml
    ``roboflow:`` block when available)
  * **Original state** -- image count, label format, classes, layout, timestamp
    of first open
  * **Operations log** -- every dataset op (remove bad frames, remap class,
    remove grayscale, remove duplicates, ...) with: op name, how many frames
    affected, parameters, timestamp
  * **Current state** -- recomputed counts after each op

If the dataset format / source cannot be auto-detected, the log is still
created with ``TODO: fill in`` placeholders so the user can complete it by
hand later.

Usage from main.py / dataset_ops::

    from utils.dataset_log import init_dataset_log, append_dataset_log

    init_dataset_log(app, info)                    # call once on dataset open
    append_dataset_log(app, "Removed grayscale",   # call after each op
                        affected=12, details="moved to removed/grayscale/")
"""

import os
import datetime
from typing import Optional, List

# --------------------------------------------------------------------------- #
# Roboflow provenance extraction
# --------------------------------------------------------------------------- #


def extract_roboflow_info(root: str) -> dict:
    """Extract Roboflow provenance from data.yaml.

    Roboflow exports include a ``roboflow:`` block:
        roboflow:
          workspace: my-workspace
          project: my-project
          version: 1
          license: CC BY 4.0
          url: https://universe.roboflow.com/...
    We also grab top-level ``path``, ``train``, ``nc``.
    """
    info = {
        "source_url": None,
        "version": None,
        "workspace": None,
        "project": None,
        "license": None,
        "downloaded_from": "Roboflow" if _has_roboflow_block(root) else None,
    }
    data = _load_yaml(root)
    if not data:
        return info
    rf = data.get("roboflow", {})
    if isinstance(rf, dict):
        info["source_url"] = rf.get("url")
        info["version"] = rf.get("version")
        info["workspace"] = rf.get("workspace")
        info["project"] = rf.get("project")
        info["license"] = rf.get("license")
    return info


def _has_roboflow_block(root: str) -> bool:
    for name in ("data.yaml", "data.yml"):
        p = os.path.join(root, name)
        if os.path.isfile(p):
            try:
                with open(p, "r", encoding="utf-8") as f:
                    return "roboflow:" in f.read()
            except OSError:
                pass
    return False


def _load_yaml(root: str) -> Optional[dict]:
    """Load data.yaml as a dict. Uses PyYAML if available, else None."""
    try:
        import yaml
    except ImportError:
        return None
    for name in ("data.yaml", "data.yml"):
        p = os.path.join(root, name)
        if os.path.isfile(p):
            try:
                with open(p, "r", encoding="utf-8") as f:
                    return yaml.safe_load(f)
            except Exception:
                return None
    return None


# --------------------------------------------------------------------------- #
# Log file management
# --------------------------------------------------------------------------- #


def _log_path(app) -> str:
    """Return the absolute path to DATASET_LOG.md for the current dataset."""
    info = getattr(app, "_viat_dataset_info", None)
    if info is not None and hasattr(info, "root"):
        return os.path.join(info.root, "DATASET_LOG.md")
    # fallback: use the folder of the first image
    image_files = getattr(app, "image_files", None)
    if image_files:
        return os.path.join(os.path.dirname(os.path.commonpath(image_files)), "DATASET_LOG.md")
    return None


def _now() -> str:
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def init_dataset_log(app, info) -> str:
    """Create or update DATASET_LOG.md at the dataset root.

    If the log already exists, this only updates the "Current State" section
    and leaves the provenance / original state / operations log intact.

    Returns the path to the log file, or None if no dataset is loaded.
    """
    path = os.path.join(info.root, "DATASET_LOG.md")
    rf = extract_roboflow_info(info.root)

    # Per-split counts
    per_split = {}
    for s in info.splits:
        per_split[s.name] = len(s.images)

    classes_str = ", ".join(info.classes) if info.classes else "(none detected)"

    # If the log already exists, preserve everything and just refresh the
    # Current State section.
    if os.path.isfile(path):
        try:
            existing = open(path, "r", encoding="utf-8").read()
        except OSError:
            existing = ""
        updated = _refresh_current_state(existing, app, info)
        try:
            open(path, "w", encoding="utf-8").write(updated)
        except OSError:
            pass
        return path

    # --- New log ---
    lines = []
    lines.append(f"# Dataset Log — {os.path.basename(info.root)}")
    lines.append("")
    lines.append(f"_Created: {_now()}_")
    lines.append("")

    # Provenance
    lines.append("## Provenance")
    lines.append("")
    if rf["downloaded_from"]:
        lines.append(f"- **Source**: {rf['downloaded_from']}")
        lines.append(f"- **URL**: {rf['source_url'] or 'TODO: fill in'}")
        lines.append(f"- **Version**: {rf['version'] or 'TODO: fill in'}")
        lines.append(f"- **Workspace**: {rf['workspace'] or 'TODO: fill in'}")
        lines.append(f"- **Project**: {rf['project'] or 'TODO: fill in'}")
        lines.append(f"- **License**: {rf['license'] or 'TODO: fill in'}")
    else:
        lines.append("- **Source**: TODO: fill in (format not auto-detected as Roboflow)")
        lines.append("- **URL**: TODO: fill in")
        lines.append("- **Version**: TODO: fill in")
        lines.append("- **License**: TODO: fill in")
    lines.append("")

    # Original state
    lines.append("## Original State")
    lines.append("")
    lines.append(f"- **Dataset root**: `{info.root}`")
    lines.append(f"- **Layout**: {info.layout}")
    lines.append(f"- **Label format**: {info.label_format or 'none detected — TODO: fill in'}")
    lines.append(f"- **Total images**: {info.image_count}")
    lines.append(f"- **Splits**: {', '.join(f'{k}={v}' for k, v in per_split.items())}")
    lines.append(f"- **Classes** ({len(info.classes)}): {classes_str}")
    lines.append(f"- **Classes source**: {info.classes_source or 'inferred / TODO: verify'}")
    if info.classes_conflict:
        lines.append(f"- ⚠ **Class conflict**: {info.classes_conflict}")
    lines.append("")

    # Operations log (empty)
    lines.append("## Operations Log")
    lines.append("")
    lines.append("| Timestamp | Operation | Affected | Details |")
    lines.append("|---|---|---|---|")
    lines.append("")

    # Current state (filled by _refresh_current_state on subsequent runs)
    lines.append("## Current State")
    lines.append("")
    lines.append(f"_Last updated: {_now()}_")
    lines.append("")
    lines.append(f"- **Total images**: {getattr(app, 'total_frames', info.image_count)}")
    lines.append(f"- **Annotated frames**: {len(getattr(app, 'frame_annotations', {}))}")
    lines.append("")

    content = "\n".join(lines) + "\n"
    try:
        open(path, "w", encoding="utf-8").write(content)
    except OSError:
        pass
    return path


def append_dataset_log(app, operation: str, affected: int = 0, details: str = "") -> None:
    """Append a row to the Operations Log table in DATASET_LOG.md.

    Also refreshes the Current State section.

    Args:
        app: the VideoAnnotationTool main window.
        operation: short name, e.g. "Removed grayscale images".
        affected: how many frames/files were affected.
        details: free-text details (folder moved to, class renamed, etc.).
    """
    path = _log_path(app)
    if path is None or not os.path.isfile(path):
        # No log yet -- can't append. Silently skip (init should have run).
        return

    try:
        content = open(path, "r", encoding="utf-8").read()
    except OSError:
        return

    ts = _now()
    # Escape pipes in details for markdown table
    details_safe = details.replace("|", "\\|").replace("\n", " ")
    row = f"| {ts} | {operation} | {affected} | {details_safe} |"

    # Insert the row right after the table header + separator.
    # The table starts with "| Timestamp | Operation |" and the next line is
    # "|---|---|---|---|". We insert after that separator line.
    marker = "|---|---|---|---|"
    idx = content.find(marker)
    if idx == -1:
        # Table structure missing; append at end of Operations Log section
        content += "\n" + row + "\n"
    else:
        insert_pos = idx + len(marker)
        content = content[:insert_pos] + "\n" + row + content[insert_pos:]

    # Refresh Current State
    info = getattr(app, "_viat_dataset_info", None)
    if info is not None:
        content = _refresh_current_state(content, app, info)

    try:
        open(path, "w", encoding="utf-8").write(content)
    except OSError:
        pass


def _refresh_current_state(content: str, app, info) -> str:
    """Replace the '## Current State' section with fresh numbers."""
    lines = content.split("\n")
    out = []
    in_current = False
    for line in lines:
        if line.strip().startswith("## Current State"):
            in_current = True
            out.append(line)
            out.append("")
            out.append(f"_Last updated: {_now()}_")
            out.append("")
            out.append(f"- **Total images**: {getattr(app, 'total_frames', '?')}")
            out.append(f"- **Annotated frames**: {len(getattr(app, 'frame_annotations', {}))}")
            out.append("")
            # Skip the old current-state content (until the next ## or EOF)
            continue
        if in_current:
            if line.strip().startswith("## ") or line.strip().startswith("# "):
                in_current = False
                out.append(line)
            # else: skip old current-state line
        else:
            out.append(line)
    return "\n".join(out)
