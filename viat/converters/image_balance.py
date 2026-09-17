"""
YOLO Image Dataset Class Balancer.

Applies the same evenly-spaced, non-random class-balancing logic used by
the video-to-YOLO converter (see `video_to_yolo.py`'s `select_with_gap` /
`compute_class_targets`) to an *already-existing* YOLO image dataset -
images/ + labels/, either a single flat folder or split into
train/val/test subfolders. Useful for balancing a dataset that wasn't
necessarily produced by this tool's own video converter (manually
captured photos, a third-party export, a dataset merged from several
sources, etc).

Like the video converter, this never touches the source folder: the
balanced subset is copied (or moved/symlinked, if requested) into a
separate output folder, mirroring whatever split-subfolder structure the
source had.

Architecture mirrors video_to_yolo.py's two-phase design:
  build_balance_plan()   - decode-free: scan labels, decide which images
                            survive. Cheap enough to preview before running.
  execute_balance_plan() - copy/move/symlink the kept pairs + data.yaml.
"""

import os
import shutil
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Callable

try:
    import yaml
except ImportError:
    yaml = None

from .video_to_yolo import select_with_gap, compute_class_targets, load_yaml_classes

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


# --------------------------------------------------------------------------- #
# Discovery & scanning
# --------------------------------------------------------------------------- #

def discover_image_label_pairs(source_dir: Path) -> List[Tuple[Path, Path]]:
    """
    Find every image with a matching YOLO .txt label anywhere under
    source_dir. Handles both a flat images/+labels/ layout and a split
    layout (train/images+labels, val/images+labels, ...) generically: for
    every directory literally named "images", the sibling "labels"
    directory (same parent) is checked for a same-stem .txt file.
    """
    pairs = []
    for root, _dirs, files in os.walk(source_dir):
        root_p = Path(root)
        if root_p.name != "images":
            continue
        labels_dir = root_p.parent / "labels"
        if not labels_dir.is_dir():
            continue
        for f in sorted(files):
            if os.path.splitext(f)[1].lower() in IMAGE_EXTENSIONS:
                img_path = root_p / f
                lbl_path = labels_dir / (Path(f).stem + ".txt")
                if lbl_path.is_file():
                    pairs.append((img_path, lbl_path))
    return pairs


def parse_yolo_label_classes(label_path: Path) -> List[int]:
    """Class indices present in a YOLO label file (one entry per box line, ignoring blank/comment lines)."""
    if not label_path.is_file():
        return []
    indices = []
    for line in label_path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if not parts:
            continue
        try:
            indices.append(int(float(parts[0])))
        except ValueError:
            continue
    return indices


def scan_image_dataset_statistics(
    source_dir: Path,
    yaml_path: Optional[Path] = None,
    progress_callback: Optional[Callable[[int, str], None]] = None,
) -> Dict[str, Any]:
    """
    Discovery pass: what classes exist in this image dataset, how many
    instances/images each has. Cheap (label-text only, no image decode) -
    safe to call before the user has configured any balance settings.
    """
    source_dir = Path(source_dir).expanduser().resolve()
    pairs = discover_image_label_pairs(source_dir)
    if not pairs:
        return {"pairs": [], "per_image_classes": {}, "classes": {}, "global_names": [], "total_images": 0}

    global_names: List[str] = []
    if yaml_path and Path(yaml_path).is_file():
        global_names, _ = load_yaml_classes(Path(yaml_path))
    elif (source_dir / "data.yaml").is_file():
        global_names, _ = load_yaml_classes(source_dir / "data.yaml")

    def name_for(idx: int) -> str:
        return global_names[idx] if 0 <= idx < len(global_names) else f"class_{idx}"

    instance_counts: Dict[str, int] = {}
    image_counts: Dict[str, int] = {}
    per_image_classes: Dict[Path, List[str]] = {}

    total = len(pairs)
    for i, (img_path, lbl_path) in enumerate(pairs):
        if progress_callback and i % 200 == 0:
            progress_callback(int(i / total * 100), f"Scanning {lbl_path.name}...")
        names_here = [name_for(ix) for ix in parse_yolo_label_classes(lbl_path)]
        per_image_classes[img_path] = names_here
        seen = set()
        for n in names_here:
            instance_counts[n] = instance_counts.get(n, 0) + 1
            if n not in seen:
                image_counts[n] = image_counts.get(n, 0) + 1
                seen.add(n)

    classes_report = {
        n: {"instance_count": instance_counts[n], "image_count": image_counts.get(n, 0)}
        for n in sorted(instance_counts.keys())
    }

    if progress_callback:
        progress_callback(100, f"Scanned {total} image(s), {len(classes_report)} class(es) found.")

    return {
        "pairs": pairs,
        "per_image_classes": per_image_classes,
        "classes": classes_report,
        "global_names": global_names,
        "total_images": total,
    }


# --------------------------------------------------------------------------- #
# Phase A: Planning
# --------------------------------------------------------------------------- #

def build_balance_plan(
    source_dir: Path,
    yaml_path: Optional[Path] = None,
    balance_mode: str = "auto_min",  # "manual" | "auto_min"
    manual_caps: Optional[Dict[str, int]] = None,
    classes_in_balance: Optional[List[str]] = None,
    min_gap: int = 0,
    progress_callback: Optional[Callable[[int, str], None]] = None,
    cancel_callback: Optional[Callable[[], bool]] = None,
) -> Dict[str, Any]:
    """
    Decide which images survive, without touching any files. Reuses the
    same deterministic, evenly-spaced selection (`select_with_gap`) the
    video converter uses, so thinning an over-represented class doesn't
    just drop images at random - it spreads the kept ones evenly across
    the whole (path-sorted) collection, and a scarce class sharing an
    image with an over-represented one still protects that image, exactly
    like the video converter's frame-level balancing.
    """
    stats = scan_image_dataset_statistics(source_dir, yaml_path, progress_callback)
    pairs = stats["pairs"]
    if not pairs:
        raise ValueError(f"No YOLO image+label pairs found under {source_dir} (looked for images/+labels/ folders)")
    if cancel_callback and cancel_callback():
        return {"cancelled": True}

    per_image_classes = stats["per_image_classes"]
    # Sort deterministically so evenly-spaced selection spreads picks across
    # the whole collection (and, since filenames from this tool's own video
    # converter are "<video>_<frame>...", across different source videos
    # too) instead of depending on directory-listing order.
    ordered = sorted(pairs, key=lambda p: str(p[0]))
    index_of = {p[0]: i for i, p in enumerate(ordered)}

    class_image_indices: Dict[str, List[int]] = {}
    for img_path, _lbl in ordered:
        for cname in set(per_image_classes[img_path]):
            class_image_indices.setdefault(cname, []).append(index_of[img_path])

    resolved_counts = {c: len(idxs) for c, idxs in class_image_indices.items()}

    counts_lower = {c.lower(): v for c, v in resolved_counts.items()}
    caps_lower = {c.lower(): v for c, v in (manual_caps or {}).items()}
    balance_pool = {c.strip().lower() for c in classes_in_balance} if classes_in_balance else None
    targets_lower = compute_class_targets(counts_lower, balance_mode, caps_lower, balance_pool)
    name_by_lower = {c.lower(): c for c in resolved_counts}
    targets = {name_by_lower[k]: v for k, v in targets_lower.items() if k in name_by_lower}

    keep_sets: Dict[str, set] = {}
    for cname, target in targets.items():
        if target is None:
            continue
        keep_sets[cname] = select_with_gap(sorted(class_image_indices[cname]), target, min_gap)

    kept: List[Tuple[Path, Path]] = []
    dropped: List[Tuple[Path, Path]] = []
    for img_path, lbl_path in ordered:
        idx = index_of[img_path]
        classes_here = set(per_image_classes[img_path])
        if not classes_here:
            kept.append((img_path, lbl_path))  # background image - not subject to class balancing
            continue
        keep = any(
            targets.get(cname) is None or idx in keep_sets.get(cname, set())
            for cname in classes_here
        )
        (kept if keep else dropped).append((img_path, lbl_path))

    if progress_callback:
        progress_callback(50, "Balance plan complete.")

    return {
        "cancelled": False,
        "source_dir": Path(source_dir).expanduser().resolve(),
        "global_names": stats["global_names"],
        "resolved_instance_counts": resolved_counts,
        "targets": targets,
        "kept": kept,
        "dropped": dropped,
        "total_images": len(ordered),
    }


# --------------------------------------------------------------------------- #
# Phase B: Execution
# --------------------------------------------------------------------------- #

def execute_balance_plan(
    plan: Dict[str, Any],
    output_dir: Path,
    copy_mode: str = "copy",  # "copy" | "move" | "symlink"
    progress_callback: Optional[Callable[[int, str], None]] = None,
    cancel_callback: Optional[Callable[[], bool]] = None,
):
    """Copy/move/symlink the kept image+label pairs (+ data.yaml) into output_dir, mirroring the source's subfolder layout."""
    output_dir = Path(output_dir).expanduser().resolve()
    source_dir = plan["source_dir"]
    kept = plan["kept"]
    total = len(kept)

    def _copy(src: str, dst: str):
        shutil.copy2(src, dst)

    def _move(src: str, dst: str):
        shutil.move(src, dst)

    def _symlink(src: str, dst: str):
        if os.path.lexists(dst):
            os.remove(dst)
        os.symlink(os.path.abspath(src), dst)

    op = {"copy": _copy, "move": _move, "symlink": _symlink}.get(copy_mode, _copy)

    for i, (img_path, lbl_path) in enumerate(kept):
        if cancel_callback and cancel_callback():
            yield 0, "Balance cancelled by user."
            return
        if progress_callback and i % 50 == 0:
            pct = int(50 + (i / total) * 45) if total else 95
            progress_callback(pct, f"Copying {i + 1}/{total}: {img_path.name}")
            yield pct, f"Copying {i + 1}/{total}: {img_path.name}"

        rel_img = img_path.relative_to(source_dir)
        rel_lbl = lbl_path.relative_to(source_dir)
        dest_img, dest_lbl = output_dir / rel_img, output_dir / rel_lbl
        dest_img.parent.mkdir(parents=True, exist_ok=True)
        dest_lbl.parent.mkdir(parents=True, exist_ok=True)
        op(str(img_path), str(dest_img))
        op(str(lbl_path), str(dest_lbl))

    src_yaml = source_dir / "data.yaml"
    global_names = plan.get("global_names") or []
    if src_yaml.is_file():
        shutil.copy2(str(src_yaml), str(output_dir / "data.yaml"))
    elif global_names and yaml:
        out_yaml = {
            "path": str(output_dir),
            "names": {i: n for i, n in enumerate(global_names)},
            "nc": len(global_names),
            "train": "images",
            "val": "images",
        }
        (output_dir / "data.yaml").write_text(yaml.safe_dump(out_yaml, sort_keys=False), encoding="utf-8")

    summary = (
        f"Balanced dataset written: {len(kept)} image(s) kept, {len(plan['dropped'])} dropped "
        f"out of {plan['total_images']} total."
    )
    if progress_callback:
        progress_callback(100, summary)
    yield 100, summary


def balance_image_dataset(
    source_dir: Path,
    output_dir: Path,
    yaml_path: Optional[Path] = None,
    balance_mode: str = "auto_min",
    manual_caps: Optional[Dict[str, int]] = None,
    classes_in_balance: Optional[List[str]] = None,
    min_gap: int = 0,
    copy_mode: str = "copy",
    progress_callback: Optional[Callable[[int, str], None]] = None,
    cancel_callback: Optional[Callable[[], bool]] = None,
):
    """Combined entry point: build_balance_plan() then execute_balance_plan(), back to back. Yields (progress_percent, status_message)."""
    plan = build_balance_plan(
        source_dir=source_dir, yaml_path=yaml_path, balance_mode=balance_mode,
        manual_caps=manual_caps, classes_in_balance=classes_in_balance, min_gap=min_gap,
        progress_callback=progress_callback, cancel_callback=cancel_callback,
    )
    if plan.get("cancelled"):
        yield 0, "Balance cancelled by user."
        return

    for pct, msg in execute_balance_plan(
        plan, output_dir=output_dir, copy_mode=copy_mode,
        progress_callback=progress_callback, cancel_callback=cancel_callback,
    ):
        yield pct, msg
