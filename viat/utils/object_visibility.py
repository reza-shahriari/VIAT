"""
Per-object visibility management for VIAT video annotations.

Redesigned (patch10):
  * NO dialog -- the main window injects toolbar buttons that replace the
    normal frame-control buttons while the mode is active.
  * Ranges loaded SEPARATELY -- each object may have multiple visible ranges
    (disjoint frame segments). The user works on ONE range at a time, not
    all at once.
  * Free frame navigation -- the user can still use the slider/arrows; the
    canvas just shows only the current object's annotations.

Toolbar buttons (shown while mode is active):
  [< Prev Object]  [Object: xxx (1/N)]  [Next Object >]
  [< Prev Range]   [Range: 0-15 (1/3)]  [Next Range >]
  [Set Start]  [Set End]  [Delete Frame]  [Delete Object]  [FINISH]
"""

from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple


class ObjectVisibilityManager:
    """Manages the per-object visibility editing mode (toolbar-based)."""

    def __init__(self, app):
        self.app = app
        self.active = False

        # {actor_id: sorted list of frame numbers where this actor appears}
        self.object_frames: Dict[str, List[int]] = {}

        # Current object being edited
        self.current_object: Optional[str] = None
        self.current_object_index: int = 0
        self.sorted_objects: List[str] = []

        # Current range index (each object may have multiple disjoint ranges)
        self.current_range_index: int = 0

        # Toolbar widgets (created by the main window, stored here for updates)
        self.toolbar_widgets: Dict = {}

    # ------------------------------------------------------------------ #
    # Mode entry / exit
    # ------------------------------------------------------------------ #

    def start(self):
        """Enter object-visibility mode."""
        self.active = True
        self._build_object_index()
        self.sorted_objects = self._sort_objects_by_time()

        if not self.sorted_objects:
            self.active = False
            return False

        self.current_object_index = 0
        self.current_object = self.sorted_objects[0]
        self.current_range_index = 0
        self._apply_filter()
        return True

    def exit(self):
        """Exit object-visibility mode, restoring normal display."""
        self.active = False
        self.current_object = None

        if hasattr(self.app, "canvas"):
            self.app.canvas.object_filter = None
            self.app.canvas.update()

    # ------------------------------------------------------------------ #
    # Object index
    # ------------------------------------------------------------------ #

    def _build_object_index(self):
        """Build {actor_id: [frame_nums]} from frame_annotations."""
        self.object_frames = defaultdict(list)
        for frame_num, anns in getattr(self.app, "frame_annotations", {}).items():
            for ann in anns:
                actor_id = self._get_actor_id(ann)
                if actor_id and frame_num not in self.object_frames[actor_id]:
                    self.object_frames[actor_id].append(frame_num)
        for actor_id in self.object_frames:
            self.object_frames[actor_id].sort()

    def _sort_objects_by_time(self) -> List[str]:
        """Sort objects by first-appearance frame (time order)."""
        if not self.object_frames:
            return []
        first_frames = []
        for actor_id, frames in self.object_frames.items():
            first_frame = min(frames) if frames else float('inf')
            first_frames.append((first_frame, actor_id))
        first_frames.sort(key=lambda x: (x[0], x[1]))
        return [actor_id for _, actor_id in first_frames]

    @staticmethod
    def _get_actor_id(ann) -> Optional[str]:
        attrs = getattr(ann, "attributes", {}) or {}
        return attrs.get("actor_id") or attrs.get("track_id")

    # ------------------------------------------------------------------ #
    # Ranges (disjoint frame segments)
    # ------------------------------------------------------------------ #

    def get_visible_ranges(self) -> List[Tuple[int, int]]:
        """Get the contiguous frame ranges for the current object."""
        frames = self.object_frames.get(self.current_object, [])
        if not frames:
            return []
        ranges = []
        start = prev = frames[0]
        for f in frames[1:]:
            if f == prev + 1:
                prev = f
            else:
                ranges.append((start, prev))
                start = prev = f
        ranges.append((start, prev))
        return ranges

    def get_current_range(self) -> Optional[Tuple[int, int]]:
        """Get the current range (start, end) being edited."""
        ranges = self.get_visible_ranges()
        if not ranges or self.current_range_index >= len(ranges):
            return None
        return ranges[self.current_range_index]

    def next_range(self) -> bool:
        """Move to the next visible range. Returns False if at the last."""
        ranges = self.get_visible_ranges()
        if self.current_range_index >= len(ranges) - 1:
            return False
        self.current_range_index += 1
        self._jump_to_range_start()
        return True

    def prev_range(self) -> bool:
        """Move to the previous visible range."""
        if self.current_range_index <= 0:
            return False
        self.current_range_index -= 1
        self._jump_to_range_start()
        return True

    def _jump_to_range_start(self):
        """Jump to the first frame of the current range."""
        r = self.get_current_range()
        if r:
            if hasattr(self.app, "seek_to_frame"):
                self.app.seek_to_frame(r[0])
            elif hasattr(self.app, "load_current_image"):
                self.app.current_frame = r[0]
                self.app.load_current_image()

    # ------------------------------------------------------------------ #
    # Filtering
    # ------------------------------------------------------------------ #

    def _apply_filter(self):
        """Filter the canvas to show only the current object."""
        if not self.current_object:
            return
        if hasattr(self.app, "canvas"):
            self.app.canvas.object_filter = self.current_object
            self.app.canvas.update()

        # Jump to the first frame of the first range
        self.current_range_index = 0
        self._jump_to_range_start()

    # ------------------------------------------------------------------ #
    # Object switching
    # ------------------------------------------------------------------ #

    def next_object(self) -> bool:
        """Move to the next object. Returns False if this was the last."""
        if self.current_object_index >= len(self.sorted_objects) - 1:
            return False
        self.current_object_index += 1
        self.current_object = self.sorted_objects[self.current_object_index]
        self.current_range_index = 0
        self._apply_filter()
        return True

    def prev_object(self) -> bool:
        """Move to the previous object."""
        if self.current_object_index <= 0:
            return False
        self.current_object_index -= 1
        self.current_object = self.sorted_objects[self.current_object_index]
        self.current_range_index = 0
        self._apply_filter()
        return True

    def select_object(self, actor_id: str) -> bool:
        """Select a specific object by actor_id."""
        if actor_id not in self.object_frames:
            return False
        self.current_object = actor_id
        self.current_object_index = self.sorted_objects.index(actor_id)
        self.current_range_index = 0
        self._apply_filter()
        return True

    # ------------------------------------------------------------------ #
    # Trimming (per-range)
    # ------------------------------------------------------------------ #

    def trim_current_frame_as_start(self):
        """Set the current frame as the start of the current range.

        Removes all annotations for the current object in frames BEFORE the
        current frame (within the current contiguous range).
        """
        cur = getattr(self.app, "current_frame", 0)
        r = self.get_current_range()
        if not r:
            return
        # Only trim if we're within the current range
        if not (r[0] <= cur <= r[1]):
            return
        removed = 0
        for frame_num in range(r[0], cur):
            removed += self._remove_object_from_frame(frame_num)
        self._rebuild_and_refresh()

    def trim_current_frame_as_end(self):
        """Set the current frame as the end of the current range."""
        cur = getattr(self.app, "current_frame", 0)
        r = self.get_current_range()
        if not r:
            return
        if not (r[0] <= cur <= r[1]):
            return
        removed = 0
        for frame_num in range(cur + 1, r[1] + 1):
            removed += self._remove_object_from_frame(frame_num)
        self._rebuild_and_refresh()

    def _remove_object_from_frame(self, frame_num: int) -> int:
        """Remove the current object's annotations from a specific frame."""
        frame_annotations = getattr(self.app, "frame_annotations", {})
        if frame_num not in frame_annotations:
            return 0
        anns = frame_annotations[frame_num]
        before = len(anns)
        frame_annotations[frame_num] = [
            a for a in anns if self._get_actor_id(a) != self.current_object
        ]
        removed = before - len(frame_annotations[frame_num])
        if not frame_annotations[frame_num]:
            del frame_annotations[frame_num]
        return removed

    def _rebuild_and_refresh(self):
        """Rebuild the object index and refresh the display."""
        self._build_object_index()
        self.sorted_objects = self._sort_objects_by_time()
        def remove_current_range(self) -> int:
            """Delete all annotations of the current object within the active range.
            Returns the number of removed annotations."""
            r = self.get_current_range()
            if not r:
                return 0
            start, end = r
            removed = 0
            for frame_num in range(start, end + 1):
                removed += self._remove_object_from_frame(frame_num)
            self._rebuild_and_refresh()
            return removed

        def get_visible_frame_numbers(self) -> list:
            """Return a list of frame numbers belonging to the current range.
            If no range is active, returns an empty list."""
            r = self.get_current_range()
            if not r:
                return []
            return list(range(r[0], r[1] + 1))

        # Clamp range index
        ranges = self.get_visible_ranges()
        if self.current_range_index >= len(ranges):
            self.current_range_index = max(0, len(ranges) - 1)
        # Refresh canvas
        if hasattr(self.app, "load_current_frame_annotations"):
            self.app.load_current_frame_annotations()
        elif hasattr(self.app, "canvas"):
            self.app.canvas.update()

    # ------------------------------------------------------------------ #
    # Delete
    # ------------------------------------------------------------------ #

    def delete_current_object_on_current_frame(self) -> int:
        """Remove the current object's annotation(s) from the current frame only."""
        cur = getattr(self.app, "current_frame", 0)
        removed = self._remove_object_from_frame(cur)
        self._rebuild_and_refresh()
        return removed

    def delete_object(self, actor_id: str = None) -> int:
        """Delete ALL annotations for an object across all frames."""
        target = actor_id or self.current_object
        if not target:
            return 0

        frame_annotations = getattr(self.app, "frame_annotations", {})
        removed = 0
        for frame_num in list(frame_annotations.keys()):
            anns = frame_annotations[frame_num]
            before = len(anns)
            frame_annotations[frame_num] = [
                a for a in anns if self._get_actor_id(a) != target
            ]
            removed += before - len(frame_annotations[frame_num])
            if not frame_annotations[frame_num]:
                del frame_annotations[frame_num]

        self._rebuild_and_refresh()
        if target == self.current_object:
            if self.sorted_objects:
                new_idx = min(self.current_object_index, len(self.sorted_objects) - 1)
                self.current_object_index = new_idx
                self.current_object = self.sorted_objects[new_idx]
                self._apply_filter()
            else:
                self.current_object = None
        return removed

    # ------------------------------------------------------------------ #
    # Status (for toolbar labels)
    # ------------------------------------------------------------------ #

    def get_status(self) -> Dict:
        """Return status info for the toolbar labels."""
        ranges = self.get_visible_ranges()
        r = self.get_current_range()
        return {
            "active": self.active,
            "current_object": self.current_object,
            "object_index": self.current_object_index,
            "total_objects": len(self.sorted_objects),
            "current_range": r,
            "range_index": self.current_range_index,
            "total_ranges": len(ranges),
            "current_frame": getattr(self.app, "current_frame", 0),
            "all_objects": self.sorted_objects,
        }

    def remove_current_range(self) -> int:
        """Delete all annotations of the current object within the active range.
        Returns the number of removed annotations."""
        r = self.get_current_range()
        if not r:
            return 0
        start, end = r
        removed = 0
        for frame_num in range(start, end + 1):
            removed += self._remove_object_from_frame(frame_num)
        self._rebuild_and_refresh()
        return removed

    def get_visible_frame_numbers(self) -> list:
        """Return a list of frame numbers belonging to the current range.
        If no range is active, returns an empty list."""
        r = self.get_current_range()
        if not r:
            return []
        return list(range(r[0], r[1] + 1))
