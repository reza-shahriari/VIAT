"""
Per-object visibility management for VIAT video annotations.

Problem: objects come in and out of the scene. The user needs to manage, per
object, which frame ranges it's actually visible in -- removing labels before
a chosen start frame and after a chosen end frame. Each object may appear
multiple times (multiple disjoint visible ranges).

UX (per the user's spec):
  * The user should NOT see other objects -- only the CURRENT object's labels
    are visible on the canvas. Other objects are hidden.
  * Only frames related to the current object are navigable (the frame list
    is filtered to frames where this object appears).
  * The user sets a start frame and end frame for the visible range, trimming
    labels outside that range.
  * The user can delete ALL labels for the object (across every frame).
  * After modification, pressing FINISH moves to the next object automatically.

This module provides the ObjectVisibilityManager class that drives the mode.
The main window wires it to a dialog / dock + keyboard shortcuts.
"""

from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple


class ObjectVisibilityManager:
    """Manages the per-object visibility editing mode.

    Lifecycle:
      1. start() -- enters the mode. Builds the object index from
         frame_annotations. Hides all annotations except the first object.
      2. For the current object:
         - The canvas shows ONLY this object's annotations.
         - Navigation is restricted to frames where this object appears.
         - User trims start/end, deletes individual annotations, or deletes
           the whole object.
      3. finish_object() -- applies changes, moves to the next object.
      4. exit() -- restores normal display (all annotations visible).
    """

    def __init__(self, app):
        self.app = app
        self.active = False

        # {actor_id: sorted list of frame numbers where this actor appears}
        self.object_frames: Dict[str, List[int]] = {}

        # Current object being edited
        self.current_object: Optional[str] = None
        self.current_object_index: int = 0  # index into sorted object list
        self.sorted_objects: List[str] = []

        # Navigation: filtered frame list for the current object
        self.filtered_frames: List[int] = []

        # The frame the user was on before entering the mode (to restore on exit)
        self.saved_frame: int = 0

        # Saved state for restoring on exit
        self._saved_canvas_filter = None

    # ------------------------------------------------------------------ #
    # Mode entry / exit
    # ------------------------------------------------------------------ #

    def start(self):
        """Enter object-visibility mode."""
        self.active = True
        self.saved_frame = getattr(self.app, "current_frame", 0)
        self._build_object_index()
        self.sorted_objects = sorted(self.object_frames.keys())

        if not self.sorted_objects:
            self.active = False
            return False

        self.current_object_index = 0
        self.current_object = self.sorted_objects[0]
        self._apply_filter()
        return True

    def exit(self):
        """Exit object-visibility mode, restoring normal display."""
        self.active = False
        self.current_object = None

        # Restore canvas filter
        if hasattr(self.app, "canvas"):
            self.app.canvas.object_filter = None
            self.app.canvas.update()

        # Restore frame navigation
        if hasattr(self.app, "_viat_filtered_frames"):
            del self.app._viat_filtered_frames

        # Go back to the saved frame
        if hasattr(self.app, "seek_to_frame"):
            self.app.seek_to_frame(self.saved_frame)
        elif hasattr(self.app, "load_current_image"):
            self.app.current_frame = self.saved_frame
            self.app.load_current_image()

    # ------------------------------------------------------------------ #
    # Object index
    # ------------------------------------------------------------------ #

    def _build_object_index(self):
        """Build {actor_id: [frame_nums]} from frame_annotations.

        Uses the 'actor_id' attribute (set by the viat_json format). Falls
        back to 'track_id' if actor_id is absent.
        """
        self.object_frames = defaultdict(list)
        for frame_num, anns in getattr(self.app, "frame_annotations", {}).items():
            for ann in anns:
                actor_id = self._get_actor_id(ann)
                if actor_id and frame_num not in self.object_frames[actor_id]:
                    self.object_frames[actor_id].append(frame_num)
        # Sort frames
        for actor_id in self.object_frames:
            self.object_frames[actor_id].sort()

    @staticmethod
    def _get_actor_id(ann) -> Optional[str]:
        """Get the actor ID from an annotation's attributes."""
        attrs = getattr(ann, "attributes", {}) or {}
        return attrs.get("actor_id") or attrs.get("track_id")

    # ------------------------------------------------------------------ #
    # Filtering
    # ------------------------------------------------------------------ #

    def _apply_filter(self):
        """Filter the canvas to show only the current object, and restrict
        navigation to frames where this object appears."""
        if not self.current_object:
            return

        # Set the canvas filter so only this object's annotations are drawn
        if hasattr(self.app, "canvas"):
            self.app.canvas.object_filter = self.current_object
            self.app.canvas.update()

        # Set the filtered frame list for navigation
        self.filtered_frames = list(self.object_frames.get(self.current_object, []))
        self.app._viat_filtered_frames = self.filtered_frames

        # Jump to the first frame of this object
        if self.filtered_frames:
            target = self.filtered_frames[0]
            if hasattr(self.app, "seek_to_frame"):
                self.app.seek_to_frame(target)
            elif hasattr(self.app, "load_current_image"):
                self.app.current_frame = target
                self.app.load_current_image()

    # ------------------------------------------------------------------ #
    # Navigation (within filtered frames)
    # ------------------------------------------------------------------ #

    def next_object_frame(self) -> bool:
        """Go to the next frame where the current object appears.

        Returns True if moved, False if at the last frame.
        """
        if not self.filtered_frames:
            return False
        cur = getattr(self.app, "current_frame", 0)
        # Find the next frame in filtered_frames > cur
        for f in self.filtered_frames:
            if f > cur:
                if hasattr(self.app, "seek_to_frame"):
                    self.app.seek_to_frame(f)
                return True
        return False

    def prev_object_frame(self) -> bool:
        """Go to the previous frame where the current object appears."""
        if not self.filtered_frames:
            return False
        cur = getattr(self.app, "current_frame", 0)
        for f in reversed(self.filtered_frames):
            if f < cur:
                if hasattr(self.app, "seek_to_frame"):
                    self.app.seek_to_frame(f)
                return True
        return False

    # ------------------------------------------------------------------ #
    # Object switching
    # ------------------------------------------------------------------ #

    def next_object(self) -> bool:
        """Move to the next object. Returns False if this was the last."""
        if self.current_object_index >= len(self.sorted_objects) - 1:
            return False
        self.current_object_index += 1
        self.current_object = self.sorted_objects[self.current_object_index]
        self._apply_filter()
        return True

    def prev_object(self) -> bool:
        """Move to the previous object."""
        if self.current_object_index <= 0:
            return False
        self.current_object_index -= 1
        self.current_object = self.sorted_objects[self.current_object_index]
        self._apply_filter()
        return True

    def select_object(self, actor_id: str) -> bool:
        """Select a specific object by actor_id."""
        if actor_id not in self.object_frames:
            return False
        self.current_object = actor_id
        self.current_object_index = self.sorted_objects.index(actor_id)
        self._apply_filter()
        return True

    # ------------------------------------------------------------------ #
    # Visible range trimming
    # ------------------------------------------------------------------ #

    def get_visible_ranges(self) -> List[Tuple[int, int]]:
        """Get the contiguous frame ranges where the current object appears.

        Returns a list of (start_frame, end_frame) tuples.
        """
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

    def trim_range(self, range_index: int, new_start: int, new_end: int) -> int:
        """Trim a visible range to [new_start, new_end].

        Removes all annotations for the current object in frames that are
        within the original range but outside [new_start, new_end].

        Args:
            range_index: which contiguous range (from get_visible_ranges).
            new_start: new start frame (inclusive).
            new_end: new end frame (inclusive).

        Returns:
            Number of frames whose annotations were removed.
        """
        ranges = self.get_visible_ranges()
        if range_index < 0 or range_index >= len(ranges):
            return 0

        orig_start, orig_end = ranges[range_index]
        removed = 0

        # Remove annotations for frames in [orig_start, new_start) and
        # (new_end, orig_end] for THIS object only.
        frame_annotations = getattr(self.app, "frame_annotations", {})
        for frame_num in range(orig_start, new_start):
            removed += self._remove_object_from_frame(frame_num)
        for frame_num in range(new_end + 1, orig_end + 1):
            removed += self._remove_object_from_frame(frame_num)

        # Rebuild index for this object
        self._build_object_index()
        self._apply_filter()
        return removed

    def trim_current_frame_as_start(self):
        """Set the current frame as the start of the visible range.

        Removes all annotations for the current object in frames BEFORE the
        current frame (within the current contiguous range).
        """
        cur = getattr(self.app, "current_frame", 0)
        ranges = self.get_visible_ranges()
        for i, (s, e) in enumerate(ranges):
            if s <= cur <= e:
                self.trim_range(i, cur, e)
                return
        # If not in any range, find the nearest range before cur
        for i, (s, e) in enumerate(ranges):
            if e < cur:
                # remove everything in this range (object is gone before cur)
                self.trim_range(i, s, s - 1)  # removes all

    def trim_current_frame_as_end(self):
        """Set the current frame as the end of the visible range.

        Removes all annotations for the current object in frames AFTER the
        current frame (within the current contiguous range).
        """
        cur = getattr(self.app, "current_frame", 0)
        ranges = self.get_visible_ranges()
        for i, (s, e) in enumerate(ranges):
            if s <= cur <= e:
                self.trim_range(i, s, cur)
                return

    def _remove_object_from_frame(self, frame_num: int) -> int:
        """Remove all annotations for the current object from a specific frame.

        Returns the number of annotations removed.
        """
        frame_annotations = getattr(self.app, "frame_annotations", {})
        if frame_num not in frame_annotations:
            return 0
        anns = frame_annotations[frame_num]
        before = len(anns)
        frame_annotations[frame_num] = [
            a for a in anns if self._get_actor_id(a) != self.current_object
        ]
        after = len(frame_annotations[frame_num])
        removed = before - after
        # If frame is now empty, we can leave it (empty list) or delete the key
        if not frame_annotations[frame_num]:
            del frame_annotations[frame_num]
        return removed

    # ------------------------------------------------------------------ #
    # Delete whole object
    # ------------------------------------------------------------------ #

    def delete_object(self, actor_id: str = None) -> int:
        """Delete ALL annotations for an object across all frames.

        Args:
            actor_id: which object to delete (defaults to current_object).

        Returns:
            Number of annotations removed.
        """
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

        # Rebuild index and move to next object
        self._build_object_index()
        self.sorted_objects = sorted(self.object_frames.keys())
        if target in self.sorted_objects:
            # shouldn't happen, but guard
            pass
        if self.current_object == target:
            if self.sorted_objects:
                # move to next available
                new_idx = min(self.current_object_index, len(self.sorted_objects) - 1)
                self.current_object_index = new_idx
                self.current_object = self.sorted_objects[new_idx]
                self._apply_filter()
            else:
                self.current_object = None
        return removed

    # ------------------------------------------------------------------ #
    # Delete single annotation on current frame
    # ------------------------------------------------------------------ #

    def delete_current_object_on_current_frame(self) -> int:
        """Remove the current object's annotation(s) from the current frame only."""
        cur = getattr(self.app, "current_frame", 0)
        removed = self._remove_object_from_frame(cur)
        self._build_object_index()
        # refresh display
        if hasattr(self.app, "load_current_frame_annotations"):
            self.app.load_current_frame_annotations()
        elif hasattr(self.app, "canvas"):
            self.app.canvas.update()
        return removed

    # ------------------------------------------------------------------ #
    # Status info (for the UI)
    # ------------------------------------------------------------------ #

    def get_status(self) -> Dict:
        """Return status info for the UI dialog."""
        return {
            "active": self.active,
            "current_object": self.current_object,
            "object_index": self.current_object_index,
            "total_objects": len(self.sorted_objects),
            "current_object_frames": self.object_frames.get(self.current_object, []),
            "visible_ranges": self.get_visible_ranges(),
            "current_frame": getattr(self.app, "current_frame", 0),
            "all_objects": self.sorted_objects,
        }
