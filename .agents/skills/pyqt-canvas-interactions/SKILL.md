---
name: pyqt-canvas-interactions
description: >-
  Use this skill when modifying, debugging, or extending PyQt5 canvas widgets, viewport
  transformations, coordinate math, bounding box or polygon rendering, mouse/drag event
  handling, or painter performance in VIAT's VideoCanvas.
---

# PyQt Canvas Interactions & Coordinate Systems

This skill guides the implementation, modification, and debugging of [canvas.py](file:///media/reza/New%20Volume/VIAT/viat/canvas.py) and related modules ([canvas_edge_movement.py](file:///media/reza/New%20Volume/VIAT/viat/canvas_edge_movement.py), [smart_edge.py](file:///media/reza/New%20Volume/VIAT/viat/smart_edge.py)).

---

## 1. Coordinate Space Architecture

VIAT operates across three distinct coordinate spaces. Mixing these coordinates is the #1 source of canvas bugs:

```
[ Image Space ]  <--(zoom & pan)-->  [ Display Space ]  <--(widget bounds)-->  [ Screen/Qt Space ]
 Raw pixel coords                     Canvas widget coords                       Qt global event pos
 (0, 0) to (W, H)                     Account for aspect-ratio letterboxing      QCursor.pos()
```

### Conversion Reference

Always use the built-in transformation methods in `VideoCanvas`:

* **Point Transformations**:
  - `self.display_to_image_pos(pos: QPoint) -> QPoint`: Converts canvas widget coordinates to raw image pixel coordinates.
  - `self.image_to_display_pos(pos: QPoint) -> QPoint`: Converts raw image pixel coordinates to canvas widget coordinates.
  - `self.display_to_image_point(pos) -> (float, float)`: Floating-point sub-pixel version for precision operations.
  - `self.image_to_display_point(x, y) -> (float, float)`: Floating-point display projection.

* **Rectangle Transformations**:
  - `self.display_to_image_rect(rect: QRect) -> QRect`: Maps a drawn selection box to raw image bounding box.
  - `self.image_to_display_rect(rect: QRect) -> QRect`: Maps an annotation's raw bounding box to the current viewport for painting.

* **Display Rect Bounds**:
  - `self.get_display_rect() -> QRect`: Returns the centered letterboxed/pillarboxed area inside the canvas where the video frame is drawn. Always clamp mouse interactions against `get_display_rect()` unless panning.

---

## 2. Rendering & QPainter Best Practices

`VideoCanvas.paintEvent` executes frequently (during video playback, scrubbing, and drag operations). Keep draw cycles lean:

1. **Cosmetic Pens for Annotations**:
   Always set `pen.setCosmetic(True)` when drawing bounding boxes, polygon edges, and handles. This ensures line thickness remains constant on screen regardless of the current zoom level:
   ```python
   pen = QPen(color, line_width)
   pen.setCosmetic(True)
   painter.setPen(pen)
   ```

2. **Selective Redraws (Dirty Rects)**:
   - For high-frequency mouse movements (e.g., hover handle highlighting, crosshair updates), avoid calling `self.repaint()` or unconditional `self.update()`.
   - Call `self.update(dirty_rect)` with the bounding rectangle of the affected region (padded with handle size and line width).

3. **Sub-pixel Polygons**:
   - For segmentation masks and polygons, construct `QPainterPath` or `QPolygonF` in display coordinates before drawing to prevent raster jitter during zoom.

4. **Coordinate Clamping**:
   - When drawing or creating bounding boxes, clamp coordinates to the video image dimensions `[0, frame_width]` and `[0, frame_height]` to prevent out-of-bounds annotations in export files.

---

## 3. Hit-Testing & Handle Management

### Bounding Box Handles
Handles are indexed clockwise starting from top-left:
* `self.get_handle_points(rect: QRect) -> list[QPoint]`: Computes the 8 handle positions (corners + midpoints).
* `self.handle_at_pos(rect: QRect, pos: QPoint, handle_size: int = 8) -> int`: Returns handle index (0–7) or `-1` if none is hit.
* Use `QRect(pt.x() - size//2, pt.y() - size//2, size, size).contains(pos)` for hit detection.

### Edge Snapping & Detection
* `self.detect_edge(rect: QRect, pos: QPoint, threshold: int = 12, inner_threshold: int = 2) -> int`: Returns `EDGE_TOP`, `EDGE_RIGHT`, `EDGE_BOTTOM`, `EDGE_LEFT`, or `EDGE_NONE`.
* When dragging an edge, delegate to `self.move_edge(rect, edge, pos, start_pos)` to preserve minimum bbox dimensions (minimum width/height ≥ 4px).
* When Smart Edge snapping is active, invoke [smart_edge.py](file:///media/reza/New%20Volume/VIAT/viat/smart_edge.py) algorithms to snap edges to image gradient extrema.

---

## 4. Mouse Event Lifecycle

Follow this state machine when handling mouse interactions:

1. **`mousePressEvent(event)`**:
   - **Left Click**:
     - If pan mode enabled (`self.pan_mode`) or middle-click: record `self.drag_start_pos` for viewport panning.
     - Else if handle hit: set `self.resize_mode = handle_index`.
     - Else if edge hit: set `self.edge_mode = edge_type`.
     - Else if annotation hit: select annotation (`self.selected_bbox = ann`).
     - Else: initiate new annotation drawing (`self.drawing = True`).
   - **Right Click**: Open context menu (`self.show_context_menu(event.pos())`).

2. **`mouseMoveEvent(event)`**:
   - Check `self.setMouseTracking(True)` is enabled in `__init__`.
   - Update hover cursor (`self.get_edge_cursor(edge)` or resize arrows).
   - If drawing or resizing: recalculate active rectangle, clamp bounds, and schedule `self.update()`.

3. **`mouseReleaseEvent(event)`**:
   - Finalize active shape, convert from display to image space, and add to `self.annotations`.
   - Reset state flags (`self.drawing = False`, `self.resize_mode = None`, `self.drag_start_pos = None`).
   - Emit state change signals to notify the main window and annotation docks.
