---
name: pyqt-headless-testing
description: >-
  Use this skill when writing, running, or debugging tests for PyQt5 widgets, canvas interactions,
  dialogs, or UI components in headless or automated CI environments without a live display.
---

# PyQt Headless Testing & Verification

This skill provides procedures for verifying PyQt5 widgets, canvas interactions, and signal flows in headless environments (Linux servers, CI pipelines, containers) where no X11 or Wayland display server is running.

---

## 1. Headless Environment Setup

Running any Qt GUI code on a headless machine without configuration results in:
`qt.qpa.plugin: Could not load the Qt platform plugin "xcb" ... Aborted`

### Solution: Offscreen Platform Plugin
Always set the offscreen platform before creating or running Qt tests:

* **In Shell**:
  ```bash
  export QT_QPA_PLATFORM=offscreen
  ```

* **In Python Scripts**:
  ```python
  import os
  os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
  ```

---

## 2. Managing the `QApplication` Fixture

Only one `QApplication` instance can exist per process. Multiple instantiations will crash the Python interpreter.

### Safe Initialization Pattern:
```python
import sys
from PyQt5.QtWidgets import QApplication

def get_qapp():
    """Retrieve or lazily initialize the singleton QApplication."""
    app = QApplication.instance()
    if app is None:
        app = QApplication(["--platform", "offscreen"] + sys.argv[1:])
    return app
```

---

## 3. Simulating User Interactions with `QTest`

Use `PyQt5.QtTest.QTest` to simulate realistic user input on widgets and canvas without a mouse or monitor.

### Testing Canvas Mouse Operations:
```python
from PyQt5.QtTest import QTest
from PyQt5.QtCore import Qt, QPoint
from viat.canvas import VideoCanvas

app = get_qapp()
canvas = VideoCanvas()
canvas.resize(800, 600)
canvas.show()

# 1. Simulate mouse click to select
pos = QPoint(100, 100)
QTest.mouseClick(canvas, Qt.LeftButton, pos=pos)

# 2. Simulate drag-and-drop to draw a bounding box
start_pos = QPoint(50, 50)
end_pos = QPoint(200, 200)

QTest.mousePress(canvas, Qt.LeftButton, pos=start_pos)
QTest.mouseMove(canvas, pos=end_pos)
QTest.mouseRelease(canvas, Qt.LeftButton, pos=end_pos)

# Verify state change
assert len(canvas.annotations) > 0, "Annotation was not created after drag"
```

### Testing Keyboard Shortcuts:
```python
# Simulate pressing Ctrl+X (e.g. for scene change mark)
QTest.keyClick(canvas, Qt.Key_X, Qt.ControlModifier)
```

---

## 4. Signal Verification

To test that user actions or background workers trigger the expected UI events:

```python
def test_dock_signal_emission():
    app = get_qapp()
    dock = MyFeatureDock(None)
    
    received_signals = []
    dock.actionRequested.connect(lambda payload: received_signals.append(payload))
    
    # Trigger button click
    QTest.mouseClick(dock.btn_run, Qt.LeftButton)
    
    assert len(received_signals) == 1
    assert "param1" in received_signals[0]
```

---

## 5. Lightweight Syntax Smoke Checks

Before committing changes to VIAT, execute a syntax smoke check across the entire package:

```bash
python -m compileall viat
```

This catches Python syntax errors, mismatched brackets, and invalid imports across all 70+ files in seconds without launching the application.
