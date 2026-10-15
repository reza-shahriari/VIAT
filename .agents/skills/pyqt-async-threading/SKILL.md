---
name: pyqt-async-threading
description: >-
  Use this skill when implementing background tasks, long-running processing, thread workers,
  progress reporting, or asynchronous operations in VIAT to keep the PyQt5 UI responsive.
---

# PyQt Async Threading & Worker Pipelines

This skill guides the implementation of asynchronous background tasks in VIAT, ensuring the PyQt5 main event loop never freezes while maintaining strict thread safety.

---

## 1. The Core Cardinal Rules of Qt Threading

1. **GUI Affinity**:
   - **NEVER** instantiate, update, read, or manipulate any `QWidget`, `QPixmap`, `QPainter`, or dialog from a background thread.
   - Touching Qt GUI objects from outside the main thread will cause intermittent race conditions, graphical corruptions, or immediate hard crashes (`Fatal Python error: Aborted`).

2. **Inter-Thread Communication via Signals**:
   - All data transfer between worker threads and the main GUI must occur via `pyqtSignal`.
   - When a signal crosses thread boundaries, Qt automatically uses a `Qt.QueuedConnection`, placing the payload into the main thread's event queue safely.

3. **Exception Shielding**:
   - Any unhandled Python exception escaping a `QThread.run()` will terminate the thread abnormally or crash the process.
   - Always wrap worker logic in a broad `try ... except Exception:` block, format the traceback with `traceback.format_exc()`, and emit an error signal.

---

## 2. Standard Pattern 1: Generator Tasks via `task_runner.py`

For tasks that iterate over frames, files, or items, use VIAT's built-in [task_runner.py](file:///media/reza/New%20Volume/VIAT/viat/utils/task_runner.py).

### How to Write a Generator Task:
```python
def export_dataset_generator(dataset_path, output_path):
    """
    Generator yielding progress updates.
    Yields (progress_percent: int, status_message: str) or final result on return.
    """
    total = len(items)
    for idx, item in enumerate(items):
        # Heavy work here
        ...
        percent = int(((idx + 1) / total) * 100)
        yield (percent, f"Processing {item.name} ({idx+1}/{total})")
    
    return {"status": "success", "count": total}
```

### Launching with Progress Dialog:
```python
from viat.utils.task_runner import run_task_with_progress

run_task_with_progress(
    parent=self,
    title="Exporting Dataset",
    label_text="Preparing export...",
    generator_func=export_dataset_generator,
    dataset_path="/path/to/in",
    output_path="/path/to/out",
    maximum=100
)
```

---

## 3. Standard Pattern 2: Dedicated `QThread` Subclass

For complex stateful background pipelines (like [AutopilotWorker](file:///media/reza/New%20Volume/VIAT/viat/managers/autopilot_controller.py) or video inference engines), subclass `QThread` directly:

```python
import traceback
from PyQt5.QtCore import QThread, pyqtSignal

class InferenceWorker(QThread):
    progress = pyqtSignal(str, int, int)   # stage_name, current_idx, total
    done = pyqtSignal(object)              # payload / result
    failed = pyqtSignal(str)               # error traceback

    def __init__(self, model, frames, config, parent=None):
        super().__init__(parent)
        self.model = model
        self.frames = frames
        self.config = config
        self._cancel_requested = False

    def cancel(self):
        """Called from the main GUI thread to signal cooperative stop."""
        self._cancel_requested = True

    def run(self):
        try:
            results = []
            total = len(self.frames)
            for idx, frame in enumerate(self.frames):
                if self._cancel_requested:
                    self.failed.emit("Operation cancelled by user.")
                    return

                # Perform model inference / CPU-heavy processing
                detection = self.model.predict(frame)
                results.append(detection)

                # Report progress
                self.progress.emit("Running model", idx + 1, total)

            self.done.emit(results)

        except Exception as exc:
            err_msg = traceback.format_exc()
            self.failed.emit(err_msg)
```

### Thread Lifecycle in the Controller / Main Window:
```python
def start_processing(self):
    self.worker = InferenceWorker(self.model, self.frames, self.cfg, parent=self)
    
    # Wire signals
    self.worker.progress.connect(self.on_worker_progress)
    self.worker.done.connect(self.on_worker_done)
    self.worker.failed.connect(self.on_worker_failed)
    
    # Clean up thread reference after completion
    self.worker.finished.connect(self.worker.deleteLater)
    
    # Disable trigger actions to prevent concurrent runs
    self.start_button.setEnabled(False)
    self.worker.start()

def on_worker_done(self, results):
    self.start_button.setEnabled(True)
    self.apply_results_to_ui(results)

def on_worker_failed(self, error_str):
    self.start_button.setEnabled(True)
    QMessageBox.critical(self, "Error", f"Processing failed:\n{error_str}")
```

---

## 4. Cancellation & GPU Memory Safety

1. **Cooperative Checks**:
   Check `self._cancel_requested` or `is_cancelled()` inside the innermost iteration loop before starting a costly step.

2. **GPU Memory Cleanup**:
   If a worker holds CUDA tensors or TensorRT engines, ensure cleanup in a `finally` block or on cancellation:
   ```python
   finally:
       import torch
       if torch.cuda.is_available():
           torch.cuda.empty_cache()
   ```

3. **Window Close Protection**:
   In `closeEvent` of dialogs or `QMainWindow`, check if workers are still active:
   ```python
   def closeEvent(self, event):
       if hasattr(self, 'worker') and self.worker.isRunning():
           self.worker.cancel()
           self.worker.wait(2000)  # Wait up to 2s for graceful exit
       event.accept()
   ```
