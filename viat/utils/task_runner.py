from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtWidgets import QProgressDialog, QMessageBox
import traceback

class WorkerThread(QThread):
    """
    A generic worker thread that runs a generator function.
    The generator should yield either:
    - (progress_value, status_message)  -> int, str
    - A single value (e.g., status_message) -> str
    
    If the task completes successfully, it emits finished(result).
    If it errors, it emits error(exception_string).
    """
    progressChanged = pyqtSignal(int, str)
    finished_with_result = pyqtSignal(object)
    error = pyqtSignal(str)

    def __init__(self, generator_func, *args, **kwargs):
        super().__init__()
        self.generator_func = generator_func
        self.args = args
        self.kwargs = kwargs
        self.is_cancelled = False

    def run(self):
        try:
            generator = self.generator_func(*self.args, **self.kwargs)
            result = None
            
            # Since generator functions don't return values directly, if the
            # function has a `return value` at the end, it raises StopIteration(value).
            while True:
                if self.is_cancelled:
                    break
                try:
                    item = next(generator)
                    if isinstance(item, tuple) and len(item) == 2:
                        val, msg = item
                        self.progressChanged.emit(val, str(msg))
                    else:
                        self.progressChanged.emit(0, str(item))
                        
                    result = item # Last yielded item might be the result as fallback
                except StopIteration as e:
                    if e.value is not None:
                        result = e.value
                    break
            
        except Exception as e:
            err_msg = traceback.format_exc()
            self.error.emit(str(err_msg))
            return

        self.finished_with_result.emit(result)

    def cancel(self):
        self.is_cancelled = True


def run_task_with_progress(parent, title, label_text, generator_func, *args, maximum=100, **kwargs):
    """
    Runs a generator function in a background thread and shows a progress dialog.
    """
    progress_dialog = QProgressDialog(label_text, "Cancel", 0, maximum, parent)
    progress_dialog.setWindowTitle(title)
    progress_dialog.setWindowModality(Qt.WindowModal)
    progress_dialog.setMinimumDuration(0)
    progress_dialog.setValue(0)

    worker = WorkerThread(generator_func, *args, **kwargs)
    
    # We use a mutable container to capture the result
    result_container = {"result": None, "error": None}

    def on_progress(val, msg):
        progress_dialog.setValue(val)
        progress_dialog.setLabelText(msg)

    def on_finished(res):
        result_container["result"] = res
        progress_dialog.accept()

    def on_error(err):
        result_container["error"] = err
        progress_dialog.reject()
        
    def on_cancel():
        worker.cancel()

    worker.progressChanged.connect(on_progress)
    worker.finished_with_result.connect(on_finished)
    worker.error.connect(on_error)
    progress_dialog.canceled.connect(on_cancel)

    worker.start()
    progress_dialog.exec_()
    
    if result_container["error"]:
        QMessageBox.critical(parent, "Error", f"An error occurred during {title}:\n{result_container['error']}")
        
    return result_container["result"]

class AutoSaveThread(QThread):
    """
    A background thread for saving project files without freezing the UI.
    """
    finished_autosave = pyqtSignal(bool, str)

    def __init__(self, filename, project_data_args, parent=None):
        super().__init__(parent)
        self.filename = filename
        self.project_data_args = project_data_args

    def run(self):
        try:
            from viat.utils.file_operations import save_project
            save_project(self.filename, **self.project_data_args)
            self.finished_autosave.emit(True, self.filename)
        except Exception as e:
            err_msg = traceback.format_exc()
            self.finished_autosave.emit(False, str(e))
