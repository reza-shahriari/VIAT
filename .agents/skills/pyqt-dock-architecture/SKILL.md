---
name: pyqt-dock-architecture
description: >-
  Use this skill when creating, modifying, or refactoring PyQt5 dock widgets, dialogs,
  toolbars, or styling UI components in VIAT, following the decoupled manager architecture.
---

# PyQt Dock Architecture & Component Decoupling

This skill guides the creation and maintenance of UI docks ([viat/widgets/](file:///media/reza/New%20Volume/VIAT/viat/widgets/)), dialogs, and styling in VIAT.

---

## 1. The Core Architecture Rule

> **From [AGENTS.md](file:///media/reza/New%20Volume/VIAT/AGENTS.md):**
> `viat/main.py` is the `VideoAnnotationTool(QMainWindow)` coordinator. Keep feature logic in `viat/utils/` or `viat/managers/` and wire thin calls from `main.py` rather than expanding the coordinator.

* **Do NOT** put domain logic, file reading/writing, or model execution inside dock widgets or `main.py`.
* **DO** encapsulate business logic in a manager class (e.g., `viat/managers/my_feature_manager.py`) or a utility module (`viat/utils/my_feature.py`).
* Docks emit signals declaring user intent; the coordinator (`main.py`) or controller connects those signals to manager calls.

---

## 2. Standard Dock Widget Anatomy

All docks in VIAT subclass `QDockWidget`:

```python
from PyQt5.QtWidgets import (
    QDockWidget, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QGroupBox
)
from PyQt5.QtCore import Qt, pyqtSignal

class MyFeatureDock(QDockWidget):
    """
    Modular dock widget.
    Emits signals when user triggers actions rather than reaching into main_window internals.
    """
    # Define custom signals for user actions
    actionRequested = pyqtSignal(dict)
    settingChanged = pyqtSignal(str, object)

    def __init__(self, main_window):
        super().__init__("Feature Title", main_window)
        self.main_window = main_window
        self.setObjectName("MyFeatureDock")  # Required for QMainWindow.saveState()
        
        # Configure allowed dock zones
        self.setAllowedAreas(Qt.LeftDockWidgetArea | Qt.RightDockWidgetArea)
        
        self.init_ui()

    def init_ui(self):
        container = QWidget(self)
        layout = QVBoxLayout(container)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        # Controls
        group = QGroupBox("Configuration")
        group_layout = QVBoxLayout()
        ...
        group.setLayout(group_layout)
        layout.addWidget(group)

        # Action Buttons
        self.btn_run = QPushButton("Execute")
        self.btn_run.clicked.connect(self._on_execute_clicked)
        layout.addWidget(self.btn_run)

        layout.addStretch()
        self.setWidget(container)

    def _on_execute_clicked(self):
        # Package UI values into payload and emit
        payload = {"param1": 123}
        self.actionRequested.emit(payload)
```

---

## 3. Thin Wiring in `main.py`

When integrating a new dock into [main.py](file:///media/reza/New%20Volume/VIAT/viat/main.py):

1. **Instantiate and Add Dock**:
   ```python
   # Inside VideoAnnotationTool.init_ui() or setup_docks()
   self.my_feature_dock = MyFeatureDock(self)
   self.addDockWidget(Qt.RightDockWidgetArea, self.my_feature_dock)
   ```

2. **Add View Menu Toggle Action**:
   Ensure users can hide/show the dock from the application menu:
   ```python
   self.view_menu.addAction(self.my_feature_dock.toggleViewAction())
   ```

3. **Wire Thin Signals to Managers**:
   ```python
   # Thin connection: dock signal -> manager execution
   self.my_feature_dock.actionRequested.connect(self.my_feature_manager.run)
   ```

---

## 4. Theming, Styles & Icons

VIAT supports dynamic theme switching (Fusion, Dark, Light) via [styles.py](file:///media/reza/New%20Volume/VIAT/viat/widgets/styles.py) and `styles.qss`.

1. **Consistent Styling**:
   - Avoid hardcoding arbitrary hex colors directly on widgets unless drawing on canvas.
   - Use standard palette colors (`QPalette.Window`, `QPalette.Highlight`, `QPalette.Text`) or class styles from `styles.py`.

2. **Icons via `IconProvider`**:
   - Always load application icons through [icon_provider.py](file:///media/reza/New%20Volume/VIAT/viat/utils/icon_provider.py) instead of raw `QIcon("path")` calls:
   ```python
   from viat.utils.icon_provider import IconProvider
   
   my_button.setIcon(IconProvider.get_icon("play"))
   ```
   This ensures fallback resolution and proper SVG rendering across different operating systems.

---

## 5. Dock State & Geometry Persistence

VIAT saves layout configurations between sessions:
* Always provide a unique `setObjectName(...)` for every dock widget.
* Do not change object names after release, as it breaks restored user window geometries (`self.restoreState(settings.value("windowState"))`).
