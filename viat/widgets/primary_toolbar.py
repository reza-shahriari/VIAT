"""Primary annotation toolbar widgets.

The toolbar in this module deliberately contains only the controls used most
often while annotating.  It emits intent signals and leaves all application
state changes to the caller.
"""

from collections.abc import Iterable

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QKeySequence
from PyQt5.QtWidgets import (
    QComboBox,
    QMenu,
    QShortcut,
    QToolBar,
    QToolButton,
    QWidget,
)

from viat.utils.icon_provider import IconProvider


class PrimaryAnnotationToolbar(QToolBar):
    """Compact toolbar for the primary annotation workflow.

    Parameters
    ----------
    main_window : object, optional
        An existing application window.  It is used only to discover class
        names for :meth:`update_class_selector`; no application methods are
        called by this widget.
    icon_provider : object, optional
        Object implementing the project's ``get_icon(name)`` convention.  A
        default :class:`~viat.utils.icon_provider.IconProvider` is created
        when one is not supplied.
    parent : QWidget, optional
        Parent widget for the toolbar.
    class_names : iterable of str, optional
        Initial class names.  When omitted, names are read from
        ``main_window.canvas.class_colors`` when available.
    """

    class_selected = pyqtSignal(str)
    method_selected = pyqtSignal(str)
    add_annotation_requested = pyqtSignal()
    edit_annotation_requested = pyqtSignal()
    delete_annotation_requested = pyqtSignal()
    pan_toggled = pyqtSignal(bool)
    zoom_in_requested = pyqtSignal()
    zoom_out_requested = pyqtSignal()
    zoom_reset_requested = pyqtSignal()
    add_class_requested = pyqtSignal()

    def __init__(
        self,
        main_window=None,
        icon_provider=None,
        parent=None,
        class_names=None,
    ):
        # The legacy toolbar accepted ``main_window`` as its first argument.
        # Use it as a Qt parent only when it really is a QWidget; test doubles
        # and lightweight callers can still pass an arbitrary object here.
        if parent is None and isinstance(main_window, QWidget):
            parent = main_window

        super().__init__("Primary Annotation Toolbar", parent)
        self.setObjectName("PrimaryAnnotationToolbar")

        self.main_window = main_window
        self.icon_provider = icon_provider or IconProvider()
        self._icon_widgets = {}

        self._create_selectors()
        self._create_annotation_buttons()
        self._create_navigation_buttons()

        self.pan_shortcut = QShortcut(QKeySequence("P"), self)
        self.pan_shortcut.setObjectName("PrimaryAnnotationPanShortcut")
        self.pan_shortcut.activated.connect(
            lambda: self.pan_button.setChecked(not self.pan_button.isChecked())
        )

        self.update_class_selector(class_names)
        self.refresh_icons()

    def _create_selectors(self):
        """Create the class and drawing-method selectors."""
        self.class_selector = QComboBox(self)
        self.class_selector.setObjectName("PrimaryAnnotationClassSelector")
        self.class_selector.setToolTip("Select annotation class")
        self.class_selector.setMinimumContentsLength(8)
        self.class_selector.setSizeAdjustPolicy(QComboBox.AdjustToContents)
        self.class_selector.currentTextChanged.connect(self._on_class_selected)
        self.addWidget(self.class_selector)

        # Keep exactly one drawing-method selector in the primary toolbar.
        self.method_selector = QComboBox(self)
        self.drawing_method_selector = self.method_selector
        self.method_selector.setObjectName("PrimaryAnnotationMethodSelector")
        self.method_selector.addItems(["Drag", "TwoClick"])
        self.method_selector.setToolTip(
            "Drawing method: Drag creates a box by dragging; "
            "TwoClick creates a box from two corners"
        )
        self.method_selector.currentTextChanged.connect(self._on_method_selected)
        self.addWidget(self.method_selector)

        self.addSeparator()

    def _create_annotation_buttons(self):
        """Create add, edit, and delete annotation controls."""
        self.add_annotation_button = self._make_icon_button(
            "PrimaryAnnotationAddButton",
            "add",
            "Add annotation",
            self._emit_add_annotation,
        )
        self.add_button = self.add_annotation_button

        self.edit_annotation_button = self._make_icon_button(
            "PrimaryAnnotationEditButton",
            "edit",
            "Edit selected annotation",
            self._emit_edit_annotation,
        )
        self.edit_button = self.edit_annotation_button

        self.delete_annotation_button = self._make_icon_button(
            "PrimaryAnnotationDeleteButton",
            "delete",
            "Delete selected annotation",
            self._emit_delete_annotation,
        )
        self.delete_button = self.delete_annotation_button

        self.addSeparator()

    def _create_navigation_buttons(self):
        """Create pan and zoom controls."""
        self.pan_button = self._make_icon_button(
            "PrimaryAnnotationPanButton",
            "pan-tool",
            "Toggle pan mode",
            self._emit_pan_toggled,
            checkable=True,
        )
        self.pan_tool_button = self.pan_button

        self.zoom_in_button = self._make_icon_button(
            "PrimaryAnnotationZoomInButton",
            "zoom-in",
            "Zoom in",
            self._emit_zoom_in,
        )
        self.zoom_out_button = self._make_icon_button(
            "PrimaryAnnotationZoomOutButton",
            "zoom-out",
            "Zoom out",
            self._emit_zoom_out,
        )
        self.zoom_reset_button = self._make_icon_button(
            "PrimaryAnnotationZoomResetButton",
            "zoom-original",
            "Reset zoom",
            self._emit_zoom_reset,
        )
        self.reset_zoom_button = self.zoom_reset_button

        self.addSeparator()

    def _create_more_button(self):
        """Create a retained compatibility menu without adding it to the bar.

        The annotation toolbar is intentionally a full, uncollapsed toolbar.
        ``more_button`` and ``more_menu`` remain available for older callers,
        but the application does not display a ``More`` overflow control.
        """
        self.more_button = QToolButton(self)
        self.more_button.setObjectName("PrimaryAnnotationMoreButton")
        self.more_button.setText("More")
        self.more_button.setToolTip("More annotation tools")
        self.more_button.setToolButtonStyle(Qt.ToolButtonTextOnly)
        self.more_button.setPopupMode(QToolButton.InstantPopup)

        self.more_menu = QMenu(self.more_button)
        self.more_menu.setObjectName("PrimaryAnnotationMoreMenu")
        self.more_button.setMenu(self.more_menu)

    def _make_icon_button(
        self,
        object_name,
        icon_name,
        tooltip,
        callback,
        checkable=False,
    ):
        """Create an icon-only tool button and add it to the toolbar."""
        button = QToolButton(self)
        button.setObjectName(object_name)
        button.setToolButtonStyle(Qt.ToolButtonIconOnly)
        button.setAutoRaise(True)
        button.setCheckable(checkable)
        button.setToolTip(tooltip)
        # Keep the icon name on the widget as well as in the private mapping;
        # this mirrors the action metadata used by existing VIAT toolbars.
        button.icon_name = icon_name
        self._icon_widgets[button] = icon_name
        if checkable:
            button.toggled.connect(callback)
        else:
            button.clicked.connect(callback)
        self.addWidget(button)
        return button

    def _on_class_selected(self, class_name):
        if class_name:
            if class_name == "Add New...":
                self.add_class_requested.emit()
            else:
                self.class_selected.emit(class_name)

    def _on_method_selected(self, method_name):
        if method_name:
            self.method_selected.emit(method_name)

    def _emit_add_annotation(self, _checked=False):
        self.add_annotation_requested.emit()

    def _emit_edit_annotation(self, _checked=False):
        self.edit_annotation_requested.emit()

    def _emit_delete_annotation(self, _checked=False):
        self.delete_annotation_requested.emit()

    def _emit_pan_toggled(self, checked):
        self.pan_toggled.emit(bool(checked))

    def _emit_zoom_in(self, _checked=False):
        self.zoom_in_requested.emit()

    def _emit_zoom_out(self, _checked=False):
        self.zoom_out_requested.emit()

    def _emit_zoom_reset(self, _checked=False):
        self.zoom_reset_requested.emit()

    def _class_names_from_main_window(self):
        """Return class names exposed by common VIAT window structures."""
        window = self.main_window
        if window is None:
            return []

        canvas = getattr(window, "canvas", None)
        class_colors = getattr(canvas, "class_colors", None)
        if class_colors is None:
            class_colors = getattr(window, "class_colors", None)
        if hasattr(class_colors, "keys"):
            return list(class_colors.keys())

        for attribute in ("class_names", "classes"):
            names = getattr(window, attribute, None)
            if names is None:
                continue
            if hasattr(names, "keys"):
                return list(names.keys())
            if isinstance(names, Iterable) and not isinstance(names, (str, bytes)):
                return list(names)
        return []

    def update_class_selector(self, class_names=None):
        """Refresh available class names while preserving the selection.

        ``class_names`` is optional for compatibility with existing VIAT
        callers.  If omitted, the names are discovered from the main window.
        Signals are blocked while rebuilding so a refresh is not mistaken for
        a user selection.
        """
        current_name = self.class_selector.currentText()
        names = (
            self._class_names_from_main_window()
            if class_names is None
            else list(class_names)
        )

        # Preserve order while removing duplicate entries and ignore null
        # values that QComboBox cannot represent meaningfully.
        unique_names = []
        for name in names:
            if name is None:
                continue
            name = str(name)
            if name and name not in unique_names:
                unique_names.append(name)

        canvas = getattr(self.main_window, "canvas", None)
        current_canvas_class = getattr(canvas, "current_class", None)

        self.class_selector.blockSignals(True)
        try:
            self.class_selector.clear()
            self.class_selector.addItems(unique_names)
            self.class_selector.addItem("Add New...")

            preferred_name = (
                current_name if current_name != "Add New..." else None
            ) or current_canvas_class
            if preferred_name:
                index = self.class_selector.findText(str(preferred_name))
                if index >= 0:
                    self.class_selector.setCurrentIndex(index)
        finally:
            self.class_selector.blockSignals(False)

    def refresh_icons(self):
        """Refresh toolbar icons through the shared ``IconProvider``."""
        get_icon = getattr(self.icon_provider, "get_icon", None)
        if not callable(get_icon):
            return
        for button, icon_name in self._icon_widgets.items():
            button.setIcon(get_icon(icon_name))
        for action, icon_name in getattr(self, "_icon_actions", {}).items():
            action.setIcon(get_icon(icon_name))

    def register_icon_action(self, action, icon_name):
        """Register an action whose icon should follow the active theme."""
        if not hasattr(self, "_icon_actions"):
            self._icon_actions = {}
        self._icon_actions[action] = icon_name
        get_icon = getattr(self.icon_provider, "get_icon", None)
        if callable(get_icon):
            action.setIcon(get_icon(icon_name))


__all__ = ["PrimaryAnnotationToolbar"]
