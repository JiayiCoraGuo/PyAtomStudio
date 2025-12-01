#AmpMap_Widgets
import sys
import os

from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QWidget, QLineEdit, QPushButton, QComboBox, QSlider, QVBoxLayout, QGridLayout,QGroupBox,QToolButton, QAction, QFileDialog
from PyQt5.QtGui import QIcon, QFont, QColor
from PyQt5.QtCore import Qt,QSize
from PyQt5.QtWidgets import QMessageBox, QCheckBox, QSpacerItem, QSizePolicy, QDialog, QTreeWidget, QTreeWidgetItem, QHeaderView




import qtawesome as qta


from QupyRibbon.GUI.RibbonWidget import RibbonWidget
from QupyRibbon.GUI.RibbonButton import RibbonButton
from QupyRibbon.GUI.Icons import get_icon
from QupyRibbon.GUI.RibbonTextbox import RibbonTextbox
from QupyRibbon.GUI.RibbonWidget import *

import numpy as np
from numpy import fft as npf
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5 import NavigationToolbar2QT as NavigationToolbar


import AmpMap as am
import tools

from matplotlib.patches import Circle
from matplotlib.backend_bases import MouseButton


class DraggableCircle:
    """Interactive circle that can be dragged and resized on matplotlib axes with bounding box."""

    def __init__(self, ax, center, radius, color='yellow', alpha=0.3):
        self.ax = ax
        self.circle = Circle(center, radius, color=color, alpha=alpha, fill=False, linewidth=2)
        self.ax.add_patch(self.circle)

        self.press = None
        self.background = None
        self.selected = False
        self.dragging_handle = None

        # Bounding box and handles
        self.bbox = None
        self.handles = []  # Corner handles for resizing
        self.handle_size = 5  # Size of corner handles

        self.connect()

    def connect(self):
        """Connect to matplotlib events."""
        self.cidpress = self.circle.figure.canvas.mpl_connect(
            'button_press_event', self.on_press)
        self.cidrelease = self.circle.figure.canvas.mpl_connect(
            'button_release_event', self.on_release)
        self.cidmotion = self.circle.figure.canvas.mpl_connect(
            'motion_notify_event', self.on_motion)
        self.cidscroll = self.circle.figure.canvas.mpl_connect(
            'scroll_event', self.on_scroll)

    def create_bounding_box(self):
        """Create bounding box and corner handles."""
        from matplotlib.patches import Rectangle

        cx, cy = self.circle.center
        r = self.circle.radius

        # Remove old bbox and handles if they exist
        if self.bbox:
            self.bbox.remove()
        for handle in self.handles:
            handle.remove()
        self.handles = []

        # Create bounding box (square around circle)
        bbox_size = r * 2
        self.bbox = Rectangle((cx - r, cy - r), bbox_size, bbox_size,
                              fill=False, edgecolor='cyan', linewidth=1, linestyle='--')
        self.ax.add_patch(self.bbox)

        # Create corner handles
        handle_positions = [
            (cx - r, cy - r),  # Bottom-left
            (cx + r, cy - r),  # Bottom-right
            (cx - r, cy + r),  # Top-left
            (cx + r, cy + r),  # Top-right
        ]

        for pos in handle_positions:
            handle = Rectangle((pos[0] - self.handle_size/2, pos[1] - self.handle_size/2),
                             self.handle_size, self.handle_size,
                             fill=True, facecolor='cyan', edgecolor='white', linewidth=1)
            self.ax.add_patch(handle)
            self.handles.append(handle)

    def remove_bounding_box(self):
        """Remove bounding box and handles."""
        if self.bbox:
            self.bbox.remove()
            self.bbox = None
        for handle in self.handles:
            handle.remove()
        self.handles = []

    def on_press(self, event):
        """Handle mouse press."""
        if event.inaxes != self.ax:
            return
        if event.button != MouseButton.LEFT:
            return

        # Check if clicking on a handle
        for i, handle in enumerate(self.handles):
            contains, _ = handle.contains(event)
            if contains:
                self.dragging_handle = i
                self.press = self.circle.center, (event.xdata, event.ydata)
                return

        # Check if clicking on circle
        contains, attrd = self.circle.contains(event)
        if contains:
            self.press = self.circle.center, (event.xdata, event.ydata)
            self.selected = True
            self.create_bounding_box()
            self.circle.figure.canvas.draw()
            return

        # Clicking outside - deselect
        if self.selected:
            self.selected = False
            self.remove_bounding_box()
            self.circle.figure.canvas.draw()

    def on_release(self, event):
        """Handle mouse release."""
        self.press = None
        self.dragging_handle = None
        self.circle.figure.canvas.draw()

    def on_motion(self, event):
        """Handle mouse motion (dragging)."""
        if self.press is None or event.inaxes != self.ax:
            return

        center, (xpress, ypress) = self.press
        dx = event.xdata - xpress
        dy = event.ydata - ypress

        if self.dragging_handle is not None:
            # Dragging a corner handle - resize circle
            # Calculate new radius based on distance from center to mouse
            new_radius = np.sqrt((event.xdata - center[0])**2 + (event.ydata - center[1])**2)
            new_radius = max(5, min(100, new_radius))
            self.circle.radius = new_radius
            self.create_bounding_box()
        else:
            # Dragging circle - move it
            self.circle.center = (center[0] + dx, center[1] + dy)
            if self.selected:
                self.create_bounding_box()

        self.circle.figure.canvas.draw()

    def on_scroll(self, event):
        """Handle scroll (resizing)."""
        if event.inaxes != self.ax:
            return

        contains, attrd = self.circle.contains(event)
        if not contains:
            return

        # Increase or decrease radius
        scale_factor = 1.1 if event.button == 'up' else 0.9
        new_radius = self.circle.radius * scale_factor

        # Set minimum and maximum radius
        new_radius = max(5, min(100, new_radius))
        self.circle.radius = new_radius

        if self.selected:
            self.create_bounding_box()

        self.circle.figure.canvas.draw()

    def get_center(self):
        """Get circle center coordinates."""
        return self.circle.center

    def get_radius(self):
        """Get circle radius."""
        return self.circle.radius

    def remove(self):
        """Remove circle, bounding box, and handles from axes."""
        self.remove_bounding_box()
        self.circle.remove()

    def disconnect(self):
        """Disconnect all callbacks."""
        self.circle.figure.canvas.mpl_disconnect(self.cidpress)
        self.circle.figure.canvas.mpl_disconnect(self.cidrelease)
        self.circle.figure.canvas.mpl_disconnect(self.cidmotion)
        self.circle.figure.canvas.mpl_disconnect(self.cidscroll)


class MetadataDialog(QDialog):
    """Dialog to display image metadata in a tree view."""

    def __init__(self, metadata_dict, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Image Metadata")
        self.setMinimumSize(600, 400)

        # Create layout
        layout = QVBoxLayout(self)

        # Create tree widget
        self.tree = QTreeWidget()
        self.tree.setHeaderLabels(["key / index", "value"])
        self.tree.header().setSectionResizeMode(0, QHeaderView.Interactive)
        self.tree.header().setSectionResizeMode(1, QHeaderView.Stretch)
        self.tree.setAlternatingRowColors(True)

        # Populate tree with metadata
        self.populate_tree(metadata_dict, self.tree.invisibleRootItem())

        # Expand all items by default
        self.tree.expandAll()

        # Adjust column widths
        self.tree.setColumnWidth(0, 300)

        layout.addWidget(self.tree)

    def populate_tree(self, data, parent_item):
        """Recursively populate tree widget with metadata."""
        if isinstance(data, dict):
            for key, value in data.items():
                if isinstance(value, (dict, list)):
                    # Create parent item with length info
                    if isinstance(value, dict):
                        item = QTreeWidgetItem(parent_item, [str(key), f"length={len(value)}"])
                    elif isinstance(value, list):
                        item = QTreeWidgetItem(parent_item, [str(key), f"length={len(value)}"])
                    self.populate_tree(value, item)
                else:
                    # Leaf node
                    item = QTreeWidgetItem(parent_item, [str(key), str(value)])
        elif isinstance(data, list):
            for idx, value in enumerate(data):
                if isinstance(value, (dict, list)):
                    item = QTreeWidgetItem(parent_item, [str(idx), f"length={len(value) if hasattr(value, '__len__') else '...'}"])
                    self.populate_tree(value, item)
                else:
                    item = QTreeWidgetItem(parent_item, [str(idx), str(value)])

class ToolsDialog(QDialog):
    """Dialog for image processing tools."""

    def __init__(self, image_data, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Image Processing Tools")
        self.setMinimumSize(500, 400)
        self.image_data = image_data
        self.processed_image = None

        # Main layout
        main_layout = QVBoxLayout(self)

        # Category selection
        category_group = QGroupBox("Tool Category")
        category_layout = QVBoxLayout()

        self.category_combo = QComboBox()
        self.category_combo.addItems(["Filters", "Corrections", "Transformations"])
        self.category_combo.currentTextChanged.connect(self.update_tool_list)
        category_layout.addWidget(self.category_combo)

        category_group.setLayout(category_layout)
        main_layout.addWidget(category_group)

        # Tool selection
        tool_group = QGroupBox("Select Tool")
        tool_layout = QVBoxLayout()

        self.tool_combo = QComboBox()
        self.tool_combo.currentTextChanged.connect(self.update_parameters)
        tool_layout.addWidget(self.tool_combo)

        tool_group.setLayout(tool_layout)
        main_layout.addWidget(tool_group)

        # Parameters
        self.params_group = QGroupBox("Parameters")
        self.params_layout = QGridLayout()
        self.params_group.setLayout(self.params_layout)
        main_layout.addWidget(self.params_group)

        # Buttons
        button_layout = QVBoxLayout()

        apply_button = QPushButton("Apply")
        apply_button.clicked.connect(self.apply_tool)
        button_layout.addWidget(apply_button)

        close_button = QPushButton("Close")
        close_button.clicked.connect(self.accept)
        button_layout.addWidget(close_button)

        main_layout.addLayout(button_layout)

        # Initialize tool list
        self.param_widgets = {}
        self.update_tool_list()

    def update_tool_list(self):
        """Update tool list based on selected category."""
        self.tool_combo.clear()
        category = self.category_combo.currentText()

        if category == "Filters":
            tools_list = [
                "Wiener Filter",
                "ABSF Filter",
                "Non-Linear Filter",
                "Butterworth Lowpass",
                "Gaussian Lowpass",
                "Median Filter",
                "Band-pass Filter",
                "Gradient Filter"
            ]
        elif category == "Corrections":
            tools_list = [
                "Line-by-Line Leveling",
                "Polynomial Background Subtraction",
                "Drift Correction",
                "Despeckle",
                "Contrast Enhancement (CLAHE)"
            ]
        else:  # Transformations
            tools_list = [
                "Crop",
                "Rotate",
                "Flip Horizontal",
                "Flip Vertical"
            ]

        self.tool_combo.addItems(tools_list)

    def update_parameters(self):
        """Update parameter inputs based on selected tool."""
        # Clear previous widgets
        for i in reversed(range(self.params_layout.count())):
            widget = self.params_layout.itemAt(i).widget()
            if widget:
                widget.deleteLater()

        self.param_widgets = {}
        tool = self.tool_combo.currentText()

        # Define parameters for each tool
        if tool == "Wiener Filter":
            self.add_param("Noise Variance", 0.001, 0.0, 1.0, 0.001)
        elif tool == "ABSF Filter":
            self.add_param("Sigma", 1.0, 0.1, 10.0, 0.1)
        elif tool == "Non-Linear Filter":
            self.add_param("Patch Size", 7, 3, 15, 2)
            self.add_param("Patch Distance", 11, 5, 20, 1)
            self.add_param("h (smoothing)", 0.1, 0.01, 1.0, 0.01)
        elif tool == "Butterworth Lowpass":
            self.add_param("Cutoff Frequency", 0.2, 0.01, 0.5, 0.01)
            self.add_param("Order", 2, 1, 10, 1)
        elif tool == "Gaussian Lowpass":
            self.add_param("Sigma", 2.0, 0.1, 10.0, 0.1)
        elif tool == "Median Filter":
            self.add_param("Size", 3, 3, 15, 2)
        elif tool == "Band-pass Filter":
            self.add_param("Low Frequency", 0.05, 0.01, 0.5, 0.01)
            self.add_param("High Frequency", 0.3, 0.01, 0.5, 0.01)
        elif tool == "Gradient Filter":
            self.add_param_combo("Direction", ["both", "x", "y"])
        elif tool == "Line-by-Line Leveling":
            self.add_param("Polynomial Order", 1, 1, 5, 1)
        elif tool == "Polynomial Background Subtraction":
            self.add_param("Polynomial Order", 2, 1, 5, 1)
        elif tool == "Drift Correction":
            self.add_param("Reference Row", 0, 0, self.image_data.shape[0]-1, 1)
        elif tool == "Despeckle":
            self.add_param("Threshold (std)", 3.0, 1.0, 10.0, 0.1)
        elif tool == "Contrast Enhancement (CLAHE)":
            self.add_param("Clip Limit", 0.03, 0.01, 0.1, 0.01)
            self.add_param("Tile Size", 8, 4, 32, 4)
        elif tool == "Crop":
            self.add_param("X Start", 0, 0, self.image_data.shape[1]-1, 1)
            self.add_param("X End", self.image_data.shape[1], 1, self.image_data.shape[1], 1)
            self.add_param("Y Start", 0, 0, self.image_data.shape[0]-1, 1)
            self.add_param("Y End", self.image_data.shape[0], 1, self.image_data.shape[0], 1)
        elif tool == "Rotate":
            self.add_param("Angle (degrees)", 0, -180, 180, 1)

    def add_param(self, name, default, min_val, max_val, step):
        """Add a parameter input widget."""
        row = len(self.param_widgets)
        label = QLabel(name + ":")
        self.params_layout.addWidget(label, row, 0)

        spinbox = QLineEdit()
        spinbox.setText(str(default))
        self.params_layout.addWidget(spinbox, row, 1)

        self.param_widgets[name] = {"widget": spinbox, "default": default}

    def add_param_combo(self, name, options):
        """Add a combo box parameter."""
        row = len(self.param_widgets)
        label = QLabel(name + ":")
        self.params_layout.addWidget(label, row, 0)

        combo = QComboBox()
        combo.addItems(options)
        self.params_layout.addWidget(combo, row, 1)

        self.param_widgets[name] = {"widget": combo, "default": options[0]}

    def apply_tool(self):
        """Apply the selected tool with parameters."""
        tool = self.tool_combo.currentText()

        try:
            # Get parameters
            params = {}
            for name, info in self.param_widgets.items():
                widget = info["widget"]
                if isinstance(widget, QComboBox):
                    params[name] = widget.currentText()
                else:
                    params[name] = float(widget.text())

            # Apply tool
            if tool == "Wiener Filter":
                self.processed_image = tools.apply_wiener_filter(
                    self.image_data, noise_variance=params.get("Noise Variance")
                )
            elif tool == "ABSF Filter":
                self.processed_image = tools.apply_absf_filter(
                    self.image_data, sigma=params.get("Sigma")
                )
            elif tool == "Non-Linear Filter":
                self.processed_image = tools.apply_nonlinear_filter(
                    self.image_data,
                    patch_size=int(params.get("Patch Size")),
                    patch_distance=int(params.get("Patch Distance")),
                    h=params.get("h (smoothing)")
                )
            elif tool == "Butterworth Lowpass":
                self.processed_image = tools.apply_butterworth_lowpass(
                    self.image_data,
                    cutoff_freq=params.get("Cutoff Frequency"),
                    order=int(params.get("Order"))
                )
            elif tool == "Gaussian Lowpass":
                self.processed_image = tools.apply_gaussian_lowpass(
                    self.image_data, sigma=params.get("Sigma")
                )
            elif tool == "Median Filter":
                self.processed_image = tools.apply_median_filter(
                    self.image_data, size=int(params.get("Size"))
                )
            elif tool == "Band-pass Filter":
                self.processed_image = tools.apply_bandpass_filter(
                    self.image_data,
                    low_freq=params.get("Low Frequency"),
                    high_freq=params.get("High Frequency")
                )
            elif tool == "Gradient Filter":
                self.processed_image = tools.gradient_filter(
                    self.image_data, direction=params.get("Direction")
                )
            elif tool == "Line-by-Line Leveling":
                self.processed_image = tools.line_by_line_leveling(
                    self.image_data, order=int(params.get("Polynomial Order"))
                )
            elif tool == "Polynomial Background Subtraction":
                self.processed_image = tools.polynomial_background_subtraction(
                    self.image_data, order=int(params.get("Polynomial Order"))
                )
            elif tool == "Drift Correction":
                self.processed_image = tools.drift_correction(
                    self.image_data, reference_row=int(params.get("Reference Row"))
                )
            elif tool == "Despeckle":
                self.processed_image = tools.despeckle(
                    self.image_data, threshold=params.get("Threshold (std)")
                )
            elif tool == "Contrast Enhancement (CLAHE)":
                self.processed_image = tools.enhance_contrast_clahe(
                    self.image_data,
                    clip_limit=params.get("Clip Limit"),
                    tile_size=int(params.get("Tile Size"))
                )
            elif tool == "Crop":
                self.processed_image = tools.crop_image(
                    self.image_data,
                    x_start=int(params.get("X Start")),
                    x_end=int(params.get("X End")),
                    y_start=int(params.get("Y Start")),
                    y_end=int(params.get("Y End"))
                )
            elif tool == "Rotate":
                self.processed_image = tools.rotate_image(
                    self.image_data, angle=params.get("Angle (degrees)")
                )
            elif tool == "Flip Horizontal":
                self.processed_image = tools.flip_image(
                    self.image_data, direction='horizontal'
                )
            elif tool == "Flip Vertical":
                self.processed_image = tools.flip_image(
                    self.image_data, direction='vertical'
                )

            self.accept()

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to apply tool:\n{str(e)}")


class FFTToolsDialog(QDialog):
    """Dialog for FFT-specific processing tools."""

    def __init__(self, original_image, fft_data, parent=None):
        super().__init__(parent)
        self.setWindowTitle("FFT Processing Tools")
        self.setMinimumSize(500, 450)
        self.original_image = original_image  # Real-space image
        self.fft_data = fft_data  # Current FFT data (complex, fftshifted)
        self.processed_fft = None

        # Main layout
        main_layout = QVBoxLayout(self)

        # FFT Type selection
        fft_type_group = QGroupBox("FFT Type")
        fft_type_layout = QVBoxLayout()

        self.fft_type_combo = QComboBox()
        self.fft_type_combo.addItems(["Standard FFT", "Window FFT"])
        self.fft_type_combo.currentTextChanged.connect(self.update_fft_params)
        fft_type_layout.addWidget(self.fft_type_combo)

        fft_type_group.setLayout(fft_type_layout)
        main_layout.addWidget(fft_type_group)

        # Window parameters (for Window FFT)
        self.window_group = QGroupBox("Window Parameters")
        window_layout = QGridLayout()

        window_layout.addWidget(QLabel("Window Type:"), 0, 0)
        self.window_type_combo = QComboBox()
        self.window_type_combo.addItems(["hanning", "hamming", "blackman", "bartlett"])
        window_layout.addWidget(self.window_type_combo, 0, 1)

        self.window_group.setLayout(window_layout)
        self.window_group.setVisible(False)
        main_layout.addWidget(self.window_group)

        # Processing operations
        operation_group = QGroupBox("Processing Operations")
        operation_layout = QVBoxLayout()

        self.operation_combo = QComboBox()
        self.operation_combo.addItems([
            "None",
            "Gaussian Mask",
            "Low Pass Filter",
            "High Pass Filter",
            "Change Origin"
        ])
        self.operation_combo.currentTextChanged.connect(self.update_operation_params)
        operation_layout.addWidget(self.operation_combo)

        operation_group.setLayout(operation_layout)
        main_layout.addWidget(operation_group)

        # Operation parameters
        self.params_group = QGroupBox("Operation Parameters")
        self.params_layout = QGridLayout()
        self.params_group.setLayout(self.params_layout)
        main_layout.addWidget(self.params_group)

        # Buttons
        button_layout = QHBoxLayout()

        apply_button = QPushButton("Apply")
        apply_button.clicked.connect(self.apply_tool)
        button_layout.addWidget(apply_button)

        close_button = QPushButton("Close")
        close_button.clicked.connect(self.reject)
        button_layout.addWidget(close_button)

        main_layout.addLayout(button_layout)

        # Initialize UI
        self.update_fft_params()
        self.update_operation_params()

    def update_fft_params(self):
        """Show/hide window parameters based on FFT type."""
        fft_type = self.fft_type_combo.currentText()
        self.window_group.setVisible(fft_type == "Window FFT")

    def update_operation_params(self):
        """Update parameter inputs based on selected operation."""
        # Clear previous parameters
        while self.params_layout.count():
            item = self.params_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        operation = self.operation_combo.currentText()

        if operation == "Gaussian Mask":
            self.params_layout.addWidget(QLabel("Center X:"), 0, 0)
            self.center_x_input = QLineEdit()
            self.center_x_input.setPlaceholderText(f"{self.fft_data.shape[1] // 2}")
            self.params_layout.addWidget(self.center_x_input, 0, 1)

            self.params_layout.addWidget(QLabel("Center Y:"), 1, 0)
            self.center_y_input = QLineEdit()
            self.center_y_input.setPlaceholderText(f"{self.fft_data.shape[0] // 2}")
            self.params_layout.addWidget(self.center_y_input, 1, 1)

            self.params_layout.addWidget(QLabel("Sigma:"), 2, 0)
            self.sigma_input = QLineEdit("20.0")
            self.params_layout.addWidget(self.sigma_input, 2, 1)

            self.params_layout.addWidget(QLabel("Inverse:"), 3, 0)
            self.inverse_check = QCheckBox()
            self.params_layout.addWidget(self.inverse_check, 3, 1)

        elif operation in ["Low Pass Filter", "High Pass Filter"]:
            self.params_layout.addWidget(QLabel("Center X:"), 0, 0)
            self.center_x_input = QLineEdit()
            self.center_x_input.setPlaceholderText(f"{self.fft_data.shape[1] // 2}")
            self.params_layout.addWidget(self.center_x_input, 0, 1)

            self.params_layout.addWidget(QLabel("Center Y:"), 1, 0)
            self.center_y_input = QLineEdit()
            self.center_y_input.setPlaceholderText(f"{self.fft_data.shape[0] // 2}")
            self.params_layout.addWidget(self.center_y_input, 1, 1)

            self.params_layout.addWidget(QLabel("Radius:"), 2, 0)
            self.radius_input = QLineEdit("30.0")
            self.params_layout.addWidget(self.radius_input, 2, 1)

        elif operation == "Change Origin":
            self.params_layout.addWidget(QLabel("New Origin X:"), 0, 0)
            self.origin_x_input = QLineEdit()
            self.origin_x_input.setPlaceholderText(f"{self.fft_data.shape[1] // 2}")
            self.params_layout.addWidget(self.origin_x_input, 0, 1)

            self.params_layout.addWidget(QLabel("New Origin Y:"), 1, 0)
            self.origin_y_input = QLineEdit()
            self.origin_y_input.setPlaceholderText(f"{self.fft_data.shape[0] // 2}")
            self.params_layout.addWidget(self.origin_y_input, 1, 1)

    def apply_tool(self):
        """Apply the selected FFT processing tool."""
        try:
            fft_type = self.fft_type_combo.currentText()
            operation = self.operation_combo.currentText()

            # Step 1: Compute FFT (standard or windowed)
            if fft_type == "Window FFT":
                window_type = self.window_type_combo.currentText()
                fft_result = tools.apply_window_fft(self.original_image, window_type)
            else:
                # Standard FFT (subtract mean before FFT, following PyAtoms approach)
                fft_result = np.fft.fftshift(np.fft.fft2(self.original_image - np.mean(self.original_image)))

            # Step 2: Apply operation to FFT
            if operation == "Gaussian Mask":
                center_x = int(self.center_x_input.text() or self.center_x_input.placeholderText())
                center_y = int(self.center_y_input.text() or self.center_y_input.placeholderText())
                sigma = float(self.sigma_input.text())
                inverse = self.inverse_check.isChecked()

                self.processed_fft = tools.apply_gaussian_mask_fft(
                    fft_result, center_x, center_y, sigma, inverse
                )

            elif operation == "Low Pass Filter":
                center_x = int(self.center_x_input.text() or self.center_x_input.placeholderText())
                center_y = int(self.center_y_input.text() or self.center_y_input.placeholderText())
                radius = float(self.radius_input.text())

                self.processed_fft = tools.apply_lowpass_filter_fft(
                    fft_result, center_x, center_y, radius
                )

            elif operation == "High Pass Filter":
                center_x = int(self.center_x_input.text() or self.center_x_input.placeholderText())
                center_y = int(self.center_y_input.text() or self.center_y_input.placeholderText())
                radius = float(self.radius_input.text())

                self.processed_fft = tools.apply_highpass_filter_fft(
                    fft_result, center_x, center_y, radius
                )

            elif operation == "Change Origin":
                origin_x = int(self.origin_x_input.text() or self.origin_x_input.placeholderText())
                origin_y = int(self.origin_y_input.text() or self.origin_y_input.placeholderText())

                self.processed_fft = tools.change_fft_origin(
                    fft_result, origin_x, origin_y
                )

            else:  # "None"
                self.processed_fft = fft_result

            self.accept()

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to apply FFT tool:\n{str(e)}")


class GPASettingsDialog(QDialog):
    """Dialog for GPA settings (Standard vs Adaptive)."""

    def __init__(self, current_algorithm='standard', current_sigma=5.0, current_auto_center=True, current_refine_peaks=False, parent=None):
        super().__init__(parent)
        self.setWindowTitle("GPA Settings")
        self.setMinimumSize(400, 350)

        self.algorithm = current_algorithm
        self.sigma = current_sigma
        self.auto_center = current_auto_center
        self.refine_peaks = current_refine_peaks

        # Main layout
        main_layout = QVBoxLayout(self)

        # Algorithm selection
        algorithm_group = QGroupBox("GPA Algorithm")
        algorithm_layout = QVBoxLayout()

        self.standard_radio = QRadioButton("Standard GPA")
        self.adaptive_radio = QRadioButton("Adaptive GPA")

        if current_algorithm == 'standard':
            self.standard_radio.setChecked(True)
        else:
            self.adaptive_radio.setChecked(True)

        algorithm_layout.addWidget(self.standard_radio)
        algorithm_layout.addWidget(self.adaptive_radio)
        algorithm_group.setLayout(algorithm_layout)
        main_layout.addWidget(algorithm_group)

        # Parameters
        params_group = QGroupBox("Parameters")
        params_layout = QGridLayout()

        params_layout.addWidget(QLabel("Smoothing Sigma:"), 0, 0)
        self.sigma_input = QLineEdit(str(current_sigma))
        params_layout.addWidget(self.sigma_input, 0, 1)

        params_group.setLayout(params_layout)
        main_layout.addWidget(params_group)

        # Peak selection options
        peak_group = QGroupBox("Peak Selection")
        peak_layout = QVBoxLayout()

        self.auto_center_checkbox = QCheckBox("Auto-center on highest intensity within circle")
        self.auto_center_checkbox.setChecked(current_auto_center)
        self.auto_center_checkbox.setToolTip("Automatically move circle center to the pixel with highest intensity within the circle")
        peak_layout.addWidget(self.auto_center_checkbox)

        self.refine_peaks_checkbox = QCheckBox("Refine peaks using plane fit (slope method)")
        self.refine_peaks_checkbox.setChecked(current_refine_peaks)
        self.refine_peaks_checkbox.setToolTip("Refine peak positions to 4 decimal places by minimizing phase slope (makes phase flat)")
        peak_layout.addWidget(self.refine_peaks_checkbox)

        peak_group.setLayout(peak_layout)
        main_layout.addWidget(peak_group)

        # Buttons
        button_layout = QHBoxLayout()

        ok_button = QPushButton("OK")
        ok_button.clicked.connect(self.accept_settings)
        button_layout.addWidget(ok_button)

        cancel_button = QPushButton("Cancel")
        cancel_button.clicked.connect(self.reject)
        button_layout.addWidget(cancel_button)

        main_layout.addLayout(button_layout)

    def accept_settings(self):
        """Accept and save settings."""
        try:
            self.algorithm = 'standard' if self.standard_radio.isChecked() else 'adaptive'
            self.sigma = float(self.sigma_input.text())
            self.auto_center = self.auto_center_checkbox.isChecked()
            self.refine_peaks = self.refine_peaks_checkbox.isChecked()
            self.accept()
        except ValueError:
            QMessageBox.warning(self, "Invalid Input", "Please enter a valid number for sigma.")


class myWidget(QWidget):
    def __init__(self, parent):
        super().__init__(parent)
        
        self.attribute_visible = True
        self.attribute_visibility_states = {}
        self.Input, self.FFT, self.Ideal, self.AmpMask, self.DisFld, self.PhaseFld = [], [], [], [], [], []
        self.PhaseA, self.PhaseB, self.StrainXX, self.StrainYY = [], [], [], []
        self._text_box1 = RibbonTextbox("", lambda: None, 80)
        self._text_box2 = RibbonTextbox("", lambda: None, 80)
        self._text_box3 = RibbonTextbox("", lambda: None, 80)
        self._text_box4 = RibbonTextbox("",  lambda: None, 80)
        self._text_box1.setPlaceholderText("133,111")
        self._text_box2.setPlaceholderText("145,123")
        self._text_box3.setPlaceholderText("204,52")
        self._text_box4.setPlaceholderText("5")
        for tb in (self._text_box1, self._text_box2,
            self._text_box3, self._text_box4):
            tb.setFixedWidth(60)


        self._text_box1.returnPressed.connect(self.on_k1_enter)
        self._text_box2.returnPressed.connect(self.on_k2_enter)
        self._text_box3.returnPressed.connect(self.on_r0_enter)
        self._text_box4.returnPressed.connect(self.on_sigma_enter)
        self.current_cmap = "plasma"
        self.all_figures  = []
        # default
        self.v1x, self.v1y = 133.0, 111.0
        self.v2x, self.v2y = 145.0, 123.0
        self.x0,  self.y0  = 204.0,  52.0
        self.sigma= 5.0
        # in __init__
        self.fft_peak_scatter = None

        # Image processing history for undo/redo
        self.image_history = {}  # Dict mapping figure_name to list of image states
        self.history_index = {}  # Dict mapping figure_name to current index in history

        # Store original unfiltered image separately
        self.Z_original = None

        # FFT display options
        self.fft_log_scale = False  # Default to linear scale

        # GPA (Geometric Phase Analysis) options
        self.bragg_peak_circles = []  # List of DraggableCircle objects
        self.gpa_algorithm = 'standard'  # 'standard' or 'adaptive'
        self.gpa_sigma = 5.0
        self.gpa_mask_radius = 20.0
        self.gpa_auto_center = True  # Auto-center circles on highest intensity
        self.gpa_refine_peaks = False  # Refine peaks using plane fit (slope method)

        # Strain and rotation data
        self.strain_xx = None
        self.strain_yy = None
        self.strain_xy = None
        self.rotation_map = None

        # Phase data
        self.phase_a = None
        self.phase_b = None



    
    def init_ribbon(self,ribbon):
        # File Tab
        file_tab = ribbon.add_ribbon_tab("Home")
        file_pane = file_tab.add_ribbon_pane("Files")
        new_action = QAction(qta.icon("fa5s.upload"), "Upload Your STM Image", self)
        new_action.triggered.connect(self.upload_image)  # Connect the action
        save_action = QAction(qta.icon("fa5s.download"), "Save Images", self)
        save_action.triggered.connect(self.download_file)  # Connect the action

        file_pane.add_ribbon_widget(self.create_tool_button(new_action))
        file_pane.add_ribbon_widget(self.create_tool_button(save_action))

        edit_pane = file_tab.add_ribbon_pane("Edit")
        grid = edit_pane.add_grid_widget(700)
        
        grid.addWidget(QLabel("Set Up"), 0, 0)
        lbl_k1  = QLabel("FFT Peak k1: ")
        lbl_k2  = QLabel("FFT Peak k2: ")
        lbl_r0  = QLabel("Reference vector r0: ")
        lbl_sig = QLabel("Gaussian Parameter \u03C3: ")

        grid.addWidget(lbl_k1,  1, 1)
        grid.addWidget(lbl_k2,  1, 3)
        grid.addWidget(lbl_r0,  1, 5)
        grid.addWidget(lbl_sig, 1, 7)
        grid.addWidget(self._text_box1, 1, 2)
        grid.addWidget(self._text_box2, 1, 4)
        grid.addWidget(self._text_box3, 1, 6)
        grid.addWidget(self._text_box4, 1, 8)

        

        view_pane = file_tab.add_ribbon_pane("View")
        bar_action = QAction(qta.icon("fa5s.bars"), "Show/Hide Attributes", self)
        bar_action.triggered.connect(self.toggle_all_attributes)

        colorBar_action = QAction(qta.icon("fa5s.tint"), "Colorbar", self)
        colorBar_action.triggered.connect(self.toggle_cmap_selector)
        show_action = QAction(qta.icon("fa5s.eye"), "Show", self)
        

        view_pane.add_ribbon_widget(self.create_tool_button(bar_action))
        view_pane.add_ribbon_widget(self._build_colorbar_widget())
        view_pane.add_ribbon_widget(self.create_tool_button(show_action))
        v_grid = view_pane.add_grid_widget(700)

        self.cmap_selector = QComboBox()
        cmap_choices = [
            'magma', 'magma_r','viridis','viridis_r','plasma','plasma_r','inferno',
            'inferno_r','Greys','Greys_r','cividis','cividis_r','bone','bone_r',
            'Blues','Blues_r','Purples','Purples_r','Greens','Greens_r','Oranges',
            'Oranges_r','Reds','Reds_r','YlOrBr','YlOrBr_r','YlOrRd','YlOrRd_r',
            'OrRd','OrRd_r','PuRd','PuRd_r','RdPu','RdPu_r','BuPu','BuPu_r','GnBu',
            'GnBu_r','PuBu','PuBu_r','YlGnBu','YlGnBu_r','PuBuGn','PuBuGn_r','BuGn',
            'BuGn_r','YlGn','YlGn_r','pink','pink_r','hot','hot_r','afmhot',
            'afmhot_r','gist_heat','gist_heat_r','PiYG','PiYG_r','PRGn','PRGn_r',
            'BrBG','BrBG_r','PuOr','PuOr_r','RdGy','RdGy_r','RdBu','RdBu_r',
            'RdYlBu','RdYlBu_r','RdYlGn','RdYlGn_r','Spectral','Spectral_r',
            'coolwarm','coolwarm_r','bwr','bwr_r','seismic','seismic_r','twilight',
            'twilight_r','twilight_shifted','twilight_shifted_r','hsv','hsv_r',
            'ocean','ocean_r','gist_earth','gist_earth_r','terrain','terrain_r',
            'gnuplot2','gnuplot2_r','CMRmap','CMRmap_r','gist_rainbow',
            'gist_rainbow_r','rainbow','rainbow_r','jet','jet_r'
        ]
        self.cmap_selector.addItems(cmap_choices)
        self.cmap_selector.setCurrentText(self.current_cmap)
        self.cmap_selector.currentTextChanged.connect(self.change_colormap)
        self.cmap_selector.setVisible(False)
        v_grid.addWidget(self.cmap_selector, 0, 8, 2, 2)




        spacer = QSpacerItem(20, 20, QSizePolicy.Minimum, QSizePolicy.Expanding)

        checkbox1 = QCheckBox("Title")
        checkbox1.setChecked(False)
        v_grid.addWidget(checkbox1, 0, 3)
        checkbox2 = QCheckBox("Legend")
        checkbox2.setChecked(False)
        v_grid.addWidget(checkbox2, 1, 3)
        checkbox3 = QCheckBox("x-Axis")
        checkbox3.setChecked(False)
        v_grid.addWidget(checkbox3, 0, 4)
        checkbox4 = QCheckBox("y-Axis")
        checkbox4.setChecked(False)
        v_grid.addWidget(checkbox4, 1, 4)
        checkbox5 = QCheckBox("x_Label")
        checkbox5.setChecked(False)
        v_grid.addWidget(checkbox5, 0, 5)
        checkbox6 = QCheckBox("y-Label")
        checkbox6.setChecked(False)
        v_grid.addWidget(checkbox6, 1, 5)
        v_grid.addItem(spacer, 0, 6)
        v_grid.addItem(spacer, 0, 7)




        # Help Tab
        help_tab = ribbon.add_ribbon_tab("Help")
        help_pane = help_tab.add_ribbon_pane("Help")
        about_action = QAction(qta.icon("fa5s.info"), "About", self)
        about_action.triggered.connect(self.show_about)
        help_pane.add_ribbon_widget(self.create_tool_button(about_action))

        shortcuts_action = QAction(qta.icon("fa5s.share"), "Shortcuts", self)
        shortcuts_action.triggered.connect(self.shortcuts)
        help_pane.add_ribbon_widget(self.create_tool_button(shortcuts_action))

    def _build_colorbar_widget(self):
        """ colorbar helper"""
        widget = QWidget()
        lay = QVBoxLayout(widget)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(2)

        # Button (uses the same QAction so tool-tips & icons stay consistent)
        action = QAction(qta.icon("fa5s.tint"), "Colorbar", self)
        btn = QToolButton()
        btn.setDefaultAction(action)
        btn.setToolButtonStyle(Qt.ToolButtonTextUnderIcon)
        btn.setIconSize(QSize(22, 22))

        # Combobox with all cmaps
        combo = QComboBox()
        cmap_choices = [
            'magma', 'magma_r','viridis','viridis_r','plasma','plasma_r','inferno',
            'inferno_r','Greys','Greys_r','cividis','cividis_r','bone','bone_r',
            'Blues','Blues_r','Purples','Purples_r','Greens','Greens_r','Oranges',
            'Oranges_r','Reds','Reds_r','YlOrBr','YlOrBr_r','YlOrRd','YlOrRd_r',
            'OrRd','OrRd_r','PuRd','PuRd_r','RdPu','RdPu_r','BuPu','BuPu_r','GnBu',
            'GnBu_r','PuBu','PuBu_r','YlGnBu','YlGnBu_r','PuBuGn','PuBuGn_r','BuGn',
            'BuGn_r','YlGn','YlGn_r','pink','pink_r','hot','hot_r','afmhot',
            'afmhot_r','gist_heat','gist_heat_r','PiYG','PiYG_r','PRGn','PRGn_r',
            'BrBG','BrBG_r','PuOr','PuOr_r','RdGy','RdGy_r','RdBu','RdBu_r',
            'RdYlBu','RdYlBu_r','RdYlGn','RdYlGn_r','Spectral','Spectral_r',
            'coolwarm','coolwarm_r','bwr','bwr_r','seismic','seismic_r','twilight',
            'twilight_r','twilight_shifted','twilight_shifted_r','hsv','hsv_r',
            'ocean','ocean_r','gist_earth','gist_earth_r','terrain','terrain_r',
            'gnuplot2','gnuplot2_r','CMRmap','CMRmap_r','gist_rainbow',
            'gist_rainbow_r','rainbow','rainbow_r','jet','jet_r'
        ]
        combo.addItems(cmap_choices)
        combo.setCurrentText(self.current_cmap)
        combo.setMaximumWidth(50)
        combo.setSizeAdjustPolicy(QComboBox.AdjustToContents)
        combo.hide()
        combo.setStyleSheet("""
        QComboBox {
            padding: 2px 6px;
            border:1px solid #888;
            border-radius:3px;
        }
        QComboBox::drop-down {width: 0px;}
        QComboBox::down-arrow {
        image: none;
        width: 0px; height: 0px;
        }
        """)

        # wiring
        btn.clicked.connect(lambda: combo.setVisible(not combo.isVisible()))
        combo.currentTextChanged.connect(self.change_colormap)

        lay.addWidget(btn,   alignment=Qt.AlignHCenter)
        lay.addWidget(combo, alignment=Qt.AlignHCenter)
        return widget


    # ----- helpers -------------------------------------------------------
    def _parse_pair(self, txt):
        parts = [p for p in txt.replace(",", " ").split() if p]
        if len(parts) != 2:
            raise ValueError
        return float(parts[0]), float(parts[1])

    # ----- callbacks -----------------------------------------------------
    def on_k1_enter(self):
        try:
            self.v1x, self.v1y = self._parse_pair(self._text_box1.text())
            self.regenerate_everything()
        except ValueError:
            QMessageBox.warning(self, "Input error",
                "k1 must be two numbers separated by comma or space.")

    def on_k2_enter(self):
        try:
            self.v2x, self.v2y = self._parse_pair(self._text_box2.text())
            self.regenerate_everything()
        except ValueError:
            QMessageBox.warning(self, "Input error",
                "k2 must be two numbers separated by comma or space.")

    def on_r0_enter(self):
        try:
            self.x0, self.y0 = self._parse_pair(self._text_box3.text())
            self.regenerate_everything()
        except ValueError:
            QMessageBox.warning(self, "Input error",
                "r0 must be two numbers separated by comma or space.")

    def on_sigma_enter(self):
        try:
            self.sigma = float(self._text_box4.text())
            self.regenerate_everything()
        except ValueError:
            QMessageBox.warning(self, "Input error", "σ must be a single number.")



    def create_tool_button(self, action):
        button = QToolButton()
        button.setDefaultAction(action)
        button.setToolButtonStyle(Qt.ToolButtonTextUnderIcon)
        button.setStyleSheet("""
        QToolButton {
            background-color: transparent;
            border: none;
            padding: 5px;
        }
        QToolButton:hover {
            background-color: #dcdcdc;
            border-radius: 5px;  
        }
        QToolButton:pressed {
            background-color: #c0c0c0;  /* Darker gray when pressed */
        }
    """)
        return button

    def upload_image(self):
        # Extended filter – add any other suffixes you need
        filter_str = ("Images or CSV (*.png *.jpg *.jpeg *.bmp *.tif *.tiff "
                      "*.csv);;All files (*)")

        file_path, _ = QFileDialog.getOpenFileName(self,
                                                   "Upload STM Image",
                                                   "",
                                                   filter_str)
        if not file_path:      # user hit cancel
            return

        img = self.load_image(file_path)
        if img is None:
            return 

        # ------------------------------------------------------------------
        # 1.  Store new data
        # ------------------------------------------------------------------
        self.atoms = img
        self.amps  = np.ones_like(img)
        self.Z     = self.atoms * self.amps
        self.Z_original = self.Z.copy()  # Store unfiltered original

        # ------------------------------------------------------------------
        # 2.  Refresh ORIGINAL-IMAGE panel
        # ------------------------------------------------------------------
        self._refresh_panel(self.Input_figure,
                            self.Input_Canvas,
                            self.Z,
                            show_axes=True)
        

        # ------------------------------------------------------------------
        # 3.  Re-compute & refresh FFT / AmpMap 
        # ------------------------------------------------------------------
        self.refresh_fft_amp_phase()         # <- implement once (see below)



    def download_file(self):
        file_name, _ = QFileDialog.getSaveFileName(self, "Download File", "", "All Files (*)")

    def show_about(self):
        pass

    def shortcuts(self):
        pass

    # ── ribbon-action helpers ─────────────────────────────────────────
    def toggle_cmap_selector(self):
        """Show / hide the global colormap drop-down."""
        vis = not self.cmap_selector.isVisible()
        self.cmap_selector.setVisible(vis)

    def change_colormap(self, cmap_name):
        """Apply the selected cmap to every image & redraw."""
        self.current_cmap = cmap_name
        for fig in self.all_figures:
            for ax in fig.axes:
                for im in ax.get_images():          # the AxesImage objects
                    im.set_cmap(cmap_name)
            fig.canvas.draw_idle()

    




    def createWidgetGroup(self, label_text, toggle_attr,
                          figure_name, canvas_name, attributes=None, show_tools=False):

        grid     = QGridLayout()
        groupbox = QGroupBox()
        grid.setSpacing(6)

        # ── 1.  Figure & Canvas (make them FIRST!) ------------------------------
        fig    = plt.figure(facecolor='none')  # Transparent background
        canvas = FigureCanvas(fig)
        canvas.setStyleSheet("background-color: transparent;")  # Transparent canvas
        self.all_figures.append(fig)

        # attach an invisible Matplotlib navigation toolbar
        toolbar = NavigationToolbar(canvas, self)
        toolbar.hide()              # don't add it to the layout, just keep it alive
        canvas.toolbar = toolbar    # make your slots happy

        setattr(self, figure_name, fig)
        setattr(self, canvas_name, canvas)

        # Create mouse position label
        mouse_label = QLabel("x: --, y: --")
        mouse_label.setVisible(False)
        mouse_label.setStyleSheet("QLabel { color: #333; font-size: 12px; font-weight: bold; padding: 5px; }")

        # Connect mouse movement event to update label
        def on_mouse_move(event):
            if event.inaxes and event.xdata is not None and event.ydata is not None:
                mouse_label.setText(f"x: {event.xdata:.2f}, y: {event.ydata:.2f}")
            else:
                mouse_label.setText("x: --, y: --")

        canvas.mpl_connect('motion_notify_event', on_mouse_move)

        # ── 2.  Title ► selector -------------------------------------------------
        selector = QComboBox()
        all_choices = ["Original Image",
                       "Fourier Transform",
                       "Phase - A",
                       "Phase - B",
                       "Amplitude Map - A",
                       "Amplitude Map - B",
                       "Strain Map - xx",
                       "Strain Map - yy",
                       "Strain Map - xy",
                       "Reconstuct Ideal Image",]
        selector.addItems(all_choices)
        if label_text in all_choices:
            selector.setCurrentText(label_text)

        # now the attributes exist, safe to capture them in the lambda
        selector.currentTextChanged.connect(
            lambda choice, fig=fig, canvas=canvas:
                self.draw_selected_plot(fig, canvas, choice))

        # ── 3. Tools button (next to selector) ---------------------------------
        tools_btn = None
        if show_tools:
            tools_btn = QPushButton()
            tools_btn.setIcon(qta.icon("fa5s.wrench"))
            tools_btn.setToolTip("Tools")
            tools_btn.setIconSize(QSize(20, 20))
            tools_btn.setFixedSize(30, 30)
            tools_btn.setFlat(True)
            tools_btn.setStyleSheet("""
                QPushButton {background: transparent; border: none; border-radius: 10px;}
                QPushButton:hover   {background:#cdcdcd;}
                QPushButton:pressed {background:#c0c0c0;}
            """)
            # Connect to _btn_tools handler
            tools_btn.clicked.connect(
                lambda _, f=fig, c=canvas, fig_name=figure_name:
                    self._btn_tools(tools_btn, f, c, None, fig_name))

        # --- BUTTON STRIP (vertical on left) -----------------------------------
        # one tuple per button:  (Font-Awesome icon, tooltip text, slot-name)
        button_specs = [
            ("fa5s.sync",        "Reset",              "_btn_reset"),
            ("fa5s.undo",        "Undo",           "_btn_undo"),
            ("fa5s.redo",        "Redo",           "_btn_redo"),
            ("fa5s.info",        "Metadata",       "_btn_view"),
            ("fa5s.arrows-alt",  "Pan / move image",        "_btn_pan"),
            ("fa5s.search-plus", "Zoom rectangle",          "_btn_zoom"),
            ("fa5s.download",    "Download",          "_btn_download"),
        ]

        buttons = []
        for icon_name, tip, slot_name in button_specs:
            btn = QPushButton()
            btn.setIcon(qta.icon(icon_name))
            btn.setToolTip(tip)
            btn.setIconSize(QSize(25, 25))
            btn.setFixedSize(30, 35)
            btn.setFlat(True)
            btn.setCheckable(slot_name in ("_btn_pan", "_btn_zoom"))  #only these two toggle
            btn.setStyleSheet("""
                QPushButton {background: transparent; border: none; border-radius: 10px;}
                QPushButton:hover   {background:#cdcdcd;}
                QPushButton:pressed {background:#c0c0c0;}
                QPushButton:checked {background:#909090;}      /* darker when ON   */
            """)
        
            # capture references for the slot so it can update the button state
            btn.clicked.connect(
                lambda _,
                       slot=getattr(self, slot_name),
                       btn_ref=btn, f=fig, c=canvas, toggle_attr=toggle_attr, fig_name=figure_name:
                       slot(btn_ref, f, c, toggle_attr, fig_name))
        
            if slot_name == "_btn_pan":
                canvas.pan_button  = btn
            if slot_name == "_btn_zoom":
                canvas.zoom_button = btn
        
            btn.setVisible(False)
            buttons.append(btn)



        tool_button = QPushButton()
        tool_button.setIcon(qta.icon('fa5s.bars'))
        tool_button.setToolTip("Toggle Attributes")
        tool_button.setStyleSheet(self.toggleButtonStyle())
        tool_button.clicked.connect(lambda: self.toggleAttribute(toggle_attr))

        # layout - selector on left, tools button next to it, toggle button at right
        grid.addWidget(selector,    0, 1, 1, 4, Qt.AlignLeft)
        if tools_btn:
            grid.addWidget(tools_btn,   0, 4, 1, 1, Qt.AlignLeft)
        grid.addWidget(tool_button, 0, 6, 1, 1, Qt.AlignTop | Qt.AlignRight)
        for i, button in enumerate(buttons):
            grid.addWidget(button, i + 1, 0)
        grid.addWidget(mouse_label, len(buttons) + 2, 0, 2, 7)
        grid.addWidget(canvas,      1, 1, 8, 6)

        groupbox.setLayout(grid)

        setattr(self, toggle_attr, buttons + [mouse_label])
        self.attribute_visibility_states[toggle_attr] = False
        return groupbox
    
    def draw_selected_plot(self, fig, canvas, kind):
        """plot selection helper"""
        # Determine which image to use: filtered (self.Z) for Input panel, original (self.Z_original) for others
        source_image = self.Z if fig == self.Input_figure else (self.Z_original if self.Z_original is not None else self.Z)

        if kind == "Original Image":
            data = source_image

        elif kind == "Fourier Transform":
            # Subtract mean before FFT (following PyAtoms approach)
            fft_result = npf.fftshift(npf.fft2(source_image - np.mean(source_image)))
            # Apply log scale if enabled
            if self.fft_log_scale:
                data = np.log(np.abs(fft_result) + 1)
            else:
                data = np.abs(fft_result)

        elif kind == "Reconstuct Ideal Image":
            # Ideal image reconstruction
            data = self.atoms

        elif kind == "Amplitude Map - A":
            # Amplitude Map using v1 (k1) - use stored data if available from GPA
            if hasattr(self, 'amplitude_a') and self.amplitude_a is not None:
                data = self.amplitude_a
            else:
                v1 = np.array([self.v1x, self.v1y])
                v2 = np.array([self.v2x, self.v2y])
                r0 = np.array([self.x0,  self.y0])
                data = np.abs(am.CDWampmap2D(source_image, r0, v1, v2, self.sigma, self.current_cmap))

        elif kind == "Amplitude Map - B":
            # Amplitude Map using v2 (k2) - use stored data if available from GPA
            if hasattr(self, 'amplitude_b') and self.amplitude_b is not None:
                data = self.amplitude_b
            else:
                v1 = np.array([self.v1x, self.v1y])
                v2 = np.array([self.v2x, self.v2y])
                r0 = np.array([self.x0,  self.y0])
                # Using v2 as the primary vector for B version
                data = np.abs(am.CDWampmap2D(source_image, r0, v2, v1, self.sigma, self.current_cmap))

        elif kind == "Phase - A":
            # Phase Field A - use calculated data if available, otherwise zeros
            if hasattr(self, 'phase_a') and self.phase_a is not None:
                data = self.phase_a
            else:
                data = np.zeros_like(source_image)

        elif kind == "Phase - B":
            # Phase Field B - use calculated data if available, otherwise zeros
            if hasattr(self, 'phase_b') and self.phase_b is not None:
                data = self.phase_b
            else:
                data = np.zeros_like(source_image)

        elif kind == "Displacement Field - A":
            # Displacement Field A - placeholder
            # TODO: Implement actual displacement field calculation for A
            data = np.zeros_like(source_image)

        elif kind == "Displacement Field - B":
            # Displacement Field B - placeholder
            # TODO: Implement actual displacement field calculation for B
            data = np.zeros_like(source_image)

        elif kind == "Strain Map - xx":
            # Strain Map XX - use calculated data if available, otherwise zeros
            if hasattr(self, 'strain_xx') and self.strain_xx is not None:
                data = self.strain_xx
            else:
                data = np.zeros_like(source_image)

        elif kind == "Strain Map - yy":
            # Strain Map YY - use calculated data if available, otherwise zeros
            if hasattr(self, 'strain_yy') and self.strain_yy is not None:
                data = self.strain_yy
            else:
                data = np.zeros_like(source_image)

        elif kind == "Strain Map - xy":
            # Strain Map XY - use calculated data if available, otherwise zeros
            if hasattr(self, 'strain_xy') and self.strain_xy is not None:
                data = self.strain_xy
            else:
                data = np.zeros_like(source_image)

        else:            # fallback – should never happen
            data = source_image

        self._refresh_panel(fig, canvas, data, show_axes=(kind == "Original Image"))

        # ========== PER-PANEL BUTTON SLOTS ==================================
    # ---------- navigation & utility button callbacks -------------------
    def _btn_reset(self, _btn, _fig, canvas, toggle_attr=None, fig_name=None):
        if canvas.toolbar:
            canvas.toolbar.home()

    def _btn_undo(self, _btn, fig, canvas, toggle_attr=None, fig_name=None):
        """Undo last image processing operation or zoom/pan state."""
        if fig_name and fig_name in self.image_history:
            # Use image history for undo
            history = self.image_history[fig_name]
            idx = self.history_index[fig_name]

            if idx > 0:
                # Move back in history
                idx -= 1
                self.history_index[fig_name] = idx
                # Restore previous image
                self.Z = history[idx].copy()
                self._refresh_panel(fig, canvas, self.Z, show_axes=True)
            elif canvas.toolbar:
                # At beginning of image history, try toolbar undo for zoom/pan
                canvas.toolbar.back()
        elif canvas.toolbar:
            # Fallback to toolbar undo (zoom/pan)
            canvas.toolbar.back()

    def _btn_redo(self, _btn, fig, canvas, toggle_attr=None, fig_name=None):
        """Redo last undone image processing operation or zoom/pan state."""
        if fig_name and fig_name in self.image_history:
            # Use image history for redo
            history = self.image_history[fig_name]
            idx = self.history_index[fig_name]

            if idx < len(history) - 1:
                # Move forward in history
                idx += 1
                self.history_index[fig_name] = idx
                # Restore next image
                self.Z = history[idx].copy()
                self._refresh_panel(fig, canvas, self.Z, show_axes=True)
            elif canvas.toolbar:
                # At end of image history, try toolbar redo for zoom/pan
                canvas.toolbar.forward()
        elif canvas.toolbar:
            # Fallback to toolbar redo (zoom/pan)
            canvas.toolbar.forward()

    def _btn_view(self, _btn, fig, canvas, toggle_attr=None, fig_name=None):
        """Show metadata dialog for the current image."""
        metadata = self.get_current_metadata(fig, canvas)
        dialog = MetadataDialog(metadata, parent=self)
        dialog.exec_()

    def _btn_tools(self, _btn, fig, canvas, toggle_attr=None, fig_name=None):
        """Show tools dialog for image processing."""
        # Check if this is FFT panel
        if fig_name == "FFT_figure":
            # FFT-specific tools
            self._btn_fft_tools(_btn, fig, canvas, toggle_attr, fig_name)
            return

        # Original image processing tools
        # Get current image data
        current_data = None
        if fig.axes and fig.axes[0].images:
            current_data = fig.axes[0].images[0].get_array()
        elif hasattr(self, 'Z') and self.Z is not None:
            current_data = self.Z

        if current_data is None:
            QMessageBox.warning(self, "No Image", "No image data available to process.")
            return

        # Save current state to history before applying tool
        if fig_name and fig_name in self.image_history:
            history = self.image_history[fig_name]
            idx = self.history_index[fig_name]

            # Remove any redo history (we're creating a new branch)
            if idx < len(history) - 1:
                history = history[:idx + 1]
                self.image_history[fig_name] = history

            # Add current state to history
            history.append(current_data.copy())
            self.history_index[fig_name] = len(history) - 1

        # Show tools dialog
        dialog = ToolsDialog(current_data, parent=self)
        if dialog.exec_() == QDialog.Accepted and dialog.processed_image is not None:
            # Add processed image to history
            if fig_name and fig_name in self.image_history:
                history = self.image_history[fig_name]
                history.append(dialog.processed_image.copy())
                self.history_index[fig_name] = len(history) - 1

            # Update the image with processed result
            self.Z = dialog.processed_image
            # Refresh the display
            self._refresh_panel(fig, canvas, self.Z, show_axes=True)

    def _btn_fft_tools(self, _btn, fig, canvas, toggle_attr=None, fig_name=None):
        """Show FFT-specific tools dialog."""
        # Get original image (real space)
        original_image = self.Z_original if self.Z_original is not None else self.Z

        # Get current FFT data
        current_fft_data = None
        if fig.axes and fig.axes[0].images:
            # Get displayed data (magnitude of FFT)
            displayed_data = fig.axes[0].images[0].get_array()
            # Reconstruct FFT from original image for processing
            # Subtract mean before FFT (following PyAtoms approach)
            current_fft_data = np.fft.fftshift(np.fft.fft2(original_image - np.mean(original_image)))

        if current_fft_data is None:
            QMessageBox.warning(self, "No FFT Data", "No FFT data available to process.")
            return

        # Save current state to history before applying tool
        if fig_name and fig_name in self.image_history:
            history = self.image_history[fig_name]
            idx = self.history_index[fig_name]

            # Remove any redo history (we're creating a new branch)
            if idx < len(history) - 1:
                history = history[:idx + 1]
                self.image_history[fig_name] = history

            # Add current state to history (with current scale)
            if self.fft_log_scale:
                history.append(np.log(np.abs(current_fft_data) + 1).copy())
            else:
                history.append(np.abs(current_fft_data).copy())
            self.history_index[fig_name] = len(history) - 1

        # Show FFT tools dialog
        dialog = FFTToolsDialog(original_image, current_fft_data, parent=self)
        if dialog.exec_() == QDialog.Accepted and dialog.processed_fft is not None:
            # Add processed FFT to history (with current scale)
            if fig_name and fig_name in self.image_history:
                history = self.image_history[fig_name]
                if self.fft_log_scale:
                    history.append(np.log(np.abs(dialog.processed_fft) + 1).copy())
                else:
                    history.append(np.abs(dialog.processed_fft).copy())
                self.history_index[fig_name] = len(history) - 1

            # Refresh the display with magnitude of processed FFT (with current scale)
            if self.fft_log_scale:
                display_data = np.log(np.abs(dialog.processed_fft) + 1)
            else:
                display_data = np.abs(dialog.processed_fft)
            self._refresh_panel(fig, canvas, display_data, show_axes=False)

    def _btn_pan(self, button, _fig, canvas, toggle_attr=None, fig_name=None):
        tb = canvas.toolbar
        tb.pan()
        is_on = (tb.mode == 'pan/zoom')
        button.setChecked(is_on)
        if hasattr(canvas, "zoom_button"):
            canvas.zoom_button.setChecked(False)

    def _btn_zoom(self, button, _fig, canvas, toggle_attr=None, fig_name=None):
        tb = canvas.toolbar
        tb.zoom()
        is_on = (tb.mode == 'zoom rect')
        button.setChecked(is_on)
        if hasattr(canvas, "pan_button"):
            canvas.pan_button.setChecked(False)

    def _btn_download(self, _btn, fig, _canvas, toggle_attr=None, fig_name=None):
        filter_str = ("Images (*.png *.jpg *.jpeg *.bmp *.tif *.tiff);;"
                      "CSV height-map (*.csv);;All files (*)")
        file_path, _ = QFileDialog.getSaveFileName(self, "Save image", "", filter_str)
        if not file_path:
            return

        ext = os.path.splitext(file_path)[1].lower()
        if ext == ".csv":
            ax = fig.axes[0] if fig.axes else None
            if ax and ax.images:
                data = ax.images[0].get_array()
                np.savetxt(file_path, data, delimiter=",")
        else:
            fig.savefig(file_path, dpi=300, bbox_inches="tight")
    # --------------------------------------------------------------------





    def initInput(self):
        # Special logic for Input Widget
        groupbox = self.createWidgetGroup(
            "Original Image", "Input", "Input_figure", "Input_Canvas", show_tools=True
        )

        # Initialize history for this widget
        self.image_history["Input_figure"] = []
        self.history_index["Input_figure"] = -1

        # Plot logic for Input
        self.Input_figure.patch.set_alpha(0.0)  # Transparent figure background
        ax = self.Input_figure.add_subplot()
        ax.patch.set_alpha(0.0)                 # Transparent axes background

        colormap = 'plasma'

        # Load sample image from pyAtomsSample.png
        sample_image_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "pyAtomsSample.png")
        if os.path.exists(sample_image_path):
            img = self.load_image(sample_image_path)
            self.atoms = img
            self.amps = np.ones_like(img)
        else:
            # Fallback to generated data if sample not found
            pix, L, a, theta, honeycomb = 256, 10, 0.7, 45, 1
            self.atoms = am.hexatoms(pix, L, a, theta, honeycomb)
            self.amps = np.ones(self.atoms.shape)
            N = int(len(self.amps) / 2)
            self.amps[:N, :N] *= -0.6
            self.amps[:N, N:] *= 3.2
            self.amps[N:, :N] *= 1.4
            self.amps[N:, N:] *= -1.9

        self.Z = self.atoms * self.amps
        self.Z_original = self.Z.copy()  # Store unfiltered original

        # Add initial image to history
        self.image_history["Input_figure"].append(self.Z.copy())
        self.history_index["Input_figure"] = 0

        fig=ax.imshow(self.Z, cmap=colormap)
        ax.grid(False)
        ax.set_xlabel('x (pix)')
        ax.set_ylabel('y (pix)')
        ax.axis('On')
        am.colorbar(fig)
        self.Input_figure.tight_layout()

        return groupbox

    def initFFT(self):
        # Special logic for FFT Widget
        groupbox = self.createWidgetGroup(
            "Fourier Transform", "FFT", "FFT_figure", "FFT_Canvas", show_tools=True
        )

        # Initialize history for this widget
        self.image_history["FFT_figure"] = []
        self.history_index["FFT_figure"] = -1

        # Add Bragg Peak Selection button and log scale checkbox to the layout
        layout = groupbox.layout()

        # Bragg Peak Selection button
        self.bragg_peak_btn = QPushButton("Bragg Peak Selection")
        self.bragg_peak_btn.setStyleSheet("QPushButton { font-size: 10px; padding: 3px 8px; }")
        self.bragg_peak_btn.clicked.connect(self.toggle_bragg_peak_selection)
        self.bragg_peak_btn.setCheckable(True)
        layout.addWidget(self.bragg_peak_btn, 0, 5, 1, 1, Qt.AlignLeft)

        # Log scale checkbox
        self.fft_log_checkbox = QCheckBox("Log Scale")
        self.fft_log_checkbox.setChecked(self.fft_log_scale)
        self.fft_log_checkbox.setStyleSheet("QCheckBox { font-size: 11px; padding: 2px; }")
        self.fft_log_checkbox.stateChanged.connect(self.toggle_fft_log_scale)
        layout.addWidget(self.fft_log_checkbox, 1, 5, 1, 1, Qt.AlignLeft)

        # FFT peak coordinate label
        self.fft_peak_label = QLabel("g-vector: (---, ---)")
        self.fft_peak_label.setStyleSheet("QLabel { color: #333; font-size: 11px; font-weight: bold; padding: 2px; }")
        layout.addWidget(self.fft_peak_label, 2, 5, 1, 2, Qt.AlignLeft)

        # Connect custom mouse handler for FFT to show g-vectors
        def on_fft_mouse_move(event):
            if event.inaxes and event.xdata is not None and event.ydata is not None:
                # Get image shape
                if self.Z_original is not None:
                    rows, cols = self.Z_original.shape
                elif self.Z is not None:
                    rows, cols = self.Z.shape
                else:
                    rows, cols = 256, 256

                center_x, center_y = cols // 2, rows // 2

                # Calculate g-vector (relative to center)
                gx = event.xdata - center_x
                gy = event.ydata - center_y

                self.fft_peak_label.setText(f"g-vector: ({gx:.4f}, {gy:.4f})")
            else:
                self.fft_peak_label.setText("g-vector: (---, ---)")

        self.FFT_Canvas.mpl_connect('motion_notify_event', on_fft_mouse_move)

        # Plot logic for FFT
        self.FFT_figure.patch.set_alpha(0.0)  # Transparent figure background
        ax = self.FFT_figure.add_subplot()
        ax.patch.set_alpha(0.0)               # Transparent axes background
        # Use Z_original if available, otherwise initialize Z
        if self.Z_original is None:
            self.Z = self.amps * self.atoms
            self.Z_original = self.Z.copy()
        # Subtract mean before FFT (following PyAtoms approach)
        fft_atoms = npf.fftshift(npf.fft2(self.Z_original - np.mean(self.Z_original)))
        # Default to linear scale
        fft_magnitude = np.abs(fft_atoms)

        # Add initial FFT to history
        self.image_history["FFT_figure"].append(fft_magnitude.copy())
        self.history_index["FFT_figure"] = 0

        fig = ax.imshow(fft_magnitude, cmap='plasma')
        #ax.set_title("FFT")
        ax.grid(False)
        ax.axis('Off')
        am.colorbar(fig)
        self.FFT_figure.tight_layout()

        return groupbox

    def initIdeal(self):
        # Special logic for Ideal Widget
        group_box = self.createWidgetGroup(
            "Ideal Image", "Ideal", "Ideal_figure", "Ideal_Canvas"
        )
        
        return group_box
        

    def initAmpMap(self):
        # Special logic for Amplitude Mask Widget
        groupbox = self.createWidgetGroup(
            "Amplitude Map", "AmpMap", "AmpMap_figure", "AmpMap_Canvas"
        )

        # Plot logic for AmpMask
        self.AmpMap_figure.patch.set_alpha(0.0)  # Transparent figure background
        ax = self.AmpMap_figure.add_subplot()
        ax.patch.set_alpha(0.0)                  # Transparent axes background
        v1x, v1y, v2x, v2y, x0, y0, sigma = 133, 111, 145, 123, 204, 52, 5
        v1, v2, r0 = np.array([v1x, v1y]), np.array([v2x, v2y]), np.array([x0, y0])
        fig = ax.imshow(
            np.abs(am.CDWampmap2D(self.Z, r0, v1, v2, sigma, 'plasma')),
            cmap='plasma',
        )
        #ax.set_title("Z1 Amplitude Map, sigma = " + str(sigma))
        ax.axis('Off')
        am.colorbar(fig)
        self.AmpMap_figure.tight_layout()

        return groupbox

    def initDisFld(self):
        # Special logic for Displacement Field Widget
        return self.createWidgetGroup(
            "Displacement Field", "DisFld", "DisFld_figure", "DisFld_Canvas"
        )

    def initPhaseFld(self):
        # Special logic for Phase Field Widget
        return self.createWidgetGroup(
            "Phase Field", "PhaseFld", "PhaseFld_figure", "PhaseFld_Canvas"
        )

    def initPhaseA(self):
        # Phase A Widget
        groupbox = self.createWidgetGroup(
            "Phase - A", "PhaseA", "PhaseA_figure", "PhaseA_Canvas"
        )

        # Get layout to add custom buttons
        layout = groupbox.layout()

        # Run GPA button
        self.run_gpa_a_btn = QPushButton("Run GPA")
        self.run_gpa_a_btn.setStyleSheet("QPushButton { font-size: 10px; padding: 3px 8px; }")
        self.run_gpa_a_btn.clicked.connect(lambda: self.run_gpa('A'))
        layout.addWidget(self.run_gpa_a_btn, 0, 5, 1, 1, Qt.AlignLeft)

        # GPA Settings button (wrench icon)
        self.gpa_settings_a_btn = QPushButton()
        self.gpa_settings_a_btn.setIcon(qta.icon("fa5s.cog"))
        self.gpa_settings_a_btn.setToolTip("GPA Settings")
        self.gpa_settings_a_btn.setIconSize(QSize(16, 16))
        self.gpa_settings_a_btn.setFixedSize(26, 26)
        self.gpa_settings_a_btn.setFlat(True)
        self.gpa_settings_a_btn.setStyleSheet("""
            QPushButton {background: transparent; border: none; border-radius: 10px;}
            QPushButton:hover   {background:#cdcdcd;}
            QPushButton:pressed {background:#c0c0c0;}
        """)
        self.gpa_settings_a_btn.clicked.connect(self.show_gpa_settings)
        layout.addWidget(self.gpa_settings_a_btn, 0, 6, 1, 1, Qt.AlignLeft)

        # Plot logic for Phase A - default to zeros
        self.PhaseA_figure.patch.set_alpha(0.0)
        ax = self.PhaseA_figure.add_subplot()
        ax.patch.set_alpha(0.0)

        # Default to zeros
        if hasattr(self, 'Z_original') and self.Z_original is not None:
            phase_data = np.zeros_like(self.Z_original)
        else:
            phase_data = np.zeros((256, 256))

        fig = ax.imshow(phase_data, cmap='plasma')
        ax.grid(False)
        ax.axis('Off')
        am.colorbar(fig)
        self.PhaseA_figure.tight_layout()

        return groupbox

    def initPhaseB(self):
        # Phase B Widget
        groupbox = self.createWidgetGroup(
            "Phase - B", "PhaseB", "PhaseB_figure", "PhaseB_Canvas"
        )

        # Get layout to add custom buttons
        layout = groupbox.layout()

        # Run GPA button
        self.run_gpa_b_btn = QPushButton("Run GPA")
        self.run_gpa_b_btn.setStyleSheet("QPushButton { font-size: 10px; padding: 3px 8px; }")
        self.run_gpa_b_btn.clicked.connect(lambda: self.run_gpa('B'))
        layout.addWidget(self.run_gpa_b_btn, 0, 5, 1, 1, Qt.AlignLeft)

        # GPA Settings button (wrench icon)
        self.gpa_settings_b_btn = QPushButton()
        self.gpa_settings_b_btn.setIcon(qta.icon("fa5s.cog"))
        self.gpa_settings_b_btn.setToolTip("GPA Settings")
        self.gpa_settings_b_btn.setIconSize(QSize(16, 16))
        self.gpa_settings_b_btn.setFixedSize(26, 26)
        self.gpa_settings_b_btn.setFlat(True)
        self.gpa_settings_b_btn.setStyleSheet("""
            QPushButton {background: transparent; border: none; border-radius: 10px;}
            QPushButton:hover   {background:#cdcdcd;}
            QPushButton:pressed {background:#c0c0c0;}
        """)
        self.gpa_settings_b_btn.clicked.connect(self.show_gpa_settings)
        layout.addWidget(self.gpa_settings_b_btn, 0, 6, 1, 1, Qt.AlignLeft)

        # Plot logic for Phase B - default to zeros
        self.PhaseB_figure.patch.set_alpha(0.0)
        ax = self.PhaseB_figure.add_subplot()
        ax.patch.set_alpha(0.0)

        # Default to zeros
        if hasattr(self, 'Z_original') and self.Z_original is not None:
            phase_data = np.zeros_like(self.Z_original)
        else:
            phase_data = np.zeros((256, 256))

        fig = ax.imshow(phase_data, cmap='plasma')
        ax.grid(False)
        ax.axis('Off')
        am.colorbar(fig)
        self.PhaseB_figure.tight_layout()

        return groupbox

    def initStrainXX(self):
        # Strain Map XX Widget
        groupbox = self.createWidgetGroup(
            "Strain Map - xx", "StrainXX", "StrainXX_figure", "StrainXX_Canvas"
        )

        # Plot logic for Strain XX - default to zeros
        self.StrainXX_figure.patch.set_alpha(0.0)
        ax = self.StrainXX_figure.add_subplot()
        ax.patch.set_alpha(0.0)

        # Default to zeros
        if hasattr(self, 'Z_original') and self.Z_original is not None:
            strain_data = np.zeros_like(self.Z_original)
        else:
            strain_data = np.zeros((256, 256))

        fig = ax.imshow(strain_data, cmap='plasma')
        ax.grid(False)
        ax.axis('Off')
        am.colorbar(fig)
        self.StrainXX_figure.tight_layout()

        return groupbox

    def initStrainYY(self):
        # Strain Map YY Widget
        groupbox = self.createWidgetGroup(
            "Strain Map - yy", "StrainYY", "StrainYY_figure", "StrainYY_Canvas"
        )

        # Plot logic for Strain YY - default to zeros
        self.StrainYY_figure.patch.set_alpha(0.0)
        ax = self.StrainYY_figure.add_subplot()
        ax.patch.set_alpha(0.0)

        # Default to zeros
        if hasattr(self, 'Z_original') and self.Z_original is not None:
            strain_data = np.zeros_like(self.Z_original)
        else:
            strain_data = np.zeros((256, 256))

        fig = ax.imshow(strain_data, cmap='plasma')
        ax.grid(False)
        ax.axis('Off')
        am.colorbar(fig)
        self.StrainYY_figure.tight_layout()

        return groupbox

    def load_image(self, file_path):
        """
        Reads an STM file (png/jpg/bmp/tif or CSV) → 2-D float array.
        If the array is not square, asks the user whether to crop
        the centre to the largest possible square.
        """
        ext = os.path.splitext(file_path)[1].lower()

        # ── 1. Read ----------------------------------------------------------------
        if ext == ".csv":
            img = np.loadtxt(file_path, delimiter=",", dtype=float)
        else:
            img = plt.imread(file_path)
            if img.ndim == 3:           # RGB → grey
                img = np.dot(img[..., :3], [0.2989, 0.5870, 0.1140])

        img = img.astype(float)

        # ── 2. Square-check --------------------------------------------------------
        h, w = img.shape
        if h != w:
            side = min(h, w)
            reply = QMessageBox.question(
                self,
                "Non-square image",
                (f"This software needs a square matrix.\n"
                 f"Current size = {h}×{w}.\n\n"
                 f"Crop centre {side}×{side}?"),
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.Yes)

            if reply == QMessageBox.Yes:
                top  = (h - side) // 2
                left = (w - side) // 2
                img  = img[top:top + side, left:left + side]
            else:
                return None          # user declined; abort upload

        # ── 3. Normalise to 0–1 ----------------------------------------------------
        if img.max() > img.min():
            img = (img - img.min()) / (img.max() - img.min())

        return img

    def get_current_metadata(self, fig, canvas):
        """Collect metadata for the current image being displayed."""
        metadata = {}

        # Get the current image data from the figure
        current_data = None
        if fig.axes and fig.axes[0].images:
            current_data = fig.axes[0].images[0].get_array()
        elif hasattr(self, 'Z') and self.Z is not None:
            current_data = self.Z

        if current_data is not None:
            # Image properties
            metadata["Image Properties"] = {
                "Shape": f"{current_data.shape[0]} x {current_data.shape[1]}",
                "Data Type": str(current_data.dtype),
                "Min Value": f"{current_data.min():.6f}",
                "Max Value": f"{current_data.max():.6f}",
                "Mean Value": f"{current_data.mean():.6f}",
                "Std Deviation": f"{current_data.std():.6f}",
            }

        # Analysis Parameters
        metadata["Analysis Parameters"] = {
            "FFT Peak k1": {
                "x": f"{self.v1x:.2f} px",
                "y": f"{self.v1y:.2f} px",
            },
            "FFT Peak k2": {
                "x": f"{self.v2x:.2f} px",
                "y": f"{self.v2y:.2f} px",
            },
            "Reference Vector r0": {
                "x": f"{self.x0:.2f} px",
                "y": f"{self.y0:.2f} px",
            },
            "Gaussian Parameter σ": f"{self.sigma:.2f}",
        }

        # Display Settings
        metadata["Display Settings"] = {
            "Colormap": self.current_cmap,
            "Figure Size": f"{fig.get_figwidth():.2f} x {fig.get_figheight():.2f} inches",
            "DPI": f"{fig.dpi}",
        }

        # Axes information
        if fig.axes:
            ax = fig.axes[0]
            axes_info = {
                "x": {
                    "label": ax.get_xlabel() or "Not set",
                    "limits": f"{ax.get_xlim()[0]:.2f} to {ax.get_xlim()[1]:.2f}",
                    "visible": ax.xaxis.get_visible(),
                },
                "y": {
                    "label": ax.get_ylabel() or "Not set",
                    "limits": f"{ax.get_ylim()[0]:.2f} to {ax.get_ylim()[1]:.2f}",
                    "visible": ax.yaxis.get_visible(),
                }
            }
            metadata["Axes"] = axes_info

        return metadata

    def _refresh_panel(self, fig, canvas, img2d, *,
                       cmap="plasma",
                       xlabel="x (pix)",
                       ylabel="y (pix)",
                       show_axes=True):
        """
        Clears the *entire* figure, shows `img2d`, adds one colour-bar, and
        queues a repaint. Works for Input, FFT, AmpMap … any 2-D image panel.
        """
        fig.clf()                                # nuke everything
        fig.patch.set_alpha(0.0)                 # Transparent figure background
        ax = fig.add_subplot(111)
        ax.patch.set_alpha(0.0)                  # Transparent axes background

        im = ax.imshow(img2d, cmap=cmap)
        if show_axes:
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
        else:
            ax.axis("off")

        fig.colorbar(im, ax=ax)                  # one colour-bar only
        fig.tight_layout()                       # Minimize whitespace
        canvas.draw_idle()                       # repaint later


    
    def refresh_fft_amp_phase(self):
        # --------- FFT ---------------------------------------------------
        fft_atoms = npf.fftshift(npf.fft2(self.Z - self.Z.mean()))

        self.FFT_figure.clf()                 # ← nuke axes *and* old colour-bars
        self.FFT_figure.patch.set_alpha(0.0)  # Transparent figure background
        ax = self.FFT_figure.add_subplot(111)
        ax.patch.set_alpha(0.0)               # Transparent axes background

        im = ax.imshow(np.abs(fft_atoms), cmap=self.current_cmap)
        ax.axis("off")

        # draw / update red "X" at current k1
        if self.fft_peak_scatter:
            self.fft_peak_scatter.remove()
        self.fft_peak_scatter = ax.scatter(self.v1x, self.v1y,
                                           color="red", marker="x", s=60,
                                           label="User Selection")

        ax.legend(loc="upper right", fontsize=7)
        self.FFT_figure.colorbar(im, ax=ax)
        self.FFT_figure.tight_layout()        # Minimize whitespace
        self.FFT_Canvas.draw_idle()

        # --------- Amp-map etc. -----------------------------------------
        v1 = np.array([self.v1x, self.v1y])
        v2 = np.array([self.v2x, self.v2y])
        r0 = np.array([self.x0,  self.y0])
        ampmap = np.abs(am.CDWampmap2D(self.Z, r0, v1, v2,
                                       self.sigma, self.current_cmap))
        self._refresh_panel(self.AmpMap_figure, self.AmpMap_Canvas,
                            ampmap, show_axes=False)

        # ---------- Displacement & Phase fields --------------------------
        # incomplete
    def _parse_pair(self, text):
        # input helper
        parts = [p for p in text.replace(",", " ").split() if p]
        if len(parts) != 2:
            raise ValueError
        return float(parts[0]), float(parts[1])

    def regenerate_everything(self):
        """Recompute Z-space maps that depend on (k1,k2,r0,sigma) and repaint."""
        v1 = np.array([self.v1x, self.v1y])
        v2 = np.array([self.v2x, self.v2y])
        r0 = np.array([self.x0,  self.y0])

        ampmap = np.abs(am.CDWampmap2D(self.Z, r0, v1, v2, self.sigma, self.current_cmap))
        self._refresh_panel(self.AmpMap_figure, self.AmpMap_Canvas,
                            ampmap, show_axes=False)
        self.refresh_fft_amp_phase()

        # ---- if σ affects Displacement / Phase, update those here -------
        # self.refresh_fft_amp_phase()  # <- only if needed





    def toggleAttribute(self, attribute_name):
        self.attribute_visibility_states[attribute_name] = not self.attribute_visibility_states.get(attribute_name, False)
        for widget in getattr(self, attribute_name):
            widget.setVisible(self.attribute_visibility_states[attribute_name])

    def toggle_fft_log_scale(self, state):
        """Toggle FFT display between linear and log scale."""
        self.fft_log_scale = bool(state)
        # Refresh FFT display
        if hasattr(self, 'FFT_Canvas') and hasattr(self, 'FFT_figure'):
            # Get current selection from FFT panel selector
            # Re-compute and display FFT with new scale
            source_image = self.Z_original if self.Z_original is not None else self.Z
            # Subtract mean before FFT (following PyAtoms approach)
            fft_result = npf.fftshift(npf.fft2(source_image - np.mean(source_image)))

            if self.fft_log_scale:
                data = np.log(np.abs(fft_result) + 1)
            else:
                data = np.abs(fft_result)

            self._refresh_panel(self.FFT_figure, self.FFT_Canvas, data, show_axes=False)

    def toggle_bragg_peak_selection(self, checked):
        """Toggle Bragg peak selection mode."""
        if checked:
            # Enable peak selection mode
            self.start_bragg_peak_selection()
        else:
            # Disable peak selection mode
            self.clear_bragg_peak_circles()

    def start_bragg_peak_selection(self):
        """Start Bragg peak selection by adding circles to FFT."""
        # Clear any existing circles
        self.clear_bragg_peak_circles()

        # Get FFT data and find local peaks
        source_image = self.Z_original if self.Z_original is not None else self.Z
        fft_result = npf.fftshift(npf.fft2(source_image - np.mean(source_image)))
        fft_magnitude = np.abs(fft_result)

        # Find two prominent peaks (excluding center)
        from skimage.feature import peak_local_max
        rows, cols = fft_magnitude.shape
        center_y, center_x = rows // 2, cols // 2

        # Smooth FFT for better peak detection
        from scipy.ndimage import gaussian_filter
        smoothed = gaussian_filter(fft_magnitude, sigma=2)

        # Find local maxima
        peak_coords = peak_local_max(smoothed, min_distance=20, threshold_rel=0.3, num_peaks=10)

        # Filter out center peak and select two non-colinear peaks
        selected_peaks = []
        for py, px in peak_coords:
            dist_from_center = np.sqrt((px - center_x)**2 + (py - center_y)**2)
            if dist_from_center > 15:  # Exclude center region
                selected_peaks.append((px, py))
            if len(selected_peaks) >= 2:
                break

        # If we didn't find enough peaks, use default positions
        if len(selected_peaks) < 2:
            selected_peaks = [
                (int(center_x * 1.3), int(center_y * 0.7)),
                (int(center_x * 0.7), int(center_y * 1.3))
            ]

        # Add draggable circles at peak positions
        ax = self.FFT_figure.axes[0]
        for px, py in selected_peaks[:2]:
            circle = DraggableCircle(ax, (px, py), radius=self.gpa_mask_radius, color='yellow', alpha=0.3)
            self.bragg_peak_circles.append(circle)

        # Apply auto-centering if enabled
        if self.gpa_auto_center:
            for circle in self.bragg_peak_circles:
                self.auto_center_circle(circle, fft_magnitude)

        self.FFT_Canvas.draw()

    def clear_bragg_peak_circles(self):
        """Remove all Bragg peak selection circles."""
        for circle in self.bragg_peak_circles:
            circle.disconnect()
            circle.remove()
        self.bragg_peak_circles = []
        if hasattr(self, 'FFT_Canvas'):
            self.FFT_Canvas.draw()

    def auto_center_circle(self, circle, fft_magnitude):
        """Center circle on highest intensity pixel within the circle."""
        center = circle.get_center()
        radius = circle.get_radius()

        # Get circle bounds
        cx, cy = int(center[0]), int(center[1])
        r = int(radius)

        # Get image dimensions
        rows, cols = fft_magnitude.shape

        # Define search region (within circle)
        y_min = max(0, cy - r)
        y_max = min(rows, cy + r + 1)
        x_min = max(0, cx - r)
        x_max = min(cols, cx + r + 1)

        # Extract region
        region = fft_magnitude[y_min:y_max, x_min:x_max]

        # Create circular mask
        yy, xx = np.ogrid[y_min:y_max, x_min:x_max]
        circle_mask = (xx - cx)**2 + (yy - cy)**2 <= r**2

        # Apply mask
        masked_region = np.where(circle_mask[:region.shape[0], :region.shape[1]],
                                  region, -np.inf)

        # Find maximum within circle
        if masked_region.size > 0:
            max_idx = np.unravel_index(np.argmax(masked_region), masked_region.shape)
            new_cx = x_min + max_idx[1]
            new_cy = y_min + max_idx[0]

            # Update circle center
            circle.circle.center = (new_cx, new_cy)

            # Update bounding box if circle is selected
            if circle.selected and circle.bbox is not None:
                circle.create_bounding_box()

    def apply_auto_center_to_circles(self):
        """Apply auto-centering to all existing circles."""
        if not self.bragg_peak_circles:
            return

        # Get FFT data
        source_image = self.Z_original if self.Z_original is not None else self.Z
        fft_result = npf.fftshift(npf.fft2(source_image - np.mean(source_image)))
        fft_magnitude = np.abs(fft_result)

        # Auto-center each circle
        for circle in self.bragg_peak_circles:
            self.auto_center_circle(circle, fft_magnitude)

        # Redraw
        if hasattr(self, 'FFT_Canvas'):
            self.FFT_Canvas.draw()

    def show_gpa_settings(self):
        """Show GPA settings dialog."""
        dialog = GPASettingsDialog(
            current_algorithm=self.gpa_algorithm,
            current_sigma=self.gpa_sigma,
            current_auto_center=self.gpa_auto_center,
            current_refine_peaks=self.gpa_refine_peaks,
            parent=self
        )
        if dialog.exec_() == QDialog.Accepted:
            self.gpa_algorithm = dialog.algorithm
            self.gpa_sigma = dialog.sigma
            self.gpa_auto_center = dialog.auto_center
            self.gpa_refine_peaks = dialog.refine_peaks

            # If auto-center is enabled, apply it to existing circles
            if self.gpa_auto_center and self.bragg_peak_circles:
                self.apply_auto_center_to_circles()

            # If refine peaks is enabled, apply refinement to existing circles
            if self.gpa_refine_peaks and self.bragg_peak_circles:
                self.refine_peak_positions()

    def refine_peak_positions(self):
        """Refine peak positions using iterative slope phase method (plane fit)."""
        if not self.bragg_peak_circles:
            return

        # Get source image
        source_image = self.Z_original if self.Z_original is not None else self.Z
        rows, cols = source_image.shape
        center_x, center_y = cols // 2, rows // 2

        from skimage.restoration import unwrap_phase

        def filter_outliers_iqr(data, threshold=0.7):
            """Remove outliers using IQR method."""
            data = np.array(data)
            q1, q3 = np.percentile(data, [25, 75])
            iqr = q3 - q1
            lower_bound = q1 - threshold * iqr
            upper_bound = q3 + threshold * iqr
            filtered = data[(data >= lower_bound) & (data <= upper_bound)]
            return np.mean(filtered) if len(filtered) > 0 else np.mean(data)

        for circle in self.bragg_peak_circles:
            cx, cy = circle.get_center()

            # Initial g-vector in pixel coordinates (relative to center)
            v_real_x = cx
            v_real_y = cy

            # Convert to q (rad/pix)
            q_x = (v_real_x - center_x) * (2 * np.pi / cols)
            q_y = (v_real_y - center_y) * (2 * np.pi / rows)

            # Iterative refinement
            max_iterations = 50
            tol = 1e-5
            sigma_mask = 3.0  # Gaussian mask sigma

            for iteration in range(max_iterations):
                # Create coordinate grids
                Y, X = np.meshgrid(np.arange(rows), np.arange(cols), indexing='ij')

                # Demodulate: multiply by exp(-i * q · r)
                phase_carrier = -1j * (q_x * (X - center_x) + q_y * (Y - center_y))
                Z_shifted = source_image * np.exp(phase_carrier)

                # FFT of demodulated signal
                fftZ = npf.fftshift(npf.fft2(Z_shifted))

                # Apply Gaussian mask around zero frequency
                kY, kX = np.meshgrid(np.arange(rows), np.arange(cols), indexing='ij')
                kX = kX - center_x
                kY = kY - center_y
                R2 = kX**2 + kY**2
                mask = np.exp(-R2 / (2 * sigma_mask**2))

                # Masked FFT
                fftZ_masked = fftZ * mask

                # Inverse FFT
                z_ifft = npf.ifft2(npf.ifftshift(fftZ_masked))

                # Unwrap phase
                phase_unwrapped = unwrap_phase(np.angle(z_ifft))

                # Calculate derivatives using np.diff (NOT gradient!)
                dx_phase = np.diff(phase_unwrapped, axis=1).flatten()
                dy_phase = np.diff(phase_unwrapped, axis=0).flatten()

                # Filter outliers using IQR
                filtered_dx = filter_outliers_iqr(dx_phase, threshold=0.7)
                filtered_dy = filter_outliers_iqr(dy_phase, threshold=0.7)

                # G_mismatch
                G_mismatch = np.array([filtered_dx, filtered_dy])
                mismatch_norm = np.linalg.norm(G_mismatch)

                # Update q
                q_x += G_mismatch[0]
                q_y += G_mismatch[1]

                # Check convergence
                if mismatch_norm < tol:
                    break

            # Convert back to pixel coordinates
            v_real_x = center_x + q_x * (cols / (2 * np.pi))
            v_real_y = center_y + q_y * (rows / (2 * np.pi))

            # Round to 4 decimal places
            v_real_x = round(v_real_x, 4)
            v_real_y = round(v_real_y, 4)

            # Update circle position
            circle.circle.center = (v_real_x, v_real_y)

            # Update bounding box if selected
            if circle.selected and circle.bbox is not None:
                circle.create_bounding_box()

        # Redraw canvas
        if hasattr(self, 'FFT_Canvas'):
            self.FFT_Canvas.draw()

    def run_gpa(self, phase_type='A'):
        """Run GPA calculation and update panels."""
        # Check if Bragg peaks are selected
        if len(self.bragg_peak_circles) < 2:
            # Show FFT dialog for peak selection
            self.show_fft_peak_selection_dialog()
            return

        # Apply auto-centering if enabled
        if self.gpa_auto_center:
            self.apply_auto_center_to_circles()

        # Apply peak refinement if enabled
        if self.gpa_refine_peaks:
            self.refine_peak_positions()

        # Get g-vectors from circles
        g_vectors = []
        for circle in self.bragg_peak_circles:
            center = circle.get_center()
            g_vectors.append((center[0], center[1]))

        # Validate that peaks are not orthogonal/opposite
        if len(g_vectors) >= 2:
            source_image = self.Z_original if self.Z_original is not None else self.Z
            rows, cols = source_image.shape
            center_x, center_y = cols // 2, rows // 2

            # Get vectors from center to peaks
            v1 = np.array([g_vectors[0][0] - center_x, g_vectors[0][1] - center_y])
            v2 = np.array([g_vectors[1][0] - center_x, g_vectors[1][1] - center_y])

            # Check if vectors are nearly opposite (180 degrees apart)
            # If opposite, dot product will be negative and close to -|v1||v2|
            dot_product = np.dot(v1, v2)
            mag_v1 = np.linalg.norm(v1)
            mag_v2 = np.linalg.norm(v2)

            if mag_v1 > 0 and mag_v2 > 0:
                cos_angle = dot_product / (mag_v1 * mag_v2)
                # Check if angle is close to 180 degrees (cos(180°) = -1)
                if cos_angle < -0.9:  # Approximately 154-180 degrees
                    QMessageBox.warning(
                        self,
                        "Invalid Peak Selection",
                        "Error: Selected Bragg peaks are nearly opposite (orthogonal).\n\n"
                        "Please select two peaks that are not on opposite sides of the center.\n"
                        "GPA requires non-opposite g-vectors for proper strain calculation."
                    )
                    return

        # Get mask radius from first circle
        mask_radius = self.bragg_peak_circles[0].get_radius()

        # Get source image
        source_image = self.Z_original if self.Z_original is not None else self.Z

        # Run GPA calculation
        if self.gpa_algorithm == 'standard':
            exx, eyy, exy, omega, phases = tools.standard_gpa(
                source_image, g_vectors, mask_radius, sigma=self.gpa_sigma
            )
        else:
            exx, eyy, exy, omega, phases = tools.adaptive_gpa(
                source_image, g_vectors, mask_radius, sigma=self.gpa_sigma
            )

        # Store results
        self.strain_xx = exx
        self.strain_yy = eyy
        self.strain_xy = exy
        self.rotation_map = omega

        # Store phase data
        if len(phases) >= 2:
            self.phase_a = phases[0]
            self.phase_b = phases[1]
        elif len(phases) == 1:
            self.phase_a = phases[0]
            self.phase_b = np.zeros_like(phases[0])
        else:
            self.phase_a = np.zeros_like(source_image)
            self.phase_b = np.zeros_like(source_image)

        # Extract amplitude maps using CDWampmap2D
        if len(g_vectors) >= 2:
            import AmpMap as am

            # Get reference position (atom center in real space)
            r0 = np.array([self.x0, self.y0])

            # Get g-vectors in pixel coordinates
            v1 = np.array([g_vectors[0][0], g_vectors[0][1]])
            v2 = np.array([g_vectors[1][0], g_vectors[1][1]])

            # Extract amplitude maps for both g-vectors
            # Amplitude A corresponds to v1
            amp_a_complex = am.CDWampmap2D(source_image, r0, v1, v2, self.gpa_sigma, 'plasma')
            self.amplitude_a = np.abs(amp_a_complex)

            # Amplitude B corresponds to v2
            amp_b_complex = am.CDWampmap2D(source_image, r0, v2, v1, self.gpa_sigma, 'plasma')
            self.amplitude_b = np.abs(amp_b_complex)
        else:
            self.amplitude_a = np.zeros_like(source_image)
            self.amplitude_b = np.zeros_like(source_image)

        # Update strain panels
        if hasattr(self, 'StrainXX_figure') and hasattr(self, 'StrainXX_Canvas'):
            self._refresh_panel(self.StrainXX_figure, self.StrainXX_Canvas, exx, show_axes=False)

        if hasattr(self, 'StrainYY_figure') and hasattr(self, 'StrainYY_Canvas'):
            self._refresh_panel(self.StrainYY_figure, self.StrainYY_Canvas, eyy, show_axes=False)

        # Update phase panels
        if hasattr(self, 'PhaseA_figure') and hasattr(self, 'PhaseA_Canvas'):
            self._refresh_panel(self.PhaseA_figure, self.PhaseA_Canvas, self.phase_a, show_axes=False)

        if hasattr(self, 'PhaseB_figure') and hasattr(self, 'PhaseB_Canvas'):
            self._refresh_panel(self.PhaseB_figure, self.PhaseB_Canvas, self.phase_b, show_axes=False)

    def show_fft_peak_selection_dialog(self):
        """Show dialog with FFT and circles for peak selection."""
        from PyQt5.QtWidgets import QVBoxLayout

        dialog = QDialog(self)
        dialog.setWindowTitle("Select Bragg Peaks")
        dialog.setModal(True)
        dialog.resize(600, 600)

        layout = QVBoxLayout()

        # Create matplotlib figure for FFT
        fig = plt.figure(figsize=(6, 6))
        canvas = FigureCanvas(fig)
        ax = fig.add_subplot()

        # Compute and display FFT
        source_image = self.Z_original if self.Z_original is not None else self.Z
        fft_result = npf.fftshift(npf.fft2(source_image - np.mean(source_image)))
        fft_magnitude = np.abs(fft_result)

        if self.fft_log_scale:
            data = np.log(fft_magnitude + 1)
        else:
            data = fft_magnitude

        ax.imshow(data, cmap=self.current_cmap)
        ax.set_title("Select Bragg Peaks (drag circles, scroll to resize)")
        ax.grid(False)

        # Add initial circles with automatic peak detection
        from skimage.feature import peak_local_max
        from scipy.ndimage import gaussian_filter

        rows, cols = fft_magnitude.shape
        center_y, center_x = rows // 2, cols // 2
        smoothed = gaussian_filter(fft_magnitude, sigma=2)
        peak_coords = peak_local_max(smoothed, min_distance=20, threshold_rel=0.3, num_peaks=10)

        # Filter out center peak
        selected_peaks = []
        for py, px in peak_coords:
            dist_from_center = np.sqrt((px - center_x)**2 + (py - center_y)**2)
            if dist_from_center > 15:
                selected_peaks.append((px, py))

        # Add circles
        temp_circles = []
        for px, py in selected_peaks[:2]:
            circle = DraggableCircle(ax, (px, py), radius=self.gpa_mask_radius, color='yellow', alpha=0.3)
            temp_circles.append(circle)

        # Apply auto-centering if enabled
        if self.gpa_auto_center:
            for circle in temp_circles:
                self.auto_center_circle(circle, fft_magnitude)

        canvas.draw()
        layout.addWidget(canvas)

        # Add OK/Cancel buttons
        from PyQt5.QtWidgets import QDialogButtonBox
        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        button_box.accepted.connect(dialog.accept)
        button_box.rejected.connect(dialog.reject)
        layout.addWidget(button_box)

        dialog.setLayout(layout)

        # If dialog accepted, transfer circles to main FFT
        if dialog.exec_() == QDialog.Accepted:
            # Clear existing circles
            self.clear_bragg_peak_circles()

            # Get FFT axes
            ax_fft = self.FFT_figure.axes[0] if self.FFT_figure.axes else None
            if ax_fft:
                # Transfer circles to main FFT
                for temp_circle in temp_circles:
                    center = temp_circle.get_center()
                    radius = temp_circle.get_radius()
                    new_circle = DraggableCircle(ax_fft, center, radius, color='yellow', alpha=0.3)
                    self.bragg_peak_circles.append(new_circle)

                self.FFT_Canvas.draw()

                # Enable the Bragg Peak Selection button
                if hasattr(self, 'bragg_peak_btn'):
                    self.bragg_peak_btn.setChecked(True)

    def toggle_all_attributes(self):
        self.attribute_visible = not getattr(self, "attribute_visible", False)

        for attr_name in self.attribute_visibility_states.keys():
            self.attribute_visibility_states[attr_name] = self.attribute_visible
            for widget in getattr(self, attr_name):
                widget.setVisible(self.attribute_visible)



    def toggleButtonStyle(self):
        return """
            QPushButton {
                background-color: #f1f1f1;
                border: 1px solid #ccc;
                border-radius: 5px;
                padding: 2px;
            }
            QPushButton:hover {
                background-color: #eaeaea;
                border-color: #bbb;
            }
            QPushButton:pressed {
                background-color: #ddd;
                border-color: #aaa;
            }
        """



    def iconStyle(self, button):
        button.setIconSize(QSize(25, 25))
        button.setFixedSize(30, 35)
        button.setFlat(True)
        button.setStyleSheet("""
                QPushButton {
                    background: transparent;
                    border: none;
                    border-radius: 10px;
                }
                QPushButton:hover {
                    background: #cdcdcd;
                }
                QPushButton:pressed {
                background-color: #c0c0c0;
                }
            """)
        

