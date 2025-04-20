import sys
from PyQt5.QtWidgets import (QApplication, QMainWindow, QLabel, QPushButton, 
                           QVBoxLayout, QHBoxLayout, QWidget, QFileDialog, QComboBox, QGroupBox, QScrollArea)
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from scipy import signal
from PyQt5.QtWidgets import QSpinBox
from PyQt5.QtWidgets import (QSlider, QLabel, QSpinBox, QDoubleSpinBox)
from PyQt5.QtCore import Qt

class HistogramWindow(QWidget):
    def __init__(self, image, title="Histogram"):
        super().__init__()
        self.setWindowTitle(title)
        self.setGeometry(200, 200, 600, 400)
        
        layout = QVBoxLayout()
        
        # Create matplotlib figure
        fig = Figure(figsize=(6, 4))
        canvas = FigureCanvas(fig)
        layout.addWidget(canvas)
        
        # Convert image to grayscale if it's not already
        if len(image.shape) == 3:
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray_image = image
            
        # Calculate histogram
        hist = cv2.calcHist([gray_image], [0], None, [256], [0, 256])
        
        # Plot histogram
        ax = fig.add_subplot(111)
        ax.plot(hist)
        ax.set_xlim([0, 256])
        ax.set_xlabel('Intensity Level')
        ax.set_ylabel('Number of Pixels')
        ax.set_title('Grayscale Histogram')
        ax.grid(True)
        
        self.setLayout(layout)

class ClickableLabel(QLabel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.image = None
        self.histogram_window = None  # Store histogram window reference
        
    def mouseDoubleClickEvent(self, event):
        if self.image is not None:
            self.show_histogram()
    
    def show_histogram(self):
        if self.image is not None:
            # Close previous histogram window if it exists
            if self.histogram_window is not None:
                self.histogram_window.close()
            
            # Create and store new histogram window
            self.histogram_window = HistogramWindow(self.image, f"Histogram - {self.text()}")
            self.histogram_window.show()

class MedicalImageViewer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Medical Image Viewer")
        self.setGeometry(100, 100, 1400, 400)
        
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QHBoxLayout()
        
        # Create scrollable sidebar
        sidebar_scroll = QScrollArea()
        sidebar_scroll.setWidgetResizable(True)
        sidebar_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        sidebar_scroll.setMinimumWidth(200)  # Set minimum width for sidebar
        
        # Create sidebar
        sidebar = QWidget()
        sidebar_layout = QVBoxLayout()
        sidebar_layout.setSpacing(10)

        # Load image button
        self.load_button = QPushButton("Load Image")
        sidebar_layout.addWidget(self.load_button)

        # Group for Source and Target Selection
        selection_group = QGroupBox("Viewport Selection")
        selection_layout = QVBoxLayout()
        
        self.source_combo = QComboBox()
        self.source_combo.addItems(["Original Image", "Result 1", "Result 2"])
        self.target_combo = QComboBox()
        self.target_combo.addItems(["Result 1", "Result 2"])
        
        selection_layout.addWidget(QLabel("Source:"))
        selection_layout.addWidget(self.source_combo)
        selection_layout.addWidget(QLabel("Target:"))
        selection_layout.addWidget(self.target_combo)
        selection_group.setLayout(selection_layout)
        sidebar_layout.addWidget(selection_group)

        # Group for Zoom Settings
        zoom_group = QGroupBox("Zoom Settings")
        zoom_layout = QVBoxLayout()
        
        self.resizing_factor_spinbox = QSpinBox()
        self.resizing_factor_spinbox.setRange(1, 10)  # Factor between 1x and 10x
        self.resizing_factor_spinbox.setValue(2)  # Default value: 2x
        zoom_layout.addWidget(QLabel("Zoom Factor:"))
        zoom_layout.addWidget(self.resizing_factor_spinbox)

        self.linear_zoom_button = QPushButton("Zoom (Linear)")
        self.cubic_zoom_button = QPushButton("Zoom (Cubic)")
        self.nearest_neighbor_zoom_button = QPushButton("Zoom (Nearest Neighbor)")
        
        zoom_buttons = [
            self.linear_zoom_button,
            self.cubic_zoom_button,
            self.nearest_neighbor_zoom_button,
        ]
        
        for button in zoom_buttons:
            button.setMinimumHeight(40)
            zoom_layout.addWidget(button)
        
        zoom_group.setLayout(zoom_layout)
        sidebar_layout.addWidget(zoom_group)

        # Group for Noise Addition Buttons
        noise_group = QGroupBox("Add Noise")
        noise_layout = QVBoxLayout()
        
        self.noise_button = QPushButton("Add Gaussian Noise")
        self.salt_pepper_button = QPushButton("Add Salt & Pepper Noise")
        self.speckle_button = QPushButton("Add Speckle Noise")
        
        noise_buttons = [
            self.noise_button,
            self.salt_pepper_button,
            self.speckle_button
        ]
        
        for button in noise_buttons:
            button.setMinimumHeight(40)
            noise_layout.addWidget(button)
        
        noise_group.setLayout(noise_layout)
        sidebar_layout.addWidget(noise_group)

        # Group for Filters
        filter_group = QGroupBox("Filters")
        filter_layout = QVBoxLayout()
        
        self.low_pass_button = QPushButton("Apply Low-Pass Filter")
        self.high_pass_button = QPushButton("Apply High-Pass Filter")
        self.wiener_button = QPushButton("Apply Wiener Filter")
        self.anisotropic_button = QPushButton("Apply Anisotropic Diffusion")
        self.denoise_button = QPushButton("Apply Mean Filter")
        
        filter_buttons = [
            self.low_pass_button,
            self.high_pass_button,
            self.wiener_button,
            self.anisotropic_button,
            self.denoise_button
        ]
        
        for button in filter_buttons:
            button.setMinimumHeight(40)
            filter_layout.addWidget(button)
        
        filter_group.setLayout(filter_layout)
        sidebar_layout.addWidget(filter_group)

        # Group for SNR and CNR Calculations
        snr_cnr_group = QGroupBox("SNR & CNR Calculations")
        snr_cnr_layout = QVBoxLayout()
        
        self.select_roi_button = QPushButton("Calculating SNR")
        self.select_cnr_button = QPushButton("Calculating CNR")
        
        snr_cnr_buttons = [
            self.select_roi_button,
            self.select_cnr_button
        ]
        
        for button in snr_cnr_buttons:
            button.setMinimumHeight(40)
            snr_cnr_layout.addWidget(button)
        
        snr_cnr_group.setLayout(snr_cnr_layout)
        sidebar_layout.addWidget(snr_cnr_group)

        # Add Reset Button at the Bottom
        self.reset_button = QPushButton("Reset")
        self.reset_button.setMinimumHeight(40)
        sidebar_layout.addWidget(self.reset_button)

        # Add sidebar layout to the scroll area
        sidebar.setLayout(sidebar_layout)
        sidebar_scroll.setWidget(sidebar)

        # Modify the viewports layout
        viewports_layout = QHBoxLayout()
        left_layout = QVBoxLayout()
        right_layout = QVBoxLayout()
        
        # Add original image to left side
        self.viewport1 = ClickableLabel("Original Image")
        left_layout.addWidget(self.viewport1)
        
        # Add result viewports to right side
        self.viewport2 = ClickableLabel("Result 1")
        self.viewport3 = ClickableLabel("Result 2")
        right_layout.addWidget(self.viewport2)
        right_layout.addWidget(self.viewport3)
        
        # Set up viewports dictionary and properties
        self.viewports = {
            "Original Image": self.viewport1,
            "Result 1": self.viewport2,
            "Result 2": self.viewport3
        }
        
        for viewport in self.viewports.values():
            viewport.setMinimumSize(380, 300)
            viewport.setAlignment(Qt.AlignCenter)
            viewport.setStyleSheet("border: 2px solid black")
        
        # Add layouts to main layout
        viewports_layout.addLayout(left_layout)
        viewports_layout.addLayout(right_layout)
        
        # Add layouts to main layout
        main_layout.addWidget(sidebar_scroll)
        main_layout.addLayout(viewports_layout)
        main_widget.setLayout(main_layout)

        # Connect buttons to their respective functions
        self.load_button.clicked.connect(self.load_image)
        self.noise_button.clicked.connect(lambda: self.apply_process(self.add_noise))
        self.salt_pepper_button.clicked.connect(lambda: self.apply_process(self.add_salt_pepper_noise))
        self.speckle_button.clicked.connect(lambda: self.apply_process(self.add_speckle_noise))
        self.select_roi_button.clicked.connect(self.select_rois)
        self.select_cnr_button.clicked.connect(self.select_cnr_rois)
        self.linear_zoom_button.clicked.connect(self.linear_zoom)
        self.cubic_zoom_button.clicked.connect(self.cubic_zoom)
        self.nearest_neighbor_zoom_button.clicked.connect(self.nearest_neighbor_zoom)
        self.reset_button.clicked.connect(self.reset_results)
        self.low_pass_button.clicked.connect(lambda: self.apply_process(self.apply_low_pass_filter))
        self.high_pass_button.clicked.connect(lambda: self.apply_process(self.apply_high_pass_filter))
        self.wiener_button.clicked.connect(lambda: self.apply_process(self.apply_wiener_filter))
        self.anisotropic_button.clicked.connect(lambda: self.apply_process(self.apply_anisotropic_diffusion))
        self.denoise_button.clicked.connect(lambda: self.apply_process(self.denoise))
        
        # Setup CNR controls
        self.setup_cnr_controls()
        
        # Store images and their states
        self.images = {
            "Original Image": None,
            "Result 1": None,
            "Result 2": None
        }
        
        # Store zoom centers for each image
        self.zoom_centers = {
            "Original Image": None,
            "Result 1": None,
            "Result 2": None
        }
        
        # Track which images have noise
        self.has_noise = {
            "Original Image": False,
            "Result 1": False,
            "Result 2": False
        }

    def setup_cnr_controls(self):
        """Setup CNR control widgets in the sidebar"""
        # Create CNR control group
        cnr_group = QGroupBox("CNR Controls")
        cnr_layout = QVBoxLayout()
        
        # Brightness control
        brightness_label = QLabel("Brightness:")
        self.brightness_slider = QSlider(Qt.Horizontal)
        self.brightness_slider.setRange(-100, 100)
        self.brightness_slider.setValue(0)
        self.brightness_slider.valueChanged.connect(lambda: self.apply_process(self.adjust_brightness_contrast))
        
        # Contrast control
        contrast_label = QLabel("Contrast:")
        self.contrast_slider = QSlider(Qt.Horizontal)
        self.contrast_slider.setRange(-100, 100)
        self.contrast_slider.setValue(0)
        self.contrast_slider.valueChanged.connect(lambda: self.apply_process(self.adjust_brightness_contrast))
        
        # Add contrast enhancement buttons
        self.histogram_eq_button = QPushButton("Histogram Equalization")
        self.histogram_eq_button.clicked.connect(lambda: self.apply_process(self.apply_histogram_equalization))
        
        self.clahe_button = QPushButton("CLAHE")
        self.clahe_button.clicked.connect(lambda: self.apply_process(self.apply_clahe))
        
        self.adaptive_eq_button = QPushButton("Adaptive Equalization")
        self.adaptive_eq_button.clicked.connect(lambda: self.apply_process(self.apply_adaptive_equalization))
        
        # Add widgets to layout
        cnr_layout.addWidget(brightness_label)
        cnr_layout.addWidget(self.brightness_slider)
        cnr_layout.addWidget(contrast_label)
        cnr_layout.addWidget(self.contrast_slider)
        cnr_layout.addWidget(self.histogram_eq_button)
        cnr_layout.addWidget(self.clahe_button)
        cnr_layout.addWidget(self.adaptive_eq_button)
        
        cnr_group.setLayout(cnr_layout)
        
        # Add to sidebar layout
        self.findChild(QVBoxLayout, name=None).addWidget(cnr_group)

    def adjust_brightness_contrast(self, image):
        """Adjust image brightness and contrast based on slider values"""
        if image is None:
            return None
            
        # Convert image to float32
        adjusted = image.astype(np.float32)
        
        # Get slider values
        brightness = self.brightness_slider.value()
        contrast = self.contrast_slider.value() / 50.0  # Scale contrast to reasonable range
        
        # Apply brightness
        adjusted += brightness
        
        # Apply contrast
        adjusted = adjusted * (1 + contrast)
        
        # Clip values to valid range
        adjusted = np.clip(adjusted, 0, 255)
        
        # Convert back to uint8
        adjusted = adjusted.astype(np.uint8)
        
        self.statusBar().showMessage(f"Brightness: {brightness}, Contrast: {contrast:.2f}")
        return adjusted

    def apply_histogram_equalization(self, image):
        """Apply global histogram equalization"""
        if len(image.shape) == 3:
            # Convert to YUV color space
            yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
            # Apply equalization to Y channel
            yuv[:,:,0] = cv2.equalizeHist(yuv[:,:,0])
            # Convert back to BGR
            equalized = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)
        else:
            equalized = cv2.equalizeHist(image)
        
        self.statusBar().showMessage("Histogram equalization applied")
        return equalized

    def apply_clahe(self, image):
        """Apply Contrast Limited Adaptive Histogram Equalization"""
        # Create CLAHE object
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        
        if len(image.shape) == 3:
            # Convert to LAB color space
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            # Apply CLAHE to L channel
            lab[:,:,0] = clahe.apply(lab[:,:,0])
            # Convert back to BGR
            enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        else:
            enhanced = clahe.apply(image)
        
        self.statusBar().showMessage("CLAHE applied")
        return enhanced

    def apply_adaptive_equalization(self, image):
        """Apply custom adaptive contrast enhancement"""
        if len(image.shape) == 3:
            # Convert to YUV color space
            yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
            y_channel = yuv[:,:,0]
        else:
            y_channel = image
        
        # Calculate local statistics
        kernel_size = (15, 15)
        local_mean = cv2.blur(y_channel, kernel_size)
        local_std = np.sqrt(cv2.blur(y_channel**2, kernel_size) - local_mean**2)
        
        # Enhance contrast based on local statistics
        alpha = 2.0  # Contrast enhancement factor
        enhanced = np.clip((y_channel - local_mean) * alpha + local_mean, 0, 255).astype(np.uint8)
        
        if len(image.shape) == 3:
            yuv[:,:,0] = enhanced
            enhanced = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)
        
        self.statusBar().showMessage("Adaptive contrast enhancement applied")
        return enhanced

    def select_rois(self):
        source_image_key = self.source_combo.currentText()
        source_image = self.images.get(source_image_key)

        if source_image is None:
            print("No image loaded in the source viewport!")
            return

        # Create a copy of the image for display
        image_to_show = source_image.copy()
        if len(image_to_show.shape) == 2:
            image_to_show = cv2.cvtColor(image_to_show, cv2.COLOR_GRAY2BGR)

        # Use OpenCV's selectROI function to select ROIs
        rois = []
        for i in range(2):  # Allow selection of two ROIs
            roi = cv2.selectROI(f"Select ROI {i + 1} - {source_image_key}", image_to_show, fromCenter=False, showCrosshair=True)
            if roi == (0, 0, 0, 0):  # Check if no ROI was selected
                print(f"ROI {i + 1} selection canceled.")
                cv2.destroyAllWindows()
                return
            rois.append(roi)
            # Draw rectangle on the display copy only
            cv2.rectangle(image_to_show, (roi[0], roi[1]), (roi[0] + roi[2], roi[1] + roi[3]), (0, 255, 0), 2)

        cv2.destroyAllWindows()

        # Calculate metrics using the original image
        roi_pixels = []
        for roi in rois:
            x, y, w, h = roi
            roi_pixels.append(source_image[y:y + h, x:x + w])

        # Calculate average intensity and standard deviation for the selected ROIs
        signal_mean = np.mean(roi_pixels[0])  # Average intensity for the first ROI (signal)
        noise_mean = np.mean(roi_pixels[1])   # Average intensity for the second ROI (noise)
        noise_std = np.std(roi_pixels[1])     # Standard deviation for the second ROI (noise)

        # Calculate SNR as the ratio of signal mean to (noise std dev - noise mean)
        if noise_std - noise_mean == 0:
            snr = float('inf')  # Handle division by zero, SNR is infinity if the denominator is 0
        else:
            snr = signal_mean / np.abs(noise_std - noise_mean)

        # Store values
        self.roi_1_avg_intensity = signal_mean
        self.roi_2_avg_intensity = noise_mean
        self.roi_2_noise_std = noise_std
        self.snr_value = snr

        # Display the SNR in a popup
        self.show_message(f"SNR Calculation\n\nSignal Mean (ROI 1): {signal_mean:.2f}\n"
                        f"Noise Mean (ROI 2): {noise_mean:.2f}\n"
                        f"Noise Std Dev (ROI 2): {noise_std:.2f}\n"
                        f"SNR: {snr:.2f}")

    def show_message(self, message):
        """Utility function to show a popup message."""
        from PyQt5.QtWidgets import QMessageBox
        msg_box = QMessageBox()
        msg_box.setIcon(QMessageBox.Information)
        msg_box.setWindowTitle("SNR Result")
        msg_box.setText(message)
        msg_box.exec_()

    def select_cnr_rois(self):
        source_image_key = self.source_combo.currentText()
        source_image = self.images.get(source_image_key)

        if source_image is None:
            print("No image loaded in the source viewport!")
            return

        # Create a copy of the image for display
        image_to_show = source_image.copy()
        if len(image_to_show.shape) == 2:
            image_to_show = cv2.cvtColor(image_to_show, cv2.COLOR_GRAY2BGR)

        # Use OpenCV's selectROI function to select ROIs
        rois = []
        for i in range(3):  # Allow selection of three ROIs
            roi = cv2.selectROI(f"Select ROI {i + 1} - {source_image_key}", image_to_show, fromCenter=False, showCrosshair=True)
            if roi == (0, 0, 0, 0):  # Check if no ROI was selected
                print(f"ROI {i + 1} selection canceled.")
                cv2.destroyAllWindows()
                return
            rois.append(roi)
            # Draw rectangle on the display copy only
            cv2.rectangle(image_to_show, (roi[0], roi[1]), (roi[0] + roi[2], roi[1] + roi[3]), (0, 255, 0), 2)

        cv2.destroyAllWindows()

        # Calculate metrics using the original image
        roi_pixels = []
        for roi in rois:
            x, y, w, h = roi
            roi_pixels.append(source_image[y:y + h, x:x + w])

        # Calculate average intensity and standard deviation for the selected ROIs
        signal1_mean = np.mean(roi_pixels[0])  # Average intensity for the first ROI (signal 1)
        signal2_mean = np.mean(roi_pixels[1])  # Average intensity for the second ROI (signal 2)
        noise_mean = np.mean(roi_pixels[2])   # Average intensity for the third ROI (noise)
        noise_std = np.std(roi_pixels[2])     # Standard deviation for the third ROI (noise)

        # Calculate CNR as the ratio of signal contrast to noise standard deviation
        signal_contrast = np.abs(signal1_mean - signal2_mean)
        if noise_std == 0:
            cnr = float('inf')  # Handle division by zero, CNR is infinity if the denominator is 0
        else:
            cnr = signal_contrast / noise_std

        # Display the CNR in a popup
        self.show_message(f"CNR Calculation\n\nSignal 1 Mean (ROI 1): {signal1_mean:.2f}\n"
                        f"Signal 2 Mean (ROI 2): {signal2_mean:.2f}\n"
                        f"Noise Mean (ROI 3): {noise_mean:.2f}\n"
                        f"Noise Std Dev (ROI 3): {noise_std:.2f}\n"
                        f"CNR: {cnr:.2f}")

    def linear_zoom(self):
        """
        Apply zoom using linear interpolation with the factor from the spin box.
        """
        source_name = self.source_combo.currentText()
        target_name = self.target_combo.currentText()
        
        source_image = self.images[source_name]
        if source_image is None:
            self.statusBar().showMessage("No image to zoom.")
            return
        
        try:
            # Get original dimensions
            height, width = source_image.shape[:2]
            factor = self.resizing_factor_spinbox.value()
            
            # Get the center point for zooming
            center_x, center_y = self.zoom_centers[source_name] if self.zoom_centers[source_name] else (width // 2, height // 2)
            
            # Calculate the region to zoom into
            zoom_width = width // factor
            zoom_height = height // factor
            
            # Calculate region boundaries
            x1 = int(max(0, center_x - zoom_width // 2))
            y1 = int(max(0, center_y - zoom_height // 2))
            x2 = int(min(width, x1 + zoom_width))
            y2 = int(min(height, y1 + zoom_height))
            
            # Extract the region
            region = source_image[y1:y2, x1:x2]
            
            # Resize back to original dimensions using linear interpolation
            zoomed_image = cv2.resize(region, (width, height), interpolation=cv2.INTER_LINEAR)
            
            # Update target image and display
            self.images[target_name] = zoomed_image
            self.display_image(zoomed_image, self.viewports[target_name])
            
            # Update zoom center for the target
            self.zoom_centers[target_name] = (width // 2, height // 2)
            
            self.statusBar().showMessage(f"Linear interpolation zoom applied with factor {factor}x")
            
        except Exception as e:
            self.statusBar().showMessage(f"Error during linear zoom: {str(e)}")

    def cubic_zoom(self):
        """
        Apply zoom using cubic interpolation with the factor from the spin box.
        """
        source_name = self.source_combo.currentText()
        target_name = self.target_combo.currentText()
        
        source_image = self.images[source_name]
        if source_image is None:
            self.statusBar().showMessage("No image to zoom.")
            return
        
        try:
            # Get original dimensions
            height, width = source_image.shape[:2]
            factor = self.resizing_factor_spinbox.value()
            
            # Get the center point for zooming
            center_x, center_y = self.zoom_centers[source_name] if self.zoom_centers[source_name] else (width // 2, height // 2)
            
            # Calculate the region to zoom into
            zoom_width = width // factor
            zoom_height = height // factor
            
            # Calculate region boundaries
            x1 = int(max(0, center_x - zoom_width // 2))
            y1 = int(max(0, center_y - zoom_height // 2))
            x2 = int(min(width, x1 + zoom_width))
            y2 = int(min(height, y1 + zoom_height))
            
            # Extract the region
            region = source_image[y1:y2, x1:x2]
            
            # Resize back to original dimensions using cubic interpolation
            zoomed_image = cv2.resize(region, (width, height), interpolation=cv2.INTER_CUBIC)
            
            # Update target image and display
            self.images[target_name] = zoomed_image
            self.display_image(zoomed_image, self.viewports[target_name])
            
            # Update zoom center for the target
            self.zoom_centers[target_name] = (width // 2, height // 2)
            
            self.statusBar().showMessage(f"Cubic interpolation zoom applied with factor {factor}x")
            
        except Exception as e:
            self.statusBar().showMessage(f"Error during cubic zoom: {str(e)}")

    def nearest_neighbor_zoom(self):
        """
        Apply nearest-neighbor zoom that focuses on the center region of the image
        and enlarges it using nearest-neighbor interpolation.
        """
        source_name = self.source_combo.currentText()
        target_name = self.target_combo.currentText()
        
        source_image = self.images[source_name]
        if source_image is None:
            self.statusBar().showMessage("No image to zoom.")
            return
        
        try:
            # Get original dimensions
            height, width = source_image.shape[:2]
            factor = self.resizing_factor_spinbox.value()
            
            # Get the center point for zooming
            center_x, center_y = self.zoom_centers[source_name] if self.zoom_centers[source_name] else (width // 2, height // 2)
            
            # Calculate the region to zoom into
            zoom_width = width // factor
            zoom_height = height // factor
            
            # Calculate region boundaries
            x1 = int(max(0, center_x - zoom_width // 2))
            y1 = int(max(0, center_y - zoom_height // 2))
            x2 = int(min(width, x1 + zoom_width))
            y2 = int(min(height, y1 + zoom_height))
            
            # Extract the region
            region = source_image[y1:y2, x1:x2]
            
            # Resize back to original dimensions using nearest neighbor interpolation
            zoomed_image = cv2.resize(region, (width, height), interpolation=cv2.INTER_NEAREST)
            
            # Update target image and display
            self.images[target_name] = zoomed_image
            self.display_image(zoomed_image, self.viewports[target_name])
            
            # Update zoom center for the target
            self.zoom_centers[target_name] = (width // 2, height // 2)
            
            self.statusBar().showMessage(f"Nearest-neighbor zoom applied with factor {factor}x")
            
        except Exception as e:
            self.statusBar().showMessage(f"Error during nearest-neighbor zoom: {str(e)}")

    def apply_low_pass_filter(self, image):
        """
        Applies a low-pass filter (Gaussian blur) to smooth the image.
        """
        try:
            kernel_size = (5, 5)  # Kernel size for smoothing
            low_passed = cv2.GaussianBlur(image, kernel_size, 0)
            self.statusBar().showMessage("Low-pass filter applied successfully.")
            return low_passed
        except Exception as e:
            self.statusBar().showMessage(f"Error applying low-pass filter: {str(e)}")
            return image

    def apply_high_pass_filter(self, image):
        """
        Applies a high-pass filter to enhance edges in the image.
        """
        try:
            # Create a high-pass filter kernel
            kernel = np.array([[-1, -1, -1],
                            [-1,  8, -1],
                            [-1, -1, -1]])
            high_passed = cv2.filter2D(image, -1, kernel)
            self.statusBar().showMessage("High-pass filter applied successfully.")
            return high_passed
        except Exception as e:
            self.statusBar().showMessage(f"Error applying high-pass filter: {str(e)}")
            return image
    def apply_wiener_filter(self, image):
        """
        Applies Wiener filter optimized for speckle noise removal.
        Uses local statistics to estimate noise variance.
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            is_color = True
        else:
            gray = image
            is_color = False
        
        # Convert to float for processing
        img_float = gray.astype(np.float32) / 255.0
        
        # Estimate local mean and variance
        kernel_size = (5, 5)
        local_mean = cv2.blur(img_float, kernel_size)
        local_var = cv2.blur(img_float**2, kernel_size) - local_mean**2
        
        # Estimate noise variance (assuming speckle noise)
        noise_var = np.mean(local_var)
        
        # Apply Wiener filter
        K = 1.0  # Noise to signal ratio parameter
        filtered = local_mean + ((local_var - noise_var) / (local_var + K * noise_var)) * (img_float - local_mean)
        
        # Convert back to uint8
        filtered = np.clip(filtered * 255, 0, 255).astype(np.uint8)
        
        if is_color:
            filtered = cv2.cvtColor(filtered, cv2.COLOR_GRAY2BGR)
        
        self.statusBar().showMessage("Speckle-optimized Wiener filter applied successfully.")
        return filtered

    def apply_anisotropic_diffusion(self, image):
        """
        Applies Perona-Malik anisotropic diffusion filtering to the image.
        This helps in noise reduction while preserving edges.
        
        Parameters:
            image: Input image
            num_iter: Number of iterations (default = 20)
            kappa: Conduction coefficient (20-100)
            gamma: Rate of diffusion (<=0.25 for stability)
        """
        # Convert image to float32
        img_float = image.astype(np.float32)
        
        # Parameters
        num_iter = 20
        kappa = 50
        gamma = 0.15
        
        # Initialize output
        filtered_image = img_float.copy()
        
        # Convert to grayscale if image is colored
        if len(image.shape) == 3:
            filtered_image = cv2.cvtColor(filtered_image, cv2.COLOR_BGR2GRAY)
            is_color = True
        else:
            is_color = False
            
        # Calculate 2D variations in 8 directions
        dx = 1
        dy = 1
        dd = np.sqrt(2)
        
        # Create conduction gradient functions
        def g1(x, k):
            return np.exp(-(x/k)**2)
            
        for _ in range(num_iter):
            # Calculate gradients
            north = np.roll(filtered_image, -dy, axis=0) - filtered_image
            south = np.roll(filtered_image, dy, axis=0) - filtered_image
            east = np.roll(filtered_image, dx, axis=1) - filtered_image
            west = np.roll(filtered_image, -dx, axis=1) - filtered_image
            northeast = np.roll(np.roll(filtered_image, -dy, axis=0), dx, axis=1) - filtered_image
            northwest = np.roll(np.roll(filtered_image, -dy, axis=0), -dx, axis=1) - filtered_image
            southeast = np.roll(np.roll(filtered_image, dy, axis=0), dx, axis=1) - filtered_image
            southwest = np.roll(np.roll(filtered_image, dy, axis=0), -dx, axis=1) - filtered_image
            
            # Calculate conduction gradients
            c_north = g1(north, kappa)
            c_south = g1(south, kappa)
            c_east = g1(east, kappa)
            c_west = g1(west, kappa)
            c_northeast = g1(northeast, kappa)
            c_northwest = g1(northwest, kappa)
            c_southeast = g1(southeast, kappa)
            c_southwest = g1(southwest, kappa)
            
            # Update image
            filtered_image = filtered_image + gamma * (
                (1/dd)*c_northeast * northeast + (1/dx)*c_east * east + 
                (1/dd)*c_southeast * southeast + (1/dy)*c_south * south + 
                (1/dd)*c_southwest * southwest + (1/dx)*c_west * west + 
                (1/dd)*c_northwest * northwest + (1/dy)*c_north * north
            )
        
        # Normalize and convert back to uint8
        filtered_image = np.clip(filtered_image, 0, 255).astype(np.uint8)
        
        # If original image was in color, convert result back to color
        if is_color:
            filtered_image = cv2.cvtColor(filtered_image, cv2.COLOR_GRAY2BGR)
        
        self.statusBar().showMessage("Anisotropic diffusion filtering completed successfully.")
        return filtered_image

    def add_speckle_noise(self, image):
        """
        Adds speckle noise to the image.
        Speckle noise is multiplicative noise that follows a gamma distribution.
        """
        # Convert image to float32 for processing
        image_float = image.astype(np.float32) / 255.0
        
        # Generate random noise with gamma distribution
        noise_factor = 0.25  # Controls the intensity of speckle noise
        noise = np.random.gamma(1, noise_factor, image.shape)
        
        # Apply multiplicative noise
        noisy_image = image_float * noise
        
        # Normalize and convert back to uint8
        noisy_image = np.clip(noisy_image * 255, 0, 255).astype(np.uint8)
        
        self.statusBar().showMessage("Speckle noise added successfully.")
        return noisy_image

    def add_salt_pepper_noise(self, image):
        """
        Adds salt and pepper noise to the image.
        Salt is white (255) and pepper is black (0).
        """
        # Make a copy of the image
        noisy_image = image.copy()
        
        # Salt and pepper probability
        prob_salt = 0.005
        prob_pepper = 0.005
        
        # Generate random numbers for salt and pepper
        salt_mask = np.random.random(image.shape[:2]) < prob_salt
        pepper_mask = np.random.random(image.shape[:2]) < prob_pepper
        
        # Apply salt (white) noise
        noisy_image[salt_mask] = 255
        
        # Apply pepper (black) noise
        noisy_image[pepper_mask] = 0
        
        self.statusBar().showMessage("Salt and pepper noise added successfully.")
        return noisy_image

    def load_image(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Open Image", "", 
                                                 "Image Files (*.png *.jpg *.jpeg *.bmp)")
        if file_path:
            self.images["Original Image"] = cv2.imread(file_path)
            self.display_image(self.images["Original Image"], self.viewport1)
            self.reset_results()
            
            # Initialize zoom center for original image
            height, width = self.images["Original Image"].shape[:2]
            self.zoom_centers["Original Image"] = (width // 2, height // 2)
            self.has_noise["Original Image"] = False
    
    def reset_results(self):
        self.images["Result 1"] = None
        self.images["Result 2"] = None
        self.zoom_centers["Result 1"] = None
        self.zoom_centers["Result 2"] = None
        self.has_noise["Result 1"] = False
        self.has_noise["Result 2"] = False
        self.viewport2.clear()
        self.viewport2.setText("Result 1")
        self.viewport3.clear()
        self.viewport3.setText("Result 2")
    
    def apply_process(self, process_func):
        source_name = self.source_combo.currentText()
        target_name = self.target_combo.currentText()
        
        source_image = self.images[source_name]
        if source_image is None:
            return
            
        result = process_func(source_image)
        self.images[target_name] = result
        
        # Copy zoom center from source to target
        if self.zoom_centers[source_name] is not None:
            self.zoom_centers[target_name] = self.zoom_centers[source_name]
        else:
            height, width = result.shape[:2]
            self.zoom_centers[target_name] = (width // 2, height // 2)
        
        # Update noise state for the target image
        if process_func == self.add_noise:
            self.has_noise[target_name] = True
        elif process_func == self.denoise:
            self.has_noise[target_name] = False
        else:
            self.has_noise[target_name] = self.has_noise[source_name]
            
        target_viewport = self.viewports[target_name]
        self.display_image(result, target_viewport)
    
    
    def add_noise(self, image):
        """
        Adds Gaussian noise to the image.
        Noise is scaled to ensure balanced effect.
        """
        mean = 0
        stddev = 50  # Standard deviation for noise intensity

        # Generate Gaussian noise
        noise = np.random.normal(mean, stddev, image.shape).astype(np.float32)

        # Add noise to the original image
        noisy_image = image.astype(np.float32) + noise

        # Clip the pixel values to stay within valid range [0, 255]
        noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)

        self.statusBar().showMessage("Gaussian noise added successfully.")
        return noisy_image


    
    def denoise(self, image):
        """
        Enhanced denoising function that handles both Gaussian and Salt & Pepper noise.
        Uses median filter for Salt & Pepper and mean filter for Gaussian noise.
        """
        source_name = self.source_combo.currentText()

        # If no noise was added, skip processing
        if not self.has_noise[source_name]:
            self.statusBar().showMessage("No noise detected to remove.")
            return image

        try:
            # Apply both median and mean filtering
            # Median filter is particularly good at removing salt and pepper noise
            # while preserving edges
            denoised_image = cv2.medianBlur(image, 3)  # Start with median filter
            denoised_image = cv2.blur(denoised_image, (3, 3))  # Follow with mean filter

            self.statusBar().showMessage("Denoising completed successfully.")
            self.has_noise[source_name] = False

            return denoised_image

        except Exception as e:
            self.statusBar().showMessage(f"Error during denoising: {str(e)}")
            return image




    
    def display_image(self, image, viewport):
        if image is None:
            return
            
        height, width = image.shape[:2]
        viewport_size = viewport.size()
        
        # Set a fixed maximum size for display
        max_display_width = 380  # Match the minimum size we set for viewports
        max_display_height = 300
        
        # Calculate scaling factor while maintaining aspect ratio
        scale = min(max_display_width / width, max_display_height / height)
        new_width = int(width * scale)
        new_height = int(height * scale)
        
        # Resize the image for display only
        resized = cv2.resize(image, (new_width, new_height))
        rgb_image = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        qt_image = QImage(rgb_image.data, w, h, w * ch, QImage.Format_RGB888)
        viewport.setPixmap(QPixmap.fromImage(qt_image))
        viewport.image = image  # Store the original image for histogram calculation

    def select_cnr_rois(self):
        source_image_key = self.source_combo.currentText()
        source_image = self.images.get(source_image_key)

        if source_image is None:
            print("No image loaded in the source viewport!")
            return

        # Create a copy of the image for display
        image_to_show = source_image.copy()
        if len(image_to_show.shape) == 2:
            image_to_show = cv2.cvtColor(image_to_show, cv2.COLOR_GRAY2BGR)

        # Use OpenCV's selectROI function to select ROIs
        rois = []
        for i in range(3):  # Allow selection of three ROIs
            roi = cv2.selectROI(f"Select ROI {i + 1} - {source_image_key}", image_to_show, fromCenter=False, showCrosshair=True)
            if roi == (0, 0, 0, 0):  # Check if no ROI was selected
                print(f"ROI {i + 1} selection canceled.")
                cv2.destroyAllWindows()
                return
            rois.append(roi)
            # Draw rectangle on the display copy only
            cv2.rectangle(image_to_show, (roi[0], roi[1]), (roi[0] + roi[2], roi[1] + roi[3]), (0, 255, 0), 2)

        cv2.destroyAllWindows()

        # Calculate metrics using the original image
        roi_pixels = []
        for roi in rois:
            x, y, w, h = roi
            roi_pixels.append(source_image[y:y + h, x:x + w])

        # Calculate average intensity and standard deviation for the selected ROIs
        signal1_mean = np.mean(roi_pixels[0])  # Average intensity for the first ROI (signal 1)
        signal2_mean = np.mean(roi_pixels[1])  # Average intensity for the second ROI (signal 2)
        noise_mean = np.mean(roi_pixels[2])   # Average intensity for the third ROI (noise)
        noise_std = np.std(roi_pixels[2])     # Standard deviation for the third ROI (noise)

        # Calculate CNR as the ratio of signal contrast to noise standard deviation
        signal_contrast = np.abs(signal1_mean - signal2_mean)
        if noise_std == 0:
            cnr = float('inf')  # Handle division by zero, CNR is infinity if the denominator is 0
        else:
            cnr = signal_contrast / noise_std

        # Display the CNR in a popup
        self.show_message(f"CNR Calculation\n\nSignal 1 Mean (ROI 1): {signal1_mean:.2f}\n"
                        f"Signal 2 Mean (ROI 2): {signal2_mean:.2f}\n"
                        f"Noise Mean (ROI 3): {noise_mean:.2f}\n"
                        f"Noise Std Dev (ROI 3): {noise_std:.2f}\n"
                        f"CNR: {cnr:.2f}")

    def show_message(self, message):
        """Utility function to show CNR results."""
        from PyQt5.QtWidgets import QMessageBox
        msg_box = QMessageBox()
        msg_box.setIcon(QMessageBox.Information)
        msg_box.setWindowTitle("CNR Result")
        msg_box.setText(message)
        msg_box.exec_()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    viewer = MedicalImageViewer()
    viewer.show()
    sys.exit(app.exec_())
