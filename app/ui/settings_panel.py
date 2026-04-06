from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QGridLayout,
    QGroupBox,
    QLabel,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from app.config import config


class SettingsPanel(QWidget):
    """Settings panel with configuration options."""

    settings_applied = Signal(dict)

    def __init__(self, parent=None):
        super().__init__(parent)

        layout = QVBoxLayout()
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)

        # Model settings group
        model_group = QGroupBox("Recognition Settings")

        model_layout = QGridLayout()
        model_layout.setVerticalSpacing(8)
        model_layout.setHorizontalSpacing(10)

        # Model selection
        model_layout.addWidget(QLabel("Model:"), 0, 0)
        self.model_combo = QComboBox()
        models = ["VGG-Face", "Facenet", "Facenet512", "OpenFace", "DeepFace", "ArcFace", "Dlib"]
        self.model_combo.addItems(models)
        self.model_combo.setCurrentText(config.MODEL_NAME)
        model_layout.addWidget(self.model_combo, 0, 1)

        # Detector selection
        model_layout.addWidget(QLabel("Detector:"), 1, 0)
        self.detector_combo = QComboBox()
        detectors = ["opencv", "ssd", "dlib", "mtcnn", "retinaface", "mediapipe", "yolov8"]
        self.detector_combo.addItems(detectors)
        self.detector_combo.setCurrentText(config.DETECTOR)
        model_layout.addWidget(self.detector_combo, 1, 1)

        # Threshold
        model_layout.addWidget(QLabel("Threshold:"), 2, 0)
        self.threshold_spin = QDoubleSpinBox()
        self.threshold_spin.setRange(0.1, 15.0)  # cosine 0.3-0.6, euclidean 3-8
        self.threshold_spin.setDecimals(2)
        self.threshold_spin.setSingleStep(0.1)
        self.threshold_spin.setValue(config.SIMILARITY_THRESHOLD)
        model_layout.addWidget(self.threshold_spin, 2, 1)

        # Metric
        model_layout.addWidget(QLabel("Metric:"), 3, 0)
        self.metric_combo = QComboBox()
        self.metric_combo.addItems(["cosine", "euclidean", "euclidean_l2"])
        self.metric_combo.setCurrentText(config.DISTANCE_METRIC)
        model_layout.addWidget(self.metric_combo, 3, 1)

        model_group.setLayout(model_layout)
        layout.addWidget(model_group)

        # Performance settings
        perf_group = QGroupBox("Performance")

        perf_layout = QGridLayout()
        perf_layout.setVerticalSpacing(8)

        perf_layout.addWidget(QLabel("Process interval:"), 0, 0)
        self.interval_spin = QSpinBox()
        self.interval_spin.setRange(1, 10)
        self.interval_spin.setValue(config.PROCESS_INTERVAL)
        perf_layout.addWidget(self.interval_spin, 0, 1)

        perf_layout.addWidget(QLabel("Detection scale:"), 1, 0)
        self.scale_spin = QDoubleSpinBox()
        self.scale_spin.setRange(0.25, 1.0)
        self.scale_spin.setSingleStep(0.05)
        self.scale_spin.setValue(config.DETECTION_SCALE)
        perf_layout.addWidget(self.scale_spin, 1, 1)

        perf_layout.addWidget(QLabel("Max faces:"), 2, 0)
        self.max_faces_spin = QSpinBox()
        self.max_faces_spin.setRange(1, 10)
        self.max_faces_spin.setSpecialValueText("All")
        self.max_faces_spin.setValue(config.MAX_FACES_TO_CHECK if config.MAX_FACES_TO_CHECK else 5)
        perf_layout.addWidget(self.max_faces_spin, 2, 1)

        # Async processing checkbox
        self.async_check = QCheckBox("Async processing")
        self.async_check.setChecked(config.ASYNC_PROCESSING)
        perf_layout.addWidget(self.async_check, 3, 0, 1, 2)

        perf_group.setLayout(perf_layout)
        layout.addWidget(perf_group)

        # Face box expansion
        expand_group = QGroupBox("Face Box Expansion")

        expand_layout = QGridLayout()
        expand_layout.setVerticalSpacing(8)

        self.expand_check = QCheckBox("Enable expansion")
        self.expand_check.setChecked(config.EXPAND_FACE_BOX)
        expand_layout.addWidget(self.expand_check, 0, 0, 1, 2)

        expand_layout.addWidget(QLabel("Expand factor:"), 1, 0)
        self.expand_factor_spin = QDoubleSpinBox()
        self.expand_factor_spin.setRange(1.0, 2.5)
        self.expand_factor_spin.setSingleStep(0.1)
        self.expand_factor_spin.setValue(config.FACE_BOX_EXPAND_FACTOR)
        expand_layout.addWidget(self.expand_factor_spin, 1, 1)

        expand_layout.addWidget(QLabel("Headroom:"), 2, 0)
        self.headroom_spin = QDoubleSpinBox()
        self.headroom_spin.setRange(0.0, 0.3)
        self.headroom_spin.setSingleStep(0.05)
        self.headroom_spin.setValue(config.FACE_BOX_HEADROOM)
        expand_layout.addWidget(self.headroom_spin, 2, 1)

        expand_group.setLayout(expand_layout)
        layout.addWidget(expand_group)

        # Detected panel
        panel_group = QGroupBox("Detected Panel")
        panel_layout = QGridLayout()
        panel_layout.addWidget(QLabel("Face thumbnail size (px):"), 0, 0)
        self.thumb_size_spin = QSpinBox()
        self.thumb_size_spin.setRange(60, 200)
        self.thumb_size_spin.setSingleStep(10)
        self.thumb_size_spin.setValue(getattr(config, "DETECTED_FACE_THUMB_SIZE", 100))
        panel_layout.addWidget(self.thumb_size_spin, 0, 1)
        panel_group.setLayout(panel_layout)
        layout.addWidget(panel_group)

        # Apply button
        self.apply_btn = QPushButton("Apply Settings")
        self.apply_btn.clicked.connect(self.apply_settings)
        layout.addWidget(self.apply_btn)

        # Reset to defaults button
        self.reset_btn = QPushButton("Reset to Defaults")
        self.reset_btn.clicked.connect(self.reset_to_defaults)
        layout.addWidget(self.reset_btn)

        layout.addStretch()
        self.setLayout(layout)

    def apply_settings(self):
        """Apply settings and emit signal."""
        settings = {
            "model_name": self.model_combo.currentText(),
            "detector": self.detector_combo.currentText(),
            "threshold": round(self.threshold_spin.value(), 2),
            "metric": self.metric_combo.currentText(),
            "process_interval": self.interval_spin.value(),
            "detection_scale": self.scale_spin.value(),
            "max_faces": self.max_faces_spin.value(),
            "async_mode": self.async_check.isChecked(),
            "expand_enabled": self.expand_check.isChecked(),
            "expand_factor": self.expand_factor_spin.value(),
            "headroom": self.headroom_spin.value(),
            "detected_face_thumb_size": self.thumb_size_spin.value(),
        }
        self.settings_applied.emit(settings)

    def reset_to_defaults(self):
        """Reset settings to default values from config."""
        self.model_combo.setCurrentText(config.MODEL_NAME)
        self.detector_combo.setCurrentText(config.DETECTOR)
        self.threshold_spin.setValue(config.SIMILARITY_THRESHOLD)
        self.metric_combo.setCurrentText(config.DISTANCE_METRIC)
        self.interval_spin.setValue(config.PROCESS_INTERVAL)
        self.scale_spin.setValue(config.DETECTION_SCALE)
        self.max_faces_spin.setValue(config.MAX_FACES_TO_CHECK if config.MAX_FACES_TO_CHECK else 5)
        self.async_check.setChecked(config.ASYNC_PROCESSING)
        self.expand_check.setChecked(config.EXPAND_FACE_BOX)
        self.expand_factor_spin.setValue(config.FACE_BOX_EXPAND_FACTOR)
        self.headroom_spin.setValue(config.FACE_BOX_HEADROOM)
        self.thumb_size_spin.setValue(getattr(config, "DETECTED_FACE_THUMB_SIZE", 100))


