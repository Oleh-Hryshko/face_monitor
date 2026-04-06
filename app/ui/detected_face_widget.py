import cv2
import numpy as np
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor, QImage, QPixmap
from PySide6.QtWidgets import QFrame, QHBoxLayout, QLabel, QPushButton, QSizePolicy, QVBoxLayout

from app.config import config
from app.ui.overlay import (
    build_detected_face_text_section,
    calculate_detected_card_min_height,
)


class DetectedFaceWidget(QFrame):
    """Widget for displaying a detected face in the panel."""

    # Signal for removal
    remove_clicked = Signal(str)  # Passes the name of the face to remove

    def __init__(self, name, face_img, thumb_img, detection_time, info="", parent=None):
        super().__init__(parent)

        self.name = name
        self.detection_time = detection_time
        self.setFrameStyle(QFrame.StyledPanel | QFrame.Raised)
        self.setLineWidth(1)

        # Get thumbnail size from config
        self.thumb_size = config.DETECTED_FACE_THUMB_SIZE

        # Keep room for text below thumbnails to prevent overlap.
        self.setMinimumHeight(calculate_detected_card_min_height(self.thumb_size, info))

        self.setStyleSheet(
            """
            QFrame {
                background-color: #2d2d2d;
                border: 1px solid #3d3d3d;
                border-radius: 5px;
                margin: 2px;
            }
            QFrame:hover {
                background-color: #3d3d3d;
                border: 1px solid #ffaa00;
            }
            """
        )

        # Use QVBoxLayout as main layout
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(8, 8, 8, 8)
        main_layout.setSpacing(8)
        main_layout.setSizeConstraint(QVBoxLayout.SetMinimumSize)

        # Row 1: two photos
        thumb_layout = QHBoxLayout()
        thumb_layout.setSpacing(8)

        # Left image (detected face)
        if face_img is not None and face_img.size > 0:
            rgb_face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
            face_pixmap = self.cv_to_pixmap(rgb_face_img, self.thumb_size, self.thumb_size)
            face_label = QLabel()
            face_label.setPixmap(face_pixmap)
            face_label.setFixedSize(self.thumb_size, self.thumb_size)
            face_label.setAlignment(Qt.AlignCenter)
            face_label.setStyleSheet("border: 1px solid #444; background-color: #1e1e1e; border-radius: 3px;")
            thumb_layout.addWidget(face_label)
        else:
            placeholder = QLabel()
            placeholder.setFixedSize(self.thumb_size, self.thumb_size)
            placeholder.setStyleSheet("border: 1px solid #444; background-color: #1e1e1e; border-radius: 3px;")
            thumb_layout.addWidget(placeholder)

        # Right image (watchlist photo)
        if thumb_img is not None and thumb_img.size > 0:
            thumb_pixmap = self.cv_to_pixmap(thumb_img, self.thumb_size, self.thumb_size)
            thumb_label = QLabel()
            thumb_label.setPixmap(thumb_pixmap)
            thumb_label.setFixedSize(self.thumb_size, self.thumb_size)
            thumb_label.setAlignment(Qt.AlignCenter)
            thumb_label.setStyleSheet("border: 1px solid #c44; background-color: #1e1e1e; border-radius: 3px;")
            thumb_layout.addWidget(thumb_label)
        else:
            placeholder = QLabel()
            placeholder.setFixedSize(self.thumb_size, self.thumb_size)
            placeholder.setStyleSheet("border: 1px solid #c44; background-color: #1e1e1e; border-radius: 3px;")
            thumb_layout.addWidget(placeholder)

        thumb_layout.addStretch()
        main_layout.addLayout(thumb_layout)

        text_section = build_detected_face_text_section(
            name=name,
            info_text=info,
            detection_time=detection_time,
            thumb_size=self.thumb_size,
        )
        main_layout.addWidget(text_section)

        # Delete button (X) in the top right corner
        # Scale button size based on thumbnail size
        button_size = max(16, min(24, int(self.thumb_size / 5)))
        button_x = self.width() - (button_size + 5)

        self.delete_button = QPushButton("×", self)
        self.delete_button.setGeometry(button_x, 5, button_size, button_size)

        # Button styles with scaled font
        button_font_size = max(10, min(16, int(self.thumb_size / 7)))
        self.delete_button.setStyleSheet(f"""
            QPushButton {{
                background-color: #c44;
                color: white;
                border: 1px solid #a33;
                border-radius: 3px;
                font-size: {button_font_size}px;
                font-weight: bold;
                padding: 0px;
                margin: 0px;
            }}
            QPushButton:hover {{
                background-color: #d55;
                border: 1px solid #b44;
            }}
            QPushButton:pressed {{
                background-color: #b33;
            }}
        """)
        self.delete_button.setCursor(Qt.PointingHandCursor)
        self.delete_button.clicked.connect(self.on_remove_clicked)

        # Set size policy
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Minimum)

    def on_remove_clicked(self):
        """Handle remove button click."""
        self.remove_clicked.emit(self.name)

    def cv_to_pixmap(self, cv_img, target_width, target_height):
        """Convert OpenCV image to QPixmap with automatic format detection."""
        if cv_img is None or cv_img.size == 0:
            pixmap = QPixmap(target_width, target_height)
            pixmap.fill(QColor(30, 30, 30))
            return pixmap

        try:
            if len(cv_img.shape) == 3 and cv_img.shape[2] == 3:
                # Auto-detect format by channel ratios
                _ = np.mean(cv_img[:, :, 0])
                _ = np.mean(cv_img[:, :, 1])
                _ = np.mean(cv_img[:, :, 2])

                h, w, ch = cv_img.shape
                qt_img = QImage(cv_img.data, w, h, ch * w, QImage.Format_RGB888)
                qt_img = qt_img.copy()

                pixmap = QPixmap.fromImage(qt_img)
                return pixmap.scaled(target_width, target_height, Qt.KeepAspectRatio, Qt.SmoothTransformation)

            # Grayscale image
            h, w = cv_img.shape
            qt_img = QImage(cv_img.data, w, h, w, QImage.Format_Grayscale8)
            qt_img = qt_img.copy()
            pixmap = QPixmap.fromImage(qt_img)
            return pixmap.scaled(target_width, target_height, Qt.KeepAspectRatio, Qt.SmoothTransformation)

        except Exception as e:
            print(f"Error in cv_to_pixmap: {e}")
            # Fallback to empty pixmap
            pixmap = QPixmap(target_width, target_height)
            pixmap.fill(QColor(255, 0, 0))  # Red indicates error
            return pixmap

    def update_duration(self, seconds):
        """No-op: duration no longer displayed."""
        pass

    def resizeEvent(self, event):
        """Handle resize event to reposition delete button."""
        size = getattr(self, "button_size", 20)
        self.delete_button.setGeometry(self.width() - (size + 5), 5, size, size)
        super().resizeEvent(event)



