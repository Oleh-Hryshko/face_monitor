import os

import cv2
import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QFont, QImage, QPixmap
from PySide6.QtWidgets import (
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QScrollArea,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from app.config import config
from app.storage.files import imread_utf8


class StopListWidget(QFrame):
    """Widget for displaying a single watchlist person with their photos."""

    def __init__(self, person_name, photo_paths, info_text="", thumb_size=80, parent=None):
        super().__init__(parent)

        self.person_name = person_name
        self.photo_paths = photo_paths
        self.info_text = info_text
        self.thumb_size = thumb_size

        self.setFrameStyle(QFrame.StyledPanel | QFrame.Raised)
        self.setLineWidth(1)

        self.setStyleSheet(
            """
            QFrame {
                background-color: #2d2d2d;
                border: 1px solid #3d3d3d;
                border-radius: 8px;
                margin: 5px;
            }
            QFrame:hover {
                background-color: #3d3d3d;
                border: 1px solid #ffaa00;
            }
            """
        )

        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(8)

        # Person name header
        name_label = QLabel(person_name)
        name_font = QFont()
        name_font.setBold(True)
        name_font.setPointSize(12)
        name_label.setFont(name_font)
        name_label.setStyleSheet("color: #ffaa00;")
        name_label.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(name_label)

        # Photos grid
        photos_widget = QWidget()
        photos_layout = QGridLayout(photos_widget)
        photos_layout.setSpacing(8)
        photos_layout.setContentsMargins(5, 5, 5, 5)

        # Load and display each photo
        for idx, photo_path in enumerate(photo_paths[:12]):  # Limit to 12 photos per person
            photo_label = QLabel()
            photo_label.setFixedSize(thumb_size, thumb_size)
            photo_label.setAlignment(Qt.AlignCenter)
            photo_label.setStyleSheet(
                """
                QLabel {
                    border: 2px solid #444;
                    border-radius: 5px;
                    background-color: #1e1e1e;
                }
                QLabel:hover {
                    border: 2px solid #ffaa00;
                }
                """
            )

            # Load and display image
            pixmap = self.load_photo_pixmap(photo_path, thumb_size, thumb_size)
            photo_label.setPixmap(pixmap)
            photo_label.setToolTip(f"{person_name}\n{os.path.basename(photo_path)}")

            # Calculate grid position (4 columns max)
            row = idx // 4
            col = idx % 4
            photos_layout.addWidget(photo_label, row, col)

        # Add stretch to fill empty cells
        photos_layout.setColumnStretch(4, 1)

        main_layout.addWidget(photos_widget)

        # Add info text if exists
        if info_text and info_text.strip():
            info_label = QLabel(info_text)
            info_label.setWordWrap(True)
            info_label.setAlignment(Qt.AlignCenter)
            info_label.setStyleSheet(
                """
                QLabel {
                    color: #ccc;
                    font-size: 10px;
                    background-color: #222;
                    padding: 6px;
                    border-radius: 4px;
                    margin-top: 5px;
                }
                """
            )
            main_layout.addWidget(info_label)

        # Photo count label
        count_label = QLabel(f"{len(photo_paths)} photo(s)")
        count_label.setStyleSheet("color: #888; font-size: 9px;")
        count_label.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(count_label)

        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Minimum)

    def load_photo_pixmap(self, image_path, target_width, target_height):
        """Load image and convert to QPixmap with proper sizing."""
        try:
            img = imread_utf8(image_path)
            if img is None:
                pixmap = QPixmap(target_width, target_height)
                pixmap.fill(QColor(50, 50, 50))
                return pixmap

            # Convert BGR to RGB
            rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            h, w = rgb_img.shape[:2]

            # Calculate scaling to fit in target size while preserving aspect ratio
            scale = min(target_width / w, target_height / h)
            new_w = int(w * scale)
            new_h = int(h * scale)

            # Resize image
            resized = cv2.resize(rgb_img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

            # Create canvas and center the image
            canvas = np.zeros((target_height, target_width, 3), dtype=np.uint8)
            x_offset = (target_width - new_w) // 2
            y_offset = (target_height - new_h) // 2
            canvas[y_offset : y_offset + new_h, x_offset : x_offset + new_w] = resized

            # Convert to QPixmap
            h, w, ch = canvas.shape
            qt_img = QImage(canvas.data, w, h, ch * w, QImage.Format_RGB888)
            qt_img = qt_img.copy()

            return QPixmap.fromImage(qt_img)

        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            pixmap = QPixmap(target_width, target_height)
            pixmap.fill(QColor(255, 0, 0))
            return pixmap


class StopListPanel(QWidget):
    """Panel for displaying all watchlist photos."""

    def __init__(self, parent=None):
        super().__init__(parent)

        layout = QVBoxLayout()
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)

        # Header
        header_layout = QHBoxLayout()

        title = QLabel("📋 Watchlist Photos")
        title_font = QFont()
        title_font.setBold(True)
        title_font.setPointSize(14)
        title.setFont(title_font)
        title.setStyleSheet("color: #ffaa00;")
        header_layout.addWidget(title)

        header_layout.addStretch()

        # Info label
        self.info_label = QLabel("Loading watchlist...")
        self.info_label.setStyleSheet("color: #888; padding: 5px;")
        layout.addWidget(self.info_label)

        # Scroll area for photos
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setStyleSheet(
            """
            QScrollArea {
                border: none;
                background-color: #1e1e1e;
                border-radius: 5px;
            }
            QScrollBar:vertical {
                background-color: #2d2d2d;
                width: 10px;
                border-radius: 5px;
            }
            QScrollBar::handle:vertical {
                background-color: #555;
                border-radius: 5px;
                min-height: 20px;
            }
            QScrollBar::handle:vertical:hover {
                background-color: #666;
            }
            """
        )

        # Container for person widgets
        self.persons_container = QWidget()
        self.persons_container.setStyleSheet("background-color: #1e1e1e;")

        self.persons_layout = QVBoxLayout(self.persons_container)
        self.persons_layout.setContentsMargins(5, 5, 5, 5)
        self.persons_layout.setSpacing(10)
        self.persons_layout.setAlignment(Qt.AlignTop)

        self.scroll_area.setWidget(self.persons_container)
        layout.addWidget(self.scroll_area)

        self.setLayout(layout)

    def refresh_watchlist(self):
        """Refresh the watchlist display."""
        if self.parent() and hasattr(self.parent(), "update_watchlist_display"):
            self.parent().update_watchlist_display()

    def update_display(self, watchlist_data, watchlist_images, watchlist_info=None):
        """Update the panel with current watchlist data."""
        # Clear existing widgets
        while self.persons_layout.count() > 0:
            item = self.persons_layout.takeAt(0)
            if item and item.widget():
                item.widget().deleteLater()

        if not watchlist_data:
            # Show empty state
            empty_label = QLabel("No photos in watchlist.\n\nAdd photos to the watchlist folder to populate this list.")
            empty_label.setAlignment(Qt.AlignCenter)
            empty_label.setStyleSheet("color: #888; padding: 40px; font-size: 14px;")
            self.persons_layout.addWidget(empty_label)
            self.info_label.setText("📁 Watchlist is empty")
            print("=" * 60)
            print()
            print("📁 WATCHLIST STRUCTURE:")
            print(f"  {config.WATCHLIST_PATH}/")
            print("    ├── person1/")
            print("    │   ├── person1.txt")
            print("    │   ├── photo1.jpg")
            print("    │   └── photo2.jpg")
            print("    ├── person2.txt/")
            print("    └── person2.jpg")
            print()
            return

        # Get thumbnail size (can be configurable)
        thumb_size = getattr(config, "WATCHLIST_THUMB_SIZE", 80)

        # Sort person names alphabetically
        person_names = sorted(watchlist_data.keys())

        # Create widget for each person
        for person_name in person_names:
            photo_paths = watchlist_images.get(person_name, [])
            if photo_paths:
                # Get info text for this person
                info_text = ""
                if watchlist_info and person_name in watchlist_info:
                    info_text = watchlist_info[person_name]

                widget = StopListWidget(person_name, photo_paths, info_text, thumb_size)
                self.persons_layout.addWidget(widget)

        self.info_label.setText(
            f"📁 {len(person_names)} person(s), {sum(len(v) for v in watchlist_images.values())} photo(s)"
        )

