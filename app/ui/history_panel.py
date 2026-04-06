import os
from datetime import datetime

import cv2
from PySide6.QtCore import Qt
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QDialog,
    QDialogButtonBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from app.storage.db import fetch_detections, delete_detection
from app.storage.files import imread_utf8


class HistoryPanel(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.parent_window = parent  # Store reference to main window

        layout = QVBoxLayout(self)

        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText("Search by name...")
        self.search_input.textChanged.connect(self.load_data)
        layout.addWidget(self.search_input)

        # Table
        self.table = QTableWidget()
        self.table.setColumnCount(4)
        self.table.setHorizontalHeaderLabels(["Name", "Date", "Source", ""])
        self.table.cellClicked.connect(self.on_row_clicked)
        self.table.setColumnWidth(3, 36)
        self.table.horizontalHeader().setStretchLastSection(False)
        layout.addWidget(self.table)

        # Photo label with click handler
        self.image_label = QLabel("Select row to view image")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setMinimumHeight(200)
        self.image_label.setStyleSheet("border: 1px solid #444;")
        self.image_label.mousePressEvent = self.on_image_click  # Add click handler
        self.image_label.setCursor(Qt.PointingHandCursor)  # Show hand cursor
        layout.addWidget(self.image_label)

        self.paths = []  # store image paths
        self.row_ids = []  # store db ids for each row
        self.current_image_path = None  # Store current image path

        self.load_data()

    def on_image_click(self, event):
        """Open full-size image in a separate window when clicking on the photo."""
        if self.current_image_path and os.path.exists(self.current_image_path):
            # Create dialog for full-size image
            dialog = QDialog(self)
            dialog.setWindowTitle(f"Full Size Image - {os.path.basename(self.current_image_path)}")
            dialog.setMinimumSize(400, 300)

            # Create main layout
            main_layout = QVBoxLayout(dialog)
            main_layout.setContentsMargins(5, 5, 5, 5)
            main_layout.setSpacing(5)

            # Create scroll area
            scroll_area = QScrollArea()
            scroll_area.setWidgetResizable(True)  # Important: allows scrollbars to appear
            scroll_area.setAlignment(Qt.AlignCenter)
            scroll_area.setStyleSheet(
                """
                QScrollArea {
                    border: 1px solid #3d3d3d;
                    background-color: #1e1e1e;
                    border-radius: 3px;
                }
                QScrollBar:vertical {
                    background-color: #2d2d2d;
                    width: 12px;
                    border-radius: 6px;
                }
                QScrollBar::handle:vertical {
                    background-color: #555;
                    border-radius: 6px;
                    min-height: 20px;
                }
                QScrollBar::handle:vertical:hover {
                    background-color: #666;
                }
                QScrollBar:horizontal {
                    background-color: #2d2d2d;
                    height: 12px;
                    border-radius: 6px;
                }
                QScrollBar::handle:horizontal {
                    background-color: #555;
                    border-radius: 6px;
                    min-width: 20px;
                }
                QScrollBar::handle:horizontal:hover {
                    background-color: #666;
                }
                """
            )

            # Create image label
            image_label = QLabel()
            image_label.setAlignment(Qt.AlignCenter)
            image_label.setStyleSheet("background-color: #1e1e1e;")

            # Load and display image
            img = imread_utf8(self.current_image_path)
            if img is not None:
                # Convert BGR to RGB
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                h, w, ch = img_rgb.shape

                # Create QImage
                qt_img = QImage(img_rgb.data, w, h, ch * w, QImage.Format_RGB888)
                qt_img = qt_img.copy()

                # Create pixmap
                pixmap = QPixmap.fromImage(qt_img)

                # Set pixmap to label (full size)
                image_label.setPixmap(pixmap)

                # Set minimum size to image size to allow scrolling
                image_label.setMinimumSize(pixmap.size())

                # Set scroll area widget
                scroll_area.setWidget(image_label)

                # Add to layout
                main_layout.addWidget(scroll_area, 1)  # Take available space

                # Add info frame with path and image dimensions
                info_frame = QFrame()
                info_frame.setStyleSheet(
                    """
                    QFrame {
                        background-color: #2d2d2d;
                        border-radius: 3px;
                        padding: 5px;
                    }
                    QLabel {
                        color: #888;
                        font-size: 10px;
                    }
                    """
                )

                info_layout = QHBoxLayout(info_frame)
                info_layout.setContentsMargins(10, 5, 10, 5)

                # File info
                file_info = QLabel(f"📁 {os.path.basename(self.current_image_path)}")
                info_layout.addWidget(file_info)

                info_layout.addStretch()

                # Image dimensions
                dim_info = QLabel(f"📐 {w} x {h} pixels")
                info_layout.addWidget(dim_info)

                main_layout.addWidget(info_frame)

                # Add button box with Close and Full Screen buttons
                button_box = QDialogButtonBox()
                button_box.setStyleSheet(
                    """
                    QPushButton {
                        background-color: #3d3d3d;
                        color: #ccc;
                        border: 1px solid #555;
                        border-radius: 3px;
                        padding: 5px 15px;
                        min-width: 80px;
                    }
                    QPushButton:hover {
                        background-color: #4d4d4d;
                        border: 1px solid #ffaa00;
                    }
                    QPushButton:pressed {
                        background-color: #2d2d2d;
                    }
                    """
                )

                # Add Full Screen button
                fullscreen_btn = QPushButton("⛶ Full Screen")
                fullscreen_btn.clicked.connect(lambda: self.toggle_fullscreen(dialog))
                button_box.addButton(fullscreen_btn, QDialogButtonBox.ActionRole)

                # Add Close button
                close_btn = QPushButton("Close")
                close_btn.clicked.connect(dialog.accept)
                button_box.addButton(close_btn, QDialogButtonBox.RejectRole)

                main_layout.addWidget(button_box)

                # Set dialog size to 80% of screen if image is large
                screen = QApplication.primaryScreen().availableGeometry()
                dialog_width = min(w + 50, int(screen.width() * 0.8))
                dialog_height = min(h + 150, int(screen.height() * 0.8))
                dialog.resize(dialog_width, dialog_height)

                # Show dialog
                dialog.exec()
            else:
                QMessageBox.warning(self, "Error", "Could not load image")

    def toggle_fullscreen(self, dialog):
        """Toggle fullscreen mode for the dialog."""
        if dialog.isFullScreen():
            dialog.showNormal()
        else:
            dialog.showFullScreen()

    def add_row(self, name, timestamp, path, source="", detection_id=None):
        row_idx = 0
        self.table.insertRow(row_idx)

        self.table.setItem(row_idx, 0, QTableWidgetItem(name))
        self.table.setItem(row_idx, 1, QTableWidgetItem(self.format_date(timestamp)))
        self.table.setItem(row_idx, 2, QTableWidgetItem(source))
        
        # Вставляем в начало списков (поскольку вставляем строку в начало таблицы)
        self.paths.insert(0, path)
        self.row_ids.insert(0, detection_id)
        
        # Теперь устанавливаем кнопку с синхронизированным detection_id
        self._set_delete_button(row_idx)

    def format_date(self, iso_str):
        dt = datetime.fromisoformat(iso_str)
        return dt.strftime("%Y.%m.%d %H:%M:%S")

    def load_data(self):
        self.table.setRowCount(0)
        self.paths.clear()
        self.row_ids.clear()

        rows = fetch_detections(self.search_input.text())

        for row_idx, (detection_id, name, ts, path, source) in enumerate(rows):
            self.table.insertRow(row_idx)

            self.table.setItem(row_idx, 0, QTableWidgetItem(name))
            self.table.setItem(row_idx, 1, QTableWidgetItem(self.format_date(ts)))
            self.table.setItem(row_idx, 2, QTableWidgetItem(source if source else ""))

            self.paths.append(path)
            self.row_ids.append(detection_id)

        # После добавления всех строк, установить кнопки удаления для всех
        for row_idx in range(self.table.rowCount()):
            self._set_delete_button(row_idx)

        self.table.resizeColumnsToContents()
        self.table.setColumnWidth(3, 36)

    def _set_delete_button(self, row_idx):
        """Place a delete button in column 3 of the given row."""
        # row_idx должен быть синхронизирован с row_ids
        if row_idx >= len(self.row_ids):
            print(f"ERROR: row_idx {row_idx} >= len(row_ids) {len(self.row_ids)}")
            return
            
        detection_id = self.row_ids[row_idx]
        
        btn = QPushButton("🗑️")
        btn.setFixedSize(28, 24)
        btn.setToolTip("Delete this record")
        btn.setProperty("detection_id", detection_id)
        btn.setStyleSheet("""
            QPushButton {
                background-color: transparent;
                border: none;
                font-size: 14px;
            }
            QPushButton:hover {
                background-color: #5a1a1a;
                border-radius: 3px;
            }
        """)
        btn.clicked.connect(self.on_delete_button_clicked)

        cell_widget = QWidget()
        cell_layout = QHBoxLayout(cell_widget)
        cell_layout.addWidget(btn)
        cell_layout.setContentsMargins(2, 0, 2, 0)
        cell_layout.setAlignment(Qt.AlignCenter)
        self.table.setCellWidget(row_idx, 3, cell_widget)

    def on_delete_button_clicked(self):
        """Handle delete button click by detection_id."""
        sender = self.sender()
        if not sender:
            return
        detection_id = sender.property("detection_id")
        self.delete_row_by_id(detection_id)

    def delete_row(self, row_idx):
        """Legacy method for compatibility."""
        if row_idx >= len(self.row_ids):
            return
        detection_id = self.row_ids[row_idx]
        self.delete_row_by_id(detection_id)

    def delete_row_by_id(self, detection_id):
        """Delete a single record from DB, file system and table by detection_id."""
        # Find row index by detection_id
        try:
            row_idx = self.row_ids.index(detection_id)
        except ValueError:
            print(f"Detection ID {detection_id} not found in row_ids")
            return

        image_path = self.paths[row_idx] if row_idx < len(self.paths) else None
        
        # Get record info for dialog
        name = self.table.item(row_idx, 0).text() if self.table.item(row_idx, 0) else "Unknown"
        date_str = self.table.item(row_idx, 1).text() if self.table.item(row_idx, 1) else "Unknown"
        source = self.table.item(row_idx, 2).text() if self.table.item(row_idx, 2) else "Unknown"

        # Build detailed confirmation message
        msg = "Delete this detection record?\n\n"
        msg += f"👤 Name: {name}\n"
        msg += f"📅 Date: {date_str}\n"
        msg += f"📡 Source: {source}"
        if image_path:
            msg += f"\n📁 File: {os.path.basename(image_path)}"

        reply = QMessageBox.question(
            self, "Delete Record",
            msg,
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if reply != QMessageBox.Yes:
            return

        # Delete from DB
        if detection_id is not None:
            try:
                delete_detection(detection_id)
            except Exception as e:
                print(f"Failed to delete DB record {detection_id}: {e}")

        # Delete image file
        if image_path and os.path.isfile(image_path):
            try:
                os.remove(image_path)
            except Exception as e:
                print(f"Failed to delete file {image_path}: {e}")

        # Remove from table + internal lists
        self.table.removeRow(row_idx)
        self.paths.pop(row_idx)
        self.row_ids.pop(row_idx)

        # Re-bind remaining delete buttons (row indices shifted after removal)
        for r in range(row_idx, self.table.rowCount()):
            self._set_delete_button(r)

        # Reset image preview if it was the deleted row
        if image_path and image_path == self.current_image_path:
            self.current_image_path = None
            self.image_label.setPixmap(QPixmap())
            self.image_label.setText("Select row to view image")

    def on_row_clicked(self, row, col):
        """Show thumbnail in the panel when clicking on a row."""
        if row < len(self.paths):
            self.current_image_path = self.paths[row]
            self.show_thumbnail(self.current_image_path)

    def show_thumbnail(self, path):
        """Show thumbnail in the history panel."""
        img = imread_utf8(path)
        if img is None:
            self.image_label.setText("Image not found")
            self.current_image_path = None
            return

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, ch = img.shape
        qt_img = QImage(img.data, w, h, ch * w, QImage.Format_RGB888)
        qt_img = qt_img.copy()

        pixmap = QPixmap.fromImage(qt_img)
        pixmap = pixmap.scaled(300, 300, Qt.KeepAspectRatio, Qt.SmoothTransformation)

        self.image_label.setPixmap(pixmap)

        # Add tooltip to show click instruction
        self.image_label.setToolTip("Click to open full size image")

