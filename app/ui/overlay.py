from datetime import datetime

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from PySide6.QtCore import Qt
from PySide6.QtGui import QFont
from PySide6.QtWidgets import QLabel, QSizePolicy, QVBoxLayout, QWidget


def draw_text_unicode(frame, text, x, y):
    """Draw unicode text on an OpenCV frame via PIL."""
    img_pil = Image.fromarray(frame)
    draw = ImageDraw.Draw(img_pil)
    font = ImageFont.truetype("arial.ttf", 16)
    draw.text((x, y), text, font=font, fill=(255, 255, 255))
    return np.array(img_pil)


def draw_label(frame, text, x, y):
    """Draw a filled label with text on an OpenCV frame via PIL."""
    img_pil = Image.fromarray(frame)
    draw = ImageDraw.Draw(img_pil)
    font = ImageFont.truetype("arial.ttf", 16)

    bbox = draw.textbbox((0, 0), text, font=font)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]
    padding = 5

    draw.rectangle(
        [x, y, x + text_w + padding * 2, y + text_h + padding * 2],
        fill=(0, 0, 255),
    )
    draw.text((x + padding, y + padding), text, font=font, fill=(255, 255, 255))

    return np.array(img_pil)


def calculate_detected_card_min_height(thumb_size, info_text):
    """Return a conservative minimum height for a detected-face card."""
    # Base height for photos + padding
    base_height = thumb_size + 16
    # Name label height
    name_height = 24
    # Info label height (if present)
    info_height = 30 if (info_text and info_text.strip()) else 0
    # Time label height
    time_height = 18
    # Spacing between elements
    spacing = 6 * 4
    # Total extra space for margins and padding
    extra_space = 20
    
    total_height = base_height + name_height + info_height + time_height + spacing + extra_space
    return total_height


def build_detected_face_text_section(name, info_text, detection_time, thumb_size):
    """Build the text block that is always placed below photo thumbnails."""
    text_container = QWidget()
    text_container.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Minimum)

    text_layout = QVBoxLayout(text_container)
    text_layout.setContentsMargins(0, 6, 0, 0)
    text_layout.setSpacing(6)

    # Name label
    name_label = QLabel(name)
    name_font = QFont()
    name_font.setBold(True)
    name_font.setPointSize(max(9, min(12, int(thumb_size / 10))))
    name_label.setFont(name_font)
    name_label.setStyleSheet("color: #ffaa00; padding: 2px;")
    name_label.setAlignment(Qt.AlignCenter)
    name_label.setWordWrap(True)
    name_label.setMinimumHeight(24)
    text_layout.addWidget(name_label)

    # Info label (if present)
    if info_text and info_text.strip():
        info_label = QLabel(info_text)
        info_label.setWordWrap(True)
        info_label.setAlignment(Qt.AlignCenter)
        info_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Minimum)
        info_label.setMinimumHeight(30)
        info_label.setStyleSheet(
            """
            color: #ccc;
            font-size: 10px;
            background-color: #222;
            padding: 4px;
            border-radius: 4px;
            """
        )
        text_layout.addWidget(info_label)

    # Time label
    if isinstance(detection_time, datetime):
        time_text = detection_time.strftime("%Y-%m-%d %H:%M:%S")
    else:
        time_text = str(detection_time)

    time_label = QLabel(time_text)
    time_font_size = max(8, min(10, int(thumb_size / 12)))
    time_label.setStyleSheet(f"color: #888; font-size: {time_font_size}px; padding: 2px;")
    time_label.setAlignment(Qt.AlignCenter)
    time_label.setMinimumHeight(18)
    text_layout.addWidget(time_label)

    return text_container


def get_delete_button_style(button_size, font_size):
    """Return style for the card delete button with size-dependent geometry."""
    return f"""
        QPushButton {{
            background-color: #c44;
            color: white;
            border: 1px solid #a33;
            border-radius: {button_size // 2}px;
            font-size: {font_size}px;
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
    """




