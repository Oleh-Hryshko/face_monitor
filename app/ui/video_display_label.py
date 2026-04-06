from PySide6.QtCore import QSize
from PySide6.QtWidgets import QLabel


class VideoDisplayLabel(QLabel):
    """
    Video QLabel: does not use pixmap size as minimumSizeHint,
    otherwise the layout leaves a small space on the left and doesn't stretch when maximized.
    """

    def minimumSizeHint(self):
        return QSize(0, 0)

    def sizeHint(self):
        return QSize(640, 360)

