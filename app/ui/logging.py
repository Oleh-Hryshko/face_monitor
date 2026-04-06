from PySide6.QtCore import QObject, Signal


class LogStream(QObject):
    """Redirect stdout to a Qt signal so logs can be shown in GUI."""

    new_text = Signal(str)

    def __init__(self, original_stream=None):
        super().__init__()
        self._original = original_stream

    def write(self, text):
        if text and text.strip():
            self.new_text.emit(text.rstrip())
        if self._original:
            try:
                self._original.write(text)
            except Exception:
                pass

    def flush(self):
        if self._original:
            try:
                self._original.flush()
            except Exception:
                pass

