import queue
import threading
from datetime import datetime

from PySide6.QtCore import QObject, Signal

from .db import save_detection
from .files import save_frame


class AsyncSaver(QObject):
    """Background saver for detected frames and DB records."""

    saved_signal = Signal(str, str, str, int, str)  # name, timestamp, path, detection_id, source

    def __init__(self, maxsize=100):
        super().__init__()
        self.queue = queue.Queue(maxsize=maxsize)
        self.stop_flag = False
        self.thread = threading.Thread(target=self.worker, daemon=True)
        self.thread.start()

    def worker(self):
        while not self.stop_flag:
            try:
                frame, name, source = self.queue.get(timeout=0.1)

                path = save_frame(frame, name)
                timestamp = datetime.now().isoformat()
                detection_id = save_detection(name, source, path)

                self.saved_signal.emit(name, timestamp, path, detection_id or -1, source)

            except queue.Empty:
                continue
            except Exception as e:
                print(f"[AsyncSaver] Error: {e}")

    def save(self, frame, name, source):
        try:
            self.queue.put_nowait((frame.copy(), name, source))
        except queue.Full:
            pass

    def stop(self):
        self.stop_flag = True

