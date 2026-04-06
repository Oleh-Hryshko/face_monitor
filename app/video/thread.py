import os
import queue
import threading
import time

import cv2
import mss
import numpy as np
from PySide6.QtCore import QThread, Signal
from PySide6.QtGui import QImage

from app.config import config
from app.video.win_capture import clip_bbox_for_mss, get_window_rect_pixels, grab_window_bgr_hwnd


class VideoThread(QThread):
    """Thread for video capture and processing."""

    change_pixmap_signal = Signal(QImage)
    detection_signal = Signal(list)  # Signal for detection results (list of dicts)
    fps_signal = Signal(float)

    def __init__(self):
        super().__init__()
        self._run_flag = True
        self.cap = None
        self.video_source = 0
        self.processor = None  # Reference to face processor
        self.fps = 0
        self.frame_count = 0
        self.last_time = time.time()
        self.async_mode = config.ASYNC_PROCESSING
        self.processing_queue = queue.Queue(maxsize=2)
        self.result_queue = queue.Queue(maxsize=2)
        self.source_type = "camera"  # camera | video | screen
        self.sct = None
        self.monitor = None
        # Screen mode: None = full primary monitor, else Win32 HWND for window region
        self.screen_capture_hwnd = None
        # If HWND matches main app window (bad QVariant / user mistake), capture full monitor instead
        self.exclude_capture_hwnd = None

        if self.async_mode:
            self.start_async_worker()

    def start_async_worker(self):
        """Start background worker for async processing."""

        def worker():
            while self._run_flag:
                try:
                    # Get frame from queue
                    frame, frame_num = self.processing_queue.get(timeout=0.1)
                    if frame is not None and self.processor:
                        processed_frame, detections, original_frame = self.processor.process_frame(frame, frame_num)
                        # Put results in result queue
                        try:
                            self.result_queue.put_nowait((processed_frame, detections, original_frame))
                        except queue.Full:
                            # If queue is full, remove oldest and add new
                            try:
                                self.result_queue.get_nowait()
                                self.result_queue.put_nowait((processed_frame, detections, original_frame))
                            except Exception:
                                pass
                except queue.Empty:
                    continue
                except Exception as e:
                    print(f"Async worker error: {e}")

        self.worker_thread = threading.Thread(target=worker, daemon=True)
        self.worker_thread.start()

    def set_video_source(self, source, source_type="camera"):
        self.video_source = source
        self.source_type = source_type

    def set_processor(self, processor):
        self.processor = processor

    def set_screen_capture_hwnd(self, hwnd):
        """None = capture entire primary monitor; int = Win32 HWND of window region."""
        self.screen_capture_hwnd = hwnd

    def set_exclude_capture_hwnd(self, hwnd):
        """Native HWND of our main window — never crop capture to this handle only."""
        self.exclude_capture_hwnd = hwnd

    def run(self):
        if self.source_type in ["camera", "video"]:
            self.cap = cv2.VideoCapture(self.video_source)
            if not self.cap.isOpened():
                print("Error: Could not open video source")
                return
        elif self.source_type == "screen":
            self.sct = mss.mss()
            self.monitor = self.sct.monitors[1]  # main monitor

        self.last_time = time.time()
        self.frame_count = 0
        # Separate counter for "every N-th frame" processing (not reset by FPS)
        self._process_counter = 0
        # Always show last frame WITH face boxes (never raw when we have one)
        last_processed_frame = None

        while self._run_flag:
            if self.source_type in ["camera", "video"]:
                ret, cv_img = self.cap.read()
                if not ret:
                    if isinstance(self.video_source, str) and os.path.exists(self.video_source):
                        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        continue
                    break

            elif self.source_type == "screen":
                hwnd = getattr(self, "screen_capture_hwnd", None)
                if hwnd is not None:
                    try:
                        hwnd = int(hwnd)
                    except (TypeError, ValueError):
                        hwnd = None
                ex = getattr(self, "exclude_capture_hwnd", None)
                if hwnd and ex is not None and int(hwnd) == int(ex):
                    hwnd = None

                cv_img = None
                if hwnd:
                    cv_img = grab_window_bgr_hwnd(hwnd)

                if cv_img is None:
                    if hwnd:
                        geom = get_window_rect_pixels(hwnd)
                        if geom is None:
                            bbox = self.monitor
                        else:
                            left, top, w, h = geom
                            clipped = clip_bbox_for_mss(self.sct, left, top, w, h)
                            bbox = clipped if clipped is not None else self.monitor
                    else:
                        bbox = self.monitor
                    try:
                        screenshot = self.sct.grab(bbox)
                    except Exception:
                        try:
                            screenshot = self.sct.grab(self.monitor)
                        except Exception:
                            time.sleep(0.02)
                            continue
                    frame = np.array(screenshot)
                    cv_img = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

            self.frame_count += 1
            self._process_counter += 1

            # FPS (reset only display counter, not _process_counter)
            current_time = time.time()
            if current_time - self.last_time >= 1.0:
                self.fps = self.frame_count / (current_time - self.last_time)
                self.fps_signal.emit(self.fps)
                self.frame_count = 0
                self.last_time = current_time

            process_this_frame = self._process_counter % config.PROCESS_INTERVAL == 0

            if self.async_mode and self.processor:
                if process_this_frame:
                    try:
                        self.processing_queue.put_nowait((cv_img.copy(), self._process_counter))
                    except queue.Full:
                        try:
                            self.processing_queue.get_nowait()
                            self.processing_queue.put_nowait((cv_img.copy(), self._process_counter))
                        except queue.Empty:
                            pass

                try:
                    processed_frame, detections, original_frame = self.result_queue.get_nowait()
                    last_processed_frame = processed_frame.copy()
                    if detections and not getattr(self.processor, "detection_paused", False):
                        for detection in detections:
                            detection["full_frame"] = processed_frame
                        self.detection_signal.emit(detections)
                except queue.Empty:
                    pass
                # Display last frame with boxes, or raw only until first result
                display_frame = last_processed_frame if last_processed_frame is not None else cv_img
            else:
                if process_this_frame and self.processor:
                    try:
                        processed_frame, detections, original_frame = self.processor.process_frame(
                            cv_img.copy(), self._process_counter
                        )
                        last_processed_frame = processed_frame.copy()
                        # Only emit detection signal if detection is not paused
                        if detections and not getattr(self.processor, "detection_paused", False):
                            self.detection_signal.emit(detections)
                    except Exception as e:
                        if config.DEBUG_MODE:
                            print(f"process_frame error: {e}")
                        last_processed_frame = cv_img.copy()
                display_frame = last_processed_frame if last_processed_frame is not None else cv_img

            rgb_image = np.ascontiguousarray(cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB))
            h, w, ch = rgb_image.shape
            bytes_per_line = ch * w
            qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888).copy()
            self.change_pixmap_signal.emit(qt_image)

        if self.source_type in ["camera", "video"] and self.cap:
            self.cap.release()

    def stop(self):
        self._run_flag = False
        self.wait()

    def clear_detection_queues(self):
        """Clear processing and result queues to prevent old detections from being processed."""
        try:
            while not self.processing_queue.empty():
                self.processing_queue.get_nowait()
        except queue.Empty:
            pass
        
        try:
            while not self.result_queue.empty():
                self.result_queue.get_nowait()
        except queue.Empty:
            pass



