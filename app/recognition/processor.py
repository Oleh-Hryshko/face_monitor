import cv2
import numpy as np
from deepface import DeepFace

from app.config import config
from app.storage.files import imread_utf8
from app.ui.overlay import draw_label


class FaceProcessor:
    """Handles face detection and recognition."""

    def __init__(self, watchlist_data, watchlist_images, watchlist_info):
        self.watchlist_data = watchlist_data
        self.watchlist_images = watchlist_images
        self.watchlist_info = watchlist_info
        self.debug_mode = config.DEBUG_MODE

        # Recognition settings
        self.model_name = config.MODEL_NAME
        self.detector = config.DETECTOR
        self.threshold = config.SIMILARITY_THRESHOLD
        self.metric = config.DISTANCE_METRIC
        self.skip_detector = config.SKIP_DETECTOR_FOR_CROPPED_FACE

        # Face box expansion
        self.expand_enabled = config.EXPAND_FACE_BOX
        self.expand_factor = config.FACE_BOX_EXPAND_FACTOR
        self.headroom = config.FACE_BOX_HEADROOM

        # Cache for thumbnails
        self.thumb_cache = {}

        self.detection_paused = False

        # Vectorized watchlist matching (rebuilt lazily; invalidate on reload / metric change)
        self._wl_matrix = None
        self._wl_names = None
        self._wl_paths = None
        self._wl_index_metric = None

    def invalidate_watchlist_index(self):
        """Call after watchlist reload or when distance metric changes."""
        self._wl_matrix = None
        self._wl_names = None
        self._wl_paths = None
        self._wl_index_metric = None

    def _ensure_watchlist_index(self):
        if self._wl_matrix is not None and self._wl_index_metric == self.metric:
            return

        rows = []
        names = []
        paths = []
        for person_name, person_photos in self.watchlist_data.items():
            for photo_data in person_photos:
                rows.append(np.asarray(photo_data["embedding"], dtype=np.float64))
                names.append(person_name)
                paths.append(photo_data["path"])

        if not rows:
            self._wl_matrix = np.empty((0, 0), dtype=np.float64)
            self._wl_names = []
            self._wl_paths = []
            self._wl_index_metric = self.metric
            return

        X = np.vstack(rows)
        if self.metric in ("cosine", "euclidean_l2"):
            norms = np.linalg.norm(X, axis=1, keepdims=True)
            norms = np.where(norms < 1e-10, 1e-10, norms)
            self._wl_matrix = X / norms
        else:
            self._wl_matrix = X
        self._wl_names = names
        self._wl_paths = paths
        self._wl_index_metric = self.metric

    def expand_face_box(self, x, y, w, h, frame_shape):
        """Expand face box to include head/upper body."""
        if not self.expand_enabled:
            return x, y, w, h

        frame_h, frame_w = frame_shape[:2]

        # Calculate new dimensions
        new_w = int(w * self.expand_factor)
        new_h = int(h * self.expand_factor)

        # Center on face
        center_x = x + w // 2
        center_y = y + h // 2

        new_x = center_x - new_w // 2
        new_y = center_y - new_h // 2

        # Add headroom
        new_y -= int(new_h * self.headroom)

        # Ensure within frame
        new_x = max(0, min(new_x, frame_w - new_w))
        new_y = max(0, min(new_y, frame_h - new_h))

        if new_x + new_w > frame_w:
            new_x = frame_w - new_w
        if new_y + new_h > frame_h:
            new_y = frame_h - new_h

        return new_x, new_y, new_w, new_h

    def check_face_against_watchlist(self, face_img, frame_number=0):
        """Check face against watchlist. Returns (violator_name, best_photo_path) or (None, None)."""
        if not self.watchlist_data:
            return None, None

        # DeepFace with detector_backend="skip" does NOT normalize to [0,1], but when loading
        # from file it uses extract_faces which does img/255. So we must pass [0,1] float
        # so embeddings are in the same space as watchlist (otherwise distance is huge, >0.8).
        if isinstance(face_img, np.ndarray):
            face_img = np.ascontiguousarray(face_img)
            if face_img.dtype in (np.float32, np.float64):
                face_img = np.clip(face_img, 0, 1).astype(np.float32)
            else:
                face_img = face_img.astype(np.float32) / 255.0

        try:
            result = DeepFace.represent(
                img_path=face_img,
                model_name=self.model_name,
                detector_backend="skip" if self.skip_detector else self.detector,
                enforce_detection=False,
            )

            if not result:
                return None, None

            embedding = result[0]["embedding"]

            self._ensure_watchlist_index()
            if self._wl_matrix.shape[0] == 0:
                return None, None

            a = np.asarray(embedding, dtype=np.float64)
            if self.metric == "cosine":
                a_norm = np.linalg.norm(a)
                if a_norm < 1e-10:
                    return None, None
                a = a / a_norm
                distances = 1.0 - self._wl_matrix @ a
            elif self.metric == "euclidean_l2":
                a = a / (np.linalg.norm(a) + 1e-10)
                distances = np.linalg.norm(self._wl_matrix - a, axis=1)
            else:
                distances = np.linalg.norm(self._wl_matrix - a, axis=1)

            idx = int(np.argmin(distances))
            min_distance = float(distances[idx])
            best_match = self._wl_names[idx]
            best_photo = self._wl_paths[idx]

            # Debug: log best distance so user can adjust threshold (every 30 frames to avoid spam)
            if self.debug_mode and frame_number % 30 == 0:
                if min_distance < self.threshold:
                    print(
                        f"    [Frame {frame_number}] MATCH: {best_match} "
                        f"(distance {min_distance:.4f} < {self.threshold})"
                    )
                else:
                    print(
                        f"    [Frame {frame_number}] Best: {best_match} "
                        f"distance={min_distance:.4f} (threshold={self.threshold}) "
                        "- increase threshold if this should match)"
                    )

            if min_distance < self.threshold:
                return best_match, best_photo
            return None, None

        except Exception as e:
            if self.debug_mode:
                print(f"    Error in face check: {e}")
            return None, None

    def get_thumbnail(self, name, path=None):
        """Get thumbnail for watchlist photo."""
        cache_key = path if path else name
        if cache_key in self.thumb_cache:
            return self.thumb_cache[cache_key]

        img_path = path
        if not img_path:
            paths = self.watchlist_images.get(name)
            if not paths or len(paths) == 0:
                return None
            img_path = paths[0]

        try:
            # Use UTF-8 compatible image reading
            img = imread_utf8(img_path)
            if img is None:
                return None

            # Convert to RGB and resize
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            h, w = img.shape[:2]
            size = 70
            scale = size / max(h, w)
            new_h, new_w = int(h * scale), int(w * scale)
            img = cv2.resize(img, (new_w, new_h))

            # Create square with padding
            square = np.zeros((size, size, 3), dtype=np.uint8)
            y_offset = (size - new_h) // 2
            x_offset = (size - new_w) // 2
            square[y_offset : y_offset + new_h, x_offset : x_offset + new_w] = img

            self.thumb_cache[cache_key] = square
            return square

        except Exception as e:
            if self.debug_mode:
                print(f"Error loading thumbnail: {e}")
            return None

    def process_frame(self, frame, frame_number):
        """
        Process frame: detect faces and check against watchlist.
        Returns (frame_with_boxes, detections, original_frame)
        """
        detections = []
        scale = config.DETECTION_SCALE
        # Create a frame with bounding boxes for display
        frame_with_boxes = frame.copy()
        original_frame = frame

        # If detection is paused, return frame without recognition results.
        if getattr(self, "detection_paused", False):
            return frame_with_boxes, detections, original_frame

        try:
            if scale < 1.0:
                h, w = frame.shape[:2]
                small_w, small_h = int(w * scale), int(h * scale)
                detection_frame = cv2.resize(frame, (small_w, small_h))
                inv_scale = 1.0 / scale
            else:
                detection_frame = frame
                inv_scale = 1.0

            # Detect faces
            try:
                faces = DeepFace.extract_faces(
                    img_path=detection_frame,
                    detector_backend=self.detector,
                    enforce_detection=False,
                    align=config.ALIGN_FACES,
                )
            except Exception as ex:
                if self.debug_mode and frame_number % 60 == 0:
                    print(f"  [Frame {frame_number}] extract_faces error: {ex}")
                faces = []

            if faces and len(faces) > 0:
                if self.debug_mode and frame_number % 30 == 0:
                    print(f"  [Frame {frame_number}] Found {len(faces)} face(s)")
            elif self.debug_mode and frame_number % 60 == 0:
                print(f"  [Frame {frame_number}] No faces detected")

            # Sort faces by area (largest first)
            candidates = []
            for face_data in faces:
                facial_area = face_data.get("facial_area", {})
                if not facial_area:
                    continue

                x = int(facial_area.get("x", 0) * inv_scale)
                y = int(facial_area.get("y", 0) * inv_scale)
                w = int(facial_area.get("w", 0) * inv_scale)
                h = int(facial_area.get("h", 0) * inv_scale)

                # Save original coordinates for recognition
                orig_x, orig_y, orig_w, orig_h = x, y, w, h

                # Get face for recognition (original, unexpanded)
                face_img_aligned = face_data.get("face")
                if face_img_aligned is not None:
                    # Convert from float [0,1] to uint8 [0,255]
                    if face_img_aligned.dtype == np.float32 or face_img_aligned.dtype == np.float64:
                        face_img_aligned = (face_img_aligned * 255).astype(np.uint8)
                    face_for_rec = cv2.cvtColor(face_img_aligned, cv2.COLOR_RGB2BGR)
                else:
                    if orig_y + orig_h <= frame.shape[0] and orig_x + orig_w <= frame.shape[1]:
                        face_for_rec = frame[orig_y : orig_y + orig_h, orig_x : orig_x + orig_w].copy()
                    else:
                        continue

                # Expand box for display
                x, y, w, h = self.expand_face_box(x, y, w, h, frame.shape)

                x = max(0, min(x, frame.shape[1] - 1))
                y = max(0, min(y, frame.shape[0] - 1))
                w = min(w, frame.shape[1] - x)
                h = min(h, frame.shape[0] - y)

                if w <= 0 or h <= 0:
                    continue

                # Get expanded face for display
                face_display = frame[y : y + h, x : x + w].copy()

                area = w * h
                candidates.append((area, x, y, w, h, face_for_rec, face_display))

            # Sort and process largest faces first
            candidates.sort(key=lambda c: c[0], reverse=True)
            max_check = config.MAX_FACES_TO_CHECK if config.MAX_FACES_TO_CHECK else len(candidates)

            # Collect all detections and draw boxes DIRECTLY on frame_with_boxes
            for i, (_, x, y, w, h, face_rec, face_display) in enumerate(candidates):
                violator = None
                photo_path = None
                thumb = None

                if i < max_check:
                    violator, photo_path = self.check_face_against_watchlist(face_rec, frame_number)

                # Get thumbnail if violator
                if violator:
                    thumb = self.get_thumbnail(violator, photo_path)

                # Determine box color
                color = config.COLOR_VIOLATOR if violator else config.COLOR_NORMAL
                thickness = 2 if violator else 1

                # Draw box on frame_with_boxes (for display AND saving)
                cv2.rectangle(frame_with_boxes, (x, y), (x + w, y + h), color, thickness, cv2.LINE_AA)

                if violator:
                    # Draw name
                    label = violator
                    frame_with_boxes = draw_label(frame_with_boxes, label, x, y + h + 5)

                    # Add to detections (full_frame assigned once after all boxes drawn)
                    detections.append(
                        {
                            "bbox": (x, y, w, h),
                            "name": violator,
                            "face_img": face_display,
                            "thumb": thumb,
                            "photo_path": photo_path,
                            "info": self.watchlist_info.get(violator, ""),
                        }
                    )

        except Exception as e:
            if self.debug_mode:
                print(f"Error processing frame: {e}")
                import traceback

                traceback.print_exc()

        if detections:
            for d in detections:
                d["full_frame"] = frame_with_boxes

        return frame_with_boxes, detections, original_frame

    def pause_detection(self):
        """Pause face detection processing."""
        self.detection_paused = True


    def resume_detection(self):
        """Resume face detection processing."""
        self.detection_paused = False
