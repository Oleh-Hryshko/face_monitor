"""
Configuration file for Store Face Monitor GUI version
All settings are centralized here for easy adjustment
"""

import os
from pathlib import Path

# ============================================
# PATHS
# ============================================

# Path to watchlist folder (absolute or relative)
WATCHLIST_PATH = 'watchlist'

# Path to database
DB_PATH = "detections.db"

# ============================================
# RECOGNITION SETTINGS
# ============================================

# Model name (same as original shoplifter config.py)
# Options: "VGG-Face", "Facenet", "Facenet512", "OpenFace", "DeepFace", "DeepID", "ArcFace", "Dlib"
MODEL_NAME = "Facenet512"  # Most accurate but slower

# Face detector:
# Options: "opencv", "ssd", "dlib", "mtcnn", "fastmtcnn", "retinaface", "mediapipe", "yolov8", "yunet"
DETECTOR = "mtcnn"  # Good balance of accuracy and speed

# Similarity threshold (same as original)
# cosine: 0-2 range, 0 = identical. Typical: 0.4 (strict), 0.5 (balanced), 0.6 (tolerant)
SIMILARITY_THRESHOLD = 0.4

# Distance metric (same as original)
# cosine: 0-2 range, recommended for Facenet/Facenet512
DISTANCE_METRIC = "cosine"

# ============================================
# PERFORMANCE SETTINGS
# ============================================

# Process every Nth frame (higher = faster but may miss faces)
# 1 = process every frame (slowest), 3 = balanced, 5–7 = faster (recommended for low FPS)
PROCESS_INTERVAL = 3

# Resize frame for face detection to speed up (1.0 = no resize, 0.5 = half resolution)
# 0.75 or 0.5 = faster detection, may miss small/distant faces
DETECTION_SCALE = 0.75

# Use "skip" detector when getting embedding for already cropped face (faster, recommended True)
SKIP_DETECTOR_FOR_CROPPED_FACE = True

# Check only N largest faces per frame against stop list (rest are drawn as "normal")
# Lower = higher FPS. 1–2 recommended for weak hardware.
MAX_FACES_TO_CHECK = 5

# Run recognition in background thread: display stays smooth, result lags by ~1 interval
# False = process in same thread as video (more reliable for DeepFace/TensorFlow)
ASYNC_PROCESSING = False

# ============================================
# FACE BOX EXPANSION SETTINGS
# ============================================

# Expand face box to capture full head (like passport photo)
EXPAND_FACE_BOX = True  # Enable/disable expansion
FACE_BOX_EXPAND_FACTOR = 1.8  # 1.5-2.0 recommended (1.8 = 80% larger)
FACE_BOX_HEADROOM = 0.1  # Extra headroom (10% of box height)

# Size of face thumbnails in Detected panel (pixels)
DETECTED_FACE_THUMB_SIZE = 100

# ============================================
# DEBUG SETTINGS
# ============================================

# Enable debug output (True/False)
DEBUG_MODE = True

# ============================================
# COLORS (for OpenCV drawing)
# ============================================

# Colors in BGR format
COLOR_VIOLATOR = (0, 0, 255)      # Red
COLOR_NORMAL = (0, 255, 0)        # Green
COLOR_TEXT = (255, 255, 255)      # White
COLOR_ALERT = (0, 0, 255)         # Red
COLOR_PANEL_BG = (50, 50, 50)     # Dark gray for panel background
COLOR_PANEL_TEXT = (255, 255, 255) # White for panel text
COLOR_PANEL_SECONDARY = (200, 200, 200) # Light gray for secondary text
COLOR_PANEL_BORDER = (100, 100, 100) # Gray for borders
COLOR_PANEL_SEPARATOR = (80, 80, 80) # Darker gray for separators

# ============================================
# SOUND SETTINGS (Windows only)
# ============================================

# Alert sound frequencies (Hz) and durations (ms)
ALERT_SOUNDS = [
    (600, 800),   
]
ALERT_SOUND_DELAY = 0.05

# ============================================
# ADVANCED SETTINGS
# ============================================

# Enforce face detection when loading stop list photos
# If True, will raise error if no face found in photo
ENFORCE_DETECTION = False

# Align faces after detection (usually improves accuracy)
ALIGN_FACES = True

# ============================================
# CONFIGURATION PRESETS
# ============================================

def get_preset(preset_name="balanced"):
    """
    Get predefined configuration presets
    """
    presets = {
        "fast": {
            "MODEL_NAME": "Facenet",
            "DETECTOR": "opencv",
            "SIMILARITY_THRESHOLD": 0.5,
            "DISTANCE_METRIC": "cosine",
            "PROCESS_INTERVAL": 5,
            "description": "Fastest processing, lower accuracy"
        },
        "streaming": {
            "MODEL_NAME": "Facenet512",
            "DETECTOR": "ssd",
            "SIMILARITY_THRESHOLD": 0.35,
            "DISTANCE_METRIC": "cosine",
            "PROCESS_INTERVAL": 2,
            "ASYNC_PROCESSING": True,
            "description": "Optimized for live camera feeds and smooth tracking"
        },
        "balanced": {
            "MODEL_NAME": "Facenet512",
            "DETECTOR": "mtcnn",
            "SIMILARITY_THRESHOLD": 0.35,
            "DISTANCE_METRIC": "cosine",
            "PROCESS_INTERVAL": 3,
            "ASYNC_PROCESSING": True,
            "description": "Good balance of accuracy and speed"
        }
    }
    return presets.get(preset_name, presets["balanced"])

def apply_preset(preset_name="balanced"):
    """Apply a preset configuration"""
    preset = get_preset(preset_name)
    print(f"\n📋 Applying preset: {preset_name.upper()}")
    print(f"   {preset['description']}")
    print()
    
    # Update global variables
    for key, value in preset.items():
        if key != "description":
            globals()[key] = value
    return preset

# ============================================
# VALIDATION
# ============================================

def validate_config():
    """Validate configuration settings"""
    warnings = []
    
    # Check threshold range
    if not 0 <= SIMILARITY_THRESHOLD <= 1:
        warnings.append(f"SIMILARITY_THRESHOLD should be between 0 and 1, got {SIMILARITY_THRESHOLD}")
    
    # Check model name
    valid_models = ["VGG-Face", "Facenet", "Facenet512", "OpenFace", "DeepFace", "DeepID", "ArcFace", "Dlib"]
    if MODEL_NAME not in valid_models:
        warnings.append(f"MODEL_NAME '{MODEL_NAME}' may not be valid. Valid options: {valid_models}")
    
    # Check detector
    valid_detectors = ["opencv", "ssd", "dlib", "mtcnn", "fastmtcnn", "retinaface", "mediapipe", "yolov8", "yunet"]
    if DETECTOR not in valid_detectors:
        warnings.append(f"DETECTOR '{DETECTOR}' may not be valid. Valid options: {valid_detectors}")
    
    # Check metric
    valid_metrics = ["cosine", "euclidean", "euclidean_l2"]
    if DISTANCE_METRIC not in valid_metrics:
        warnings.append(f"DISTANCE_METRIC '{DISTANCE_METRIC}' not valid. Valid options: {valid_metrics}")
    
    # Check process interval
    if PROCESS_INTERVAL < 1:
        warnings.append(f"PROCESS_INTERVAL should be >= 1, got {PROCESS_INTERVAL}")

    if not 0.25 <= DETECTION_SCALE <= 1.0:
        warnings.append(f"DETECTION_SCALE should be between 0.25 and 1.0, got {DETECTION_SCALE}")

    if MAX_FACES_TO_CHECK is not None and MAX_FACES_TO_CHECK < 1:
        warnings.append(f"MAX_FACES_TO_CHECK should be >= 1 or None, got {MAX_FACES_TO_CHECK}")
    
    # Check face box expansion settings
    if FACE_BOX_EXPAND_FACTOR < 1.0:
        warnings.append(f"FACE_BOX_EXPAND_FACTOR should be >= 1.0, got {FACE_BOX_EXPAND_FACTOR}")
    
    if not 0 <= FACE_BOX_HEADROOM <= 0.3:
        warnings.append(f"FACE_BOX_HEADROOM should be between 0 and 0.3, got {FACE_BOX_HEADROOM}")
    
    return warnings

# If running this file directly, show configuration
if __name__ == "__main__":
    print("="*60)
    print("STORE FACE MONITOR - CONFIGURATION")
    print("="*60)
    print(f"\nCurrent settings:")
    print(f"  Model: {MODEL_NAME}")
    print(f"  Detector: {DETECTOR}")
    print(f"  Threshold: {SIMILARITY_THRESHOLD}")
    print(f"  Metric: {DISTANCE_METRIC}")
    print(f"  Process interval: {PROCESS_INTERVAL}")
    print(f"  Detection scale: {DETECTION_SCALE}")
    print(f"  Skip detector for cropped: {SKIP_DETECTOR_FOR_CROPPED_FACE}")
    print(f"  Max faces to check: {MAX_FACES_TO_CHECK}")
    print(f"  Async processing: {ASYNC_PROCESSING}")
    print(f"  Face box expansion: {EXPAND_FACE_BOX}")
    print(f"  Expand factor: {FACE_BOX_EXPAND_FACTOR}")
    print(f"  Headroom: {FACE_BOX_HEADROOM}")
    print(f"  Debug mode: {DEBUG_MODE}")
    print()
    
    # Validate
    warnings = validate_config()
    if warnings:
        print("⚠️  Warnings:")
        for w in warnings:
            print(f"  - {w}")
    else:
        print("✅ Configuration valid")
    
    print("\nAvailable presets:")
    for preset_name in ["fast", "streaming", "balanced"]:
        preset = get_preset(preset_name)
        print(f"  - {preset_name}: {preset['description']}")
    
    print("="*60)

