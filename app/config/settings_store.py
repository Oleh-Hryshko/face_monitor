import json
from pathlib import Path

from app.config import config


SETTINGS_FILE = Path(__file__).resolve().parents[2] / "settings.json"


def load_saved_settings():
    """Load settings from JSON file into config. Call before creating UI."""
    if not SETTINGS_FILE.exists():
        return

    try:
        with open(SETTINGS_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        for key, value in data.items():
            if hasattr(config, key):
                setattr(config, key, value)
    except Exception as e:
        print(f"Could not load settings: {e}")


def save_settings_to_file():
    """Save current config settings to JSON file."""
    keys = [
        "MODEL_NAME",
        "DETECTOR",
        "SIMILARITY_THRESHOLD",
        "DISTANCE_METRIC",
        "PROCESS_INTERVAL",
        "DETECTION_SCALE",
        "MAX_FACES_TO_CHECK",
        "ASYNC_PROCESSING",
        "EXPAND_FACE_BOX",
        "FACE_BOX_EXPAND_FACTOR",
        "FACE_BOX_HEADROOM",
        "DETECTED_FACE_THUMB_SIZE",
    ]

    try:
        data = {k: getattr(config, k) for k in keys if hasattr(config, k)}
        with open(SETTINGS_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
    except Exception as e:
        print(f"Could not save settings: {e}")


