import os
import sys
import time

import cv2
import numpy as np

DETECTIONS_DIR = "detections"


_CYRILLIC_TO_LATIN = {
    "\u0430": "a", "\u0431": "b", "\u0432": "v", "\u0433": "g", "\u0434": "d", "\u0435": "e", "\u0451": "e",
    "\u0436": "zh", "\u0437": "z", "\u0438": "i", "\u0439": "y", "\u043a": "k", "\u043b": "l", "\u043c": "m",
    "\u043d": "n", "\u043e": "o", "\u043f": "p", "\u0440": "r", "\u0441": "s", "\u0442": "t", "\u0443": "u",
    "\u0444": "f", "\u0445": "h", "\u0446": "ts", "\u0447": "ch", "\u0448": "sh", "\u0449": "sch",
    "\u044a": "", "\u044b": "y", "\u044c": "", "\u044d": "e", "\u044e": "yu", "\u044f": "ya",
    "\u0410": "A", "\u0411": "B", "\u0412": "V", "\u0413": "G", "\u0414": "D", "\u0415": "E", "\u0401": "E",
    "\u0416": "Zh", "\u0417": "Z", "\u0418": "I", "\u0419": "Y", "\u041a": "K", "\u041b": "L", "\u041c": "M",
    "\u041d": "N", "\u041e": "O", "\u041f": "P", "\u0420": "R", "\u0421": "S", "\u0422": "T", "\u0423": "U",
    "\u0424": "F", "\u0425": "H", "\u0426": "Ts", "\u0427": "Ch", "\u0428": "Sh", "\u0429": "Sch",
    "\u042a": "", "\u042b": "Y", "\u042c": "", "\u042d": "E", "\u042e": "Yu", "\u042f": "Ya",
    " ": "_",
}


def transliterate(text):
    """Simple Cyrillic-to-Latin transliteration for filesystem-safe names."""
    return "".join(_CYRILLIC_TO_LATIN.get(ch, ch) for ch in text)


def save_frame(frame, name, folder=DETECTIONS_DIR, jpeg_quality=80):
    """Save detection frame and return generated file path."""
    os.makedirs(folder, exist_ok=True)

    safe_name = transliterate(name)
    filename = f"{safe_name}_{int(time.time())}.jpg"
    path = os.path.join(folder, filename)

    cv2.imwrite(path, frame, [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)])
    return path


def imread_utf8(filepath):
    """Read image with UTF-8 path support on Windows."""
    try:
        if sys.platform == "win32":
            # Windows: read file as bytes and decode.
            with open(filepath, "rb") as f:
                data = np.frombuffer(f.read(), dtype=np.uint8)
            return cv2.imdecode(data, cv2.IMREAD_COLOR)
        # Linux/Mac: direct reading works.
        return cv2.imread(filepath)
    except Exception as e:
        print(f"Error reading image {filepath}: {e}")
        return None


