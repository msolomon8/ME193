"""
Settings for The Impenetrable Safe.

Everything you might want to change lives in config.json (created on first run).
The web interface edits it for you (camera, secret code).
"""
import json
import os
from pathlib import Path

ROOT = Path(__file__).parent
# FINGER_LOCK_DATA lets tests point at a scratch folder instead of your real photos
DATA_DIR = Path(os.environ.get("FINGER_LOCK_DATA", ROOT / "data"))
MODEL_DIR = Path(os.environ.get("FINGER_LOCK_MODELS", ROOT / "models"))
MODEL_PATH = MODEL_DIR / "finger_cnn.pt"
CONFIG_PATH = ROOT / "config.json"

CLASSES = [0, 1, 2, 3, 4, 5, 6]   # 0-5 = number of fingers (0 = fist), 6 = blank (no hand)
BLANK = 6
CLASS_NAMES = ["0", "1", "2", "3", "4", "5", "blank"]   # also the photo folder names
NOT_A_DIGIT = (BLANK,)           # only an empty box means "no digit"; a fist is the digit 0
# The network sees the WHOLE camera frame (with the red markers on both sides), shrunk to
# 112 x 64 pixels. 112:64 = 1.75, close to the webcam's 16:9 shape, so hands aren't squashed.
# Both numbers must be divisible by 8 (three 2x2 poolings).
IMG_W, IMG_H = 112, 64
SAVE_W, SAVE_H = 224, 128      # photos are saved at twice that size (room to change it later)

DEFAULTS = {
    "camera_index": 0,
    # area INSIDE the red frame, as fractions of the camera picture: [left, top, right, bottom].
    # Found automatically with the "Find red frame" button (box must be empty). None = whole picture.
    "roi": None,
    "secret_code": [1, 3, 5],      # three DIFFERENT digits 0-5 (0 = fist)
    "min_confidence": 0.80,        # model must be at least this sure
    "stable_seconds": 1.0,         # a digit must be held this long to count
    "clear_seconds": 0.5,          # hand must be gone this long before the next digit
    "digit_timeout": 20,           # seconds without progress -> give up and relock
    "unlock_seconds": 10,          # how long the safe stays open
    "denied_seconds": 2,
    "max_failures": 3,             # wrong codes in a row before lockout
    "lockout_seconds": 30,
}


def load():
    cfg = json.loads(json.dumps(DEFAULTS))  # deep copy
    if CONFIG_PATH.exists():
        saved = json.loads(CONFIG_PATH.read_text())
        for k, v in saved.items():
            if isinstance(v, dict) and isinstance(cfg.get(k), dict):
                cfg[k].update(v)
            else:
                cfg[k] = v
    return cfg


def save(cfg):
    CONFIG_PATH.write_text(json.dumps(cfg, indent=2))


def valid_code(code):
    return (isinstance(code, list) and len(code) == 3 and len(set(code)) == 3
            and all(isinstance(d, int) and 0 <= d <= 5 for d in code))
