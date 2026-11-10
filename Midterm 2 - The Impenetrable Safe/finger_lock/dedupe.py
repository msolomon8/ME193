"""
Remove near-duplicate photos. Inside a burst, a photo that looks almost exactly like the
last photo we KEPT is moved to data_duplicates/<class>/ (not deleted - move it back to undo).

    ~/.venvs/impenetrable-safe/bin/python dedupe.py
(or use the "Remove near-duplicates" button in the Train tab)
"""
import shutil
from collections import defaultdict

import cv2
import numpy as np

from config import CLASS_NAMES, CLASSES, DATA_DIR
from dataset import burst_id, class_dir

DUP_DIR = DATA_DIR.parent / "data_duplicates"
THRESHOLD = 3.0     # average pixel difference (0-255) below which two photos count as the same


def dedupe(log=print):
    moved = {}
    for c in CLASSES:
        bursts = defaultdict(list)
        for p in sorted(class_dir(c).glob("*.png")):
            bursts[burst_id(p)].append(p)
        n = 0
        for photos in bursts.values():
            last = None
            for p in photos:
                img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
                if img is None:
                    continue
                small = cv2.resize(img, (56, 32), interpolation=cv2.INTER_AREA).astype(np.float32)
                if last is not None and np.abs(small - last).mean() < THRESHOLD:
                    dest = DUP_DIR / CLASS_NAMES[c]
                    dest.mkdir(parents=True, exist_ok=True)
                    shutil.move(str(p), dest / p.name)
                    n += 1
                else:
                    last = small
        moved[CLASS_NAMES[c]] = n
    log("Moved near-duplicates to data_duplicates/: " + ", ".join(f"{k}: {v}" for k, v in moved.items()))
    return moved


if __name__ == "__main__":
    dedupe()
