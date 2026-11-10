"""
Camera + image preparation. The SAME functions are used for collecting photos,
training, live testing, and later on the UNO Q - so the network always sees images
prepared exactly the same way.
"""
import sys
import threading
import time

import cv2
import numpy as np
import torch

from config import IMG_H, IMG_W, SAVE_H, SAVE_W


def find_red_frame(frame_bgr):
    """Find the white area inside the red frame (the box must be EMPTY).
    Returns [left, top, right, bottom] as fractions of the picture, or None if not found.
    How: the paper is the biggest bright, colorless blob; its bounding box is the inside of
    the frame. Pulled in 1.5% on each side so no red edge sneaks in."""
    h, w = frame_bgr.shape[:2]
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    white = ((hsv[..., 1] < 60) & (hsv[..., 2] > 120)).astype(np.uint8)
    n, _, stats, _ = cv2.connectedComponentsWithStats(white)
    if n < 2:
        return None
    i = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    x, y, bw, bh, area = stats[i]
    if area < 0.2 * w * h:          # too small: probably not the paper
        return None
    m = 0.015
    return [round(x / w + m, 4), round(y / h + m, 4), round((x + bw) / w - m, 4), round((y + bh) / h - m, 4)]


def apply_roi(img_bgr, roi):
    """Cut out the area inside the red frame. Works on any picture size (fractions)."""
    if not roi:
        return img_bgr
    h, w = img_bgr.shape[:2]
    x0, y0, x1, y1 = int(roi[0] * w), int(roi[1] * h), int(roi[2] * w), int(roi[3] * h)
    return img_bgr[y0:y1, x0:x1]


def for_saving(frame_bgr):
    """Shrink a whole camera frame to the size we store on disk (224 x 128)."""
    return cv2.resize(frame_bgr, (SAVE_W, SAVE_H), interpolation=cv2.INTER_AREA)


PAPER_LEVEL = 220.0


def white_balance(img_bgr):
    """Make the white paper the SAME color and brightness in every picture (220,220,220),
    so room lighting, shadows and camera color settings can't change the network's answer.
    Paper pixels = bright and colorless; each color channel is scaled so their median hits 220."""
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    paper = (hsv[..., 1] < 70) & (hsv[..., 2] > 80)
    if paper.sum() < 0.05 * paper.size:          # almost no paper visible: leave it alone
        return img_bgr
    med = np.median(img_bgr[paper].astype(np.float32), axis=0)
    gain = PAPER_LEVEL / np.maximum(med, 1.0)
    return np.clip(img_bgr.astype(np.float32) * gain, 0, 255).astype(np.uint8)


def network_view(img_bgr, roi=None):
    """What the network sees: the inside of the red frame shrunk to 112 x 64, white-balanced (BGR)."""
    small = cv2.resize(apply_roi(img_bgr, roi), (IMG_W, IMG_H), interpolation=cv2.INTER_AREA)
    return white_balance(small)


def to_uint8_tensor(img_bgr, roi=None):
    """BGR image (any size) -> 3 x 64 x 112 uint8 RGB tensor. Shared by training and live use."""
    small = network_view(img_bgr, roi)
    rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
    return torch.from_numpy(np.ascontiguousarray(rgb)).permute(2, 0, 1)


def normalize(t_uint8):
    """uint8 0-255 -> float roughly -1..1 (centered around 0, like we did for the V data)."""
    return (t_uint8.float() / 255.0 - 0.5) / 0.5


def to_model_input(img_bgr, roi=None):
    return normalize(to_uint8_tensor(img_bgr, roi))


class Camera:
    """Reads frames in a background thread so the interface never waits on the camera."""

    def __init__(self, index=0):
        self.index = index
        self.cap = None
        self.frame = None
        self.lock = threading.Lock()
        self.running = False
        self.thread = None

    def open(self, index=None):
        self.close()
        if index is not None:
            self.index = index
        backend = cv2.CAP_AVFOUNDATION if sys.platform == "darwin" else cv2.CAP_ANY
        self.cap = cv2.VideoCapture(self.index, backend)
        if self.cap.isOpened():
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.running = True
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def _loop(self):
        while self.running:
            ok, frame = self.cap.read() if self.cap is not None else (False, None)
            if ok:
                with self.lock:
                    self.frame = frame
            else:
                time.sleep(0.2)

    def read(self):
        with self.lock:
            return None if self.frame is None else self.frame.copy()

    @property
    def ok(self):
        return self.frame is not None

    def close(self):
        self.running = False
        if self.thread is not None:
            self.thread.join(timeout=1)
        if self.cap is not None:
            self.cap.release()
        self.cap, self.frame, self.thread = None, None, None
