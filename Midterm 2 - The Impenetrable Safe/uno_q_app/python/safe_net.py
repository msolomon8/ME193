"""
The finger-counting CNN, rewritten with only numpy so it runs on the UNO Q without PyTorch.
Same math as finger_lock/model.py: 3 x (conv -> ReLU -> 2x2 max-pool), flatten -> linear -> softmax.
BatchNorm is folded into the conv weights by finger_lock/export_for_uno.py.
"""
import cv2
import numpy as np


def conv2d(x, w, b):
    """x: C x H x W, w: F x C x k x k ('same' padding). im2col + one matrix multiply."""
    c, h, wd = x.shape
    f, _, k, _ = w.shape
    p = k // 2
    xp = np.pad(x, ((0, 0), (p, p), (p, p)))
    cols = np.lib.stride_tricks.sliding_window_view(xp, (k, k), axis=(1, 2))   # C x H x W x k x k
    cols = cols.transpose(1, 2, 0, 3, 4).reshape(h * wd, c * k * k)
    return (cols @ w.reshape(f, -1).T + b).T.reshape(f, h, wd)


def maxpool2(x):
    c, h, w = x.shape
    return x[:, :h // 2 * 2, :w // 2 * 2].reshape(c, h // 2, 2, w // 2, 2).max(axis=(2, 4))


def white_balance(img_bgr, level=220.0):
    """Same as finger_lock/camera.py: make the white paper (220,220,220) in every picture."""
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    paper = (hsv[..., 1] < 70) & (hsv[..., 2] > 80)
    if paper.sum() < 0.05 * paper.size:
        return img_bgr
    med = np.median(img_bgr[paper].astype(np.float32), axis=0)
    return np.clip(img_bgr.astype(np.float32) * (level / np.maximum(med, 1.0)), 0, 255).astype(np.uint8)


class NumpyCNN:
    def __init__(self, path):
        d = np.load(path, allow_pickle=False)
        self.convs = [(d[f"conv{i}_w"], d[f"conv{i}_b"]) for i in (1, 2, 3)]
        self.lin_w, self.lin_b = d["lin_w"], d["lin_b"]
        self.roi = [float(v) for v in d["roi"]] if d["roi"].size == 4 else None
        self.img_w, self.img_h = int(d["img_w"]), int(d["img_h"])
        self.val_acc = float(d["val_acc"])
        self.white_balance = bool(d["white_balance"]) if "white_balance" in d.files else False

    def prepare(self, frame_bgr):
        """Camera frame -> 3 x 64 x 112 float image, prepared EXACTLY like in training:
        crop inside the red frame, shrink, BGR->RGB, scale to about -1..1."""
        img = frame_bgr
        if self.roi:
            h, w = img.shape[:2]
            img = img[int(self.roi[1] * h):int(self.roi[3] * h), int(self.roi[0] * w):int(self.roi[2] * w)]
        small = cv2.resize(img, (self.img_w, self.img_h), interpolation=cv2.INTER_AREA)
        if self.white_balance:
            small = white_balance(small)
        rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB).astype(np.float32)
        return ((rgb / 255.0 - 0.5) / 0.5).transpose(2, 0, 1)

    def probabilities(self, x_chw):
        x = x_chw
        for w, b in self.convs:
            x = maxpool2(np.maximum(conv2d(x, w, b), 0))
        z = self.lin_w @ x.reshape(-1) + self.lin_b
        e = np.exp(z - z.max())
        return (e / e.sum()).tolist()
