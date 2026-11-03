"""Shared data code for the step counter: loading sessions, scaling, step labels and windows.

The sensor is worn just below the RIGHT knee, so the gyro sees one big swing per stride
(right leg only). Steps = both feet, so there are two steps per stride.

Labels. Each recording comes with the number of steps counted out loud. To train on
every 0.5 s of data (not just one number per recording), mark_steps() finds the right leg's
swing peaks on the gyro, keeps the strongest ones so that 2 x strides matches the count,
and puts a left step halfway between each pair of right steps. That says WHEN each counted
step happened; the total always equals the count.
"""
import csv, json
from pathlib import Path
import numpy as np

RATE = 50                 # samples per second
WINDOW = 2 * RATE         # the network sees 2 s at a time
HOP = RATE // 2           # ...and counts the steps in the middle 0.5 s of it
CENTER = (WINDOW - HOP) // 2   # the middle 0.5 s starts 37 samples into the window
CHANNELS = ["ax", "ay", "az", "gx", "gy", "gz", "dist"]
MAX_DIST = 1.0            # m; "no echo" means nothing within range, drawn as this far
GYRO_SCALE = 250.0        # deg/s -> about +-2 for walking, the same size as the accel in g
HERE = Path(__file__).resolve().parent


def scale(raw):
    """(n, 7) raw units -> (n, 7) network inputs: accel in g, gyro / 250, distance in m (0 echo -> 1 m)."""
    x = np.array(raw, dtype=np.float32).reshape(-1, 7).copy()
    x[:, 3:6] /= GYRO_SCALE
    d = x[:, 6]
    d[(d <= 0) | (d > MAX_DIST)] = MAX_DIST
    return x


def load_sessions(folder=None):
    """Every data/*.csv with its .json -> list of dicts with x (scaled), meta, name."""
    folder = Path(folder or HERE / "data")
    sessions = []
    for path in sorted(folder.glob("*.csv")):
        meta_path = path.with_suffix(".json")
        if not meta_path.exists():
            print(f"Skipping {path.name}: no .json with the step count")
            continue
        meta = json.loads(meta_path.read_text())
        with open(path) as fh:
            rows = [[float(v) for v in row] for row in list(csv.reader(fh))[1:]]
        seq = np.array([r[0] for r in rows])
        gaps = int(np.sum(np.diff(seq) > 1))
        x = scale([r[1:] for r in rows])
        sessions.append({"name": path.stem, "x": x, "meta": meta, "gaps": gaps})
    return sessions


def swing_signal(x, start, end):
    """The gyro axis that swings the most, signed so the swing peak is positive, lightly smoothed."""
    g = x[:, 3:6]
    axis = int(np.argmax(g[start:end].std(axis=0)))
    s = g[:, axis]
    seg = s[start:end]
    if np.percentile(-seg, 99) > np.percentile(seg, 99):
        s = -s
    k = np.ones(5) / 5
    return np.convolve(s, k, mode="same"), axis


def find_peaks(s, start, end, min_gap):
    """Local maxima in [start, end), biggest first, at least min_gap samples apart."""
    i = np.arange(max(start, 1), min(end, len(s) - 1))
    cand = i[(s[i] > s[i - 1]) & (s[i] >= s[i + 1])]
    cand = cand[np.argsort(-s[cand])]
    kept = []
    for c in cand:
        if all(abs(c - k) >= min_gap for k in kept):
            kept.append(c)
    return np.array(kept, dtype=int)


def mark_steps(x, start, end, steps):
    """Times (in samples) of every counted step in [start, end). Returns (times, info)."""
    if steps <= 0:
        return np.array([]), {"strides": 0, "weakest": None}
    s, axis = swing_signal(x, start, end)
    peaks = find_peaks(s, start, end, min_gap=int(0.4 * RATE))     # strides are slower than 2.5 per second
    n_strides = int(round(steps / 2))
    right = np.sort(peaks[:n_strides]).astype(float)
    if len(right) == 0:
        return np.array([]), {"strides": 0, "weakest": None}
    period = np.median(np.diff(right)) if len(right) > 1 else RATE
    events = list(right)
    for a, b in zip(right, right[1:]):                              # left step between right steps
        if b - a < 1.6 * period:
            events.append((a + b) / 2)
    events.sort()
    # make the number of marks exactly the counted number: add half-strides at the ends, or trim
    front = True
    while len(events) < steps:
        t = events[0] - period / 2 if front else events[-1] + period / 2
        if start <= t < end:
            events.insert(0, t) if front else events.append(t)
        elif not (start <= events[0] - period / 2 < end) and not (start <= events[-1] + period / 2 < end):
            break
        front = not front
    while len(events) > steps:
        events.pop(0 if front else -1)
        front = not front
    info = {"strides": len(right), "axis": "xyz"[axis],
            "weakest": float(s[peaks[n_strides - 1]] / np.median(s[peaks[:n_strides]])) if len(peaks) >= n_strides else None}
    events = np.array(events)
    if len(events) < steps:            # could not place them all: spread the rest evenly so the total is right
        info["spread"] = steps - len(events)
    return events, info


def windows(x, start, end, step=HOP):
    """Window start indices whose middle 0.5 s tiles [start, end) (with step=HOP),
    and how much of each middle 0.5 s lies inside [start, end)."""
    starts, weights = [], []
    c = start
    while c < end:
        i = c - CENTER
        if i >= 0 and i + WINDOW <= len(x):
            starts.append(i)
            weights.append(min(HOP, end - c) / HOP)
        c += step
    return np.array(starts, dtype=int), np.array(weights)


def targets(events, starts, start, end, steps):
    """Steps inside each window's middle 0.5 s (clipped to [start, end))."""
    y = np.zeros(len(starts), dtype=np.float32)
    for k, i in enumerate(starts):
        lo, hi = max(i + CENTER, start), min(i + CENTER + HOP, end)
        y[k] = np.sum((events >= lo) & (events < hi))
    if len(events) < steps and steps > 0:    # marks we could not place: spread over the session
        y += (steps - len(events)) * HOP / max(end - start, 1)
    return y


def stretch(x, events, start, end, r):
    """Replay a recording r times as fast (r > 1 = quicker steps), with its step marks moved to match."""
    n = int(len(x) / r)
    src = np.arange(n) * r
    out = np.stack([np.interp(src, np.arange(len(x)), x[:, c]) for c in range(x.shape[1])], axis=1).astype(np.float32)
    return out, events / r, int(start / r), int(end / r)


def dataset(sessions, step=10, speeds=(0.85, 1.0, 1.15)):
    """All training windows: X (n, 7, 100), y (n,), session index (n,).
    Each recording is also replayed 15 % slower and faster, so the net sees more paces than you recorded."""
    X, Y, S = [], [], []
    for k, s in enumerate(sessions):
        m = s["meta"]
        events, _ = mark_steps(s["x"], m["start"], m["end"], m["steps"])
        for r in speeds:
            x, ev, a, b = stretch(s["x"], events, m["start"], m["end"], r) if r != 1.0 else (s["x"], events, m["start"], m["end"])
            starts, _ = windows(x, a, b, step=step)
            y = targets(ev, starts, a, b, m["steps"])
            for i, t in zip(starts, y):
                X.append(x[i:i + WINDOW].T)
                Y.append(t)
                S.append(k)
    return np.array(X, dtype=np.float32), np.array(Y, dtype=np.float32), np.array(S)
