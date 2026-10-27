"""Gait Identifier - shared feature extraction and tree evaluation.

Pure Python (no numpy) so the exact same code runs on the UNO Q and on the
laptop (ME193 `Project 6 - Gait Identifier/train.py` imports this file).
"""
import math

RATE = 50              # samples per second (set by the sketch)
WINDOW = 2 * RATE      # 2 s of samples per decision
HOP = RATE // 2        # a new decision every 0.5 s
CLASSES = ["standing", "walking", "running", "biking"]
ACCEL_SCALE = 2048.0   # counts per g     (MPU-6050 at +-16 g)
GYRO_SCALE = 16.4      # counts per deg/s (MPU-6050 at +-2000 deg/s)
AXES = ["ax", "ay", "az", "gx", "gy", "gz"]


def to_units(raw):
    """Raw MPU-6050 counts -> (ax, ay, az in g, gx, gy, gz in deg/s)."""
    return tuple(v / ACCEL_SCALE for v in raw[:3]) + tuple(v / GYRO_SCALE for v in raw[3:])


def _mean(v):
    return sum(v) / len(v)


def _std(v, m=None):
    m = _mean(v) if m is None else m
    return math.sqrt(sum((x - m) ** 2 for x in v) / len(v))


def _cadence(v):
    """Strongest repetition rate (Hz) of a signal, and how strong it is (0-1),
    from the autocorrelation. Steps and pedal strokes are 0.5-4 Hz."""
    m = _mean(v)
    d = [x - m for x in v]
    energy = sum(x * x for x in d)
    if energy < 1e-9:
        return 0.0, 0.0
    best_lag, best_r = 0, 0.0
    for lag in range(RATE // 4, min(2 * RATE, len(d) - 10) + 1):   # 4 Hz down to 0.5 Hz
        r = sum(d[i] * d[i + lag] for i in range(len(d) - lag)) / energy
        if r > best_r:
            best_lag, best_r = lag, r
    if best_lag == 0:
        return 0.0, 0.0
    return RATE / best_lag, best_r


def features(window):
    """window: list of (ax, ay, az, gx, gy, gz) in g and deg/s -> dict of features."""
    f = {}
    cols = list(zip(*window))
    for name, c in zip(AXES, cols):
        m = _mean(c)
        f["mean_" + name] = m          # for accel: which way gravity points = how the leg is tilted
        f["std_" + name] = _std(c, m)  # how much it shakes
        f["max_" + name] = max(c)      # peak (model 1 uses one of the accel peaks)
        f["min_" + name] = min(c)
    amag = [math.sqrt(a * a + b * b + c * c) for a, b, c in zip(cols[0], cols[1], cols[2])]
    gmag = [math.sqrt(a * a + b * b + c * c) for a, b, c in zip(cols[3], cols[4], cols[5])]
    f["mean_amag"] = _mean(amag)
    f["std_amag"] = _std(amag)
    f["max_amag"] = max(amag)
    f["range_amag"] = max(amag) - min(amag)
    f["mean_gmag"] = _mean(gmag)
    f["std_gmag"] = _std(gmag)
    f["max_gmag"] = max(gmag)
    # Stride / pedal rate from the gyro axis that swings the most (the leg's main rotation axis).
    # Not the magnitude: |sin| repeats twice per stride and would double the rate.
    # The gyro sees every stride and pedal stroke, impact or not. One ankle = one cycle per stride.
    swing = max(cols[3:], key=lambda c: _std(c))
    f["cadence_hz"], f["periodicity"] = _cadence(swing)
    return f


FEATURE_NAMES = list(features([(0, 0, 1, 0, 0, 0), (0.1, 0, 1, 1, 0, 0)] * (WINDOW // 2)).keys())


def predict(tree, f):
    """Walk a tree exported by train.py. Nodes are parallel lists; left == -1 marks a leaf."""
    n = 0
    while tree["left"][n] != -1:
        n = tree["left"][n] if f[tree["feature"][n]] <= tree["threshold"][n] else tree["right"][n]
    return tree["label"][n]
