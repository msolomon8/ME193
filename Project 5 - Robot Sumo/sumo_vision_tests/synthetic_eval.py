"""Stress-test the sumo-vision detectors on controlled synthetic pictures: a floor that changes with distance, the camera's
exposure and white balance drifting, tape that renders dull or cyan, opponents of different colours.

It pulls the REAL detection functions out of a main.py (without running the app, which needs the UNO Q) and runs them, so
what is measured is what the board runs.

    python synthetic_eval.py "old/.../main.py" "new/.../main.py"                # on a grey floor
    python synthetic_eval.py "old/.../main.py" "new/.../main.py" --all-floors   # also on an orange wood floor

IMPORTANT: these are stress tests of structural weaknesses, not a measurement of your real camera. The numbers say which
failure modes a detector survives; the real answer comes from your own pictures (the web page, and vision_check.py).
"""

import ast
import json
import sys
import tempfile
import warnings
from pathlib import Path

import cv2
import numpy as np

H, W = 240, 320
HORIZON_ROW = int(H * 0.45)
CONSTANTS = {"FLIP_X", "HORIZON", "TAPE_STRIP", "BLUE_LO", "BLUE_HI", "FLOOR_DIST", "MIN_OPP_AREA", "SELF_MASK_BOTTOM",
             "LUMA_WEIGHT", "DRIFT_LIMIT", "FLOOR_FRAMES"}
FUNCTIONS = {"x_of", "find_tape", "find_opponent", "ground_rows", "floor_sample", "finish_floor", "learn_tape", "masked_row_median"}

# BGR colours
FLOOR = (118, 122, 126)                                                # a grey floor (BGR)
FLOORS = {"grey floor": (118, 122, 126), "orange wood floor": (60, 110, 170)}
# how blue tape can come out on a webcam: a normal blue, a pale one (low saturation), and hue drifting toward cyan and toward purple
TAPE = {"normal blue": (200, 90, 20), "pale blue": (225, 200, 170), "cyan-leaning blue": (190, 170, 40),
        "purple-leaning blue": (190, 70, 110)}
OPPONENT = {"dark": (35, 35, 40), "red": (40, 40, 200), "white": (225, 225, 225), "LEGO teal": (181, 163, 0)}
APRON = (200, 90, 20)


class Detector:
    """The detection functions of one main.py, run in a namespace of their own."""

    def __init__(self, path):
        src = open(path, newline="").read().replace("\r\n", "\n")
        keep = []
        for node in ast.parse(src).body:
            if isinstance(node, ast.FunctionDef) and node.name in FUNCTIONS:
                keep.append(node)
            elif isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id in CONSTANTS for t in node.targets):
                keep.append(node)
        self.ns = {"np": np, "cv2": cv2, "json": json, "Path": Path, "warnings": warnings,
                   "TAPE_FILE": Path(tempfile.mkdtemp()) / "tape_calibration.json"}
        exec(compile(ast.Module(body=keep, type_ignores=[]), str(path), "exec"), self.ns)
        self.new = "floor_sample" in self.ns                       # the upgraded detector has the per-row floor model
        self.default_range = (self.ns["BLUE_LO"].copy(), self.ns["BLUE_HI"].copy())

    def reset_tape(self):
        self.ns["BLUE_LO"], self.ns["BLUE_HI"] = (a.copy() for a in self.default_range)

    def learn_floor(self, frames):
        if self.new:
            self.ns["floor_bgr"], self.ns["floor_rows"] = self.ns["finish_floor"]([self.ns["floor_sample"](f) for f in frames])
        else:                                                        # what the original loop() does
            first = int(H * self.ns["TAPE_STRIP"])
            self.ns["floor_bgr"] = np.median([np.median(f[first:].reshape(-1, 3), axis=0) for f in frames], axis=0)

    def look(self, frame):
        """(tape_ahead, opponent seen, opponent x, opponent size)"""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        tape = self.ns["find_tape"](frame, hsv)[0] if self.new else self.ns["find_tape"](hsv)[0]
        seen, x, size, _, _ = self.ns["find_opponent"](frame, hsv)
        return tape, seen, x, size

    def learn_tape(self, frame, x, y):
        return self.ns["learn_tape"](frame, x, y) if self.new else None

    def set(self, name, value):
        self.ns[name] = value


def scene(rng, gradient=0.0, gain=1.0, tint=(1.0, 1.0), noise=3.0, tape=None, apron=0, opp=None, floor=FLOOR, vee=None):
    """A picture of the ring floor. gradient: the far floor is that much brighter (+) or darker (-) than the near floor.
    tint = (blue multiplier, red multiplier): the camera's white balance drifting."""
    img = np.zeros((H, W, 3), np.float32)
    img[:HORIZON_ROW] = (70, 80, 90)                                   # the wall above the ring
    rows = np.arange(HORIZON_ROW, H)
    frac = (H - 1 - rows) / (H - 1 - HORIZON_ROW)                      # 0 at the bottom row, 1 at the horizon
    vignette = 1.0 - 0.08 * ((np.arange(W) / (W - 1)) * 2 - 1) ** 2
    img[HORIZON_ROW:] = (np.array(floor, np.float32)[None, None, :] * (1.0 + gradient * frac)[:, None, None]
                         * vignette[None, :, None])
    if tape is not None:
        color, y0, y1 = tape
        img[y0:y1, :] = color
    if vee is not None:                                                # a ring corner in front of the camera: two bands meeting at an apex
        color, thickness = vee
        apex_y, mid = int(H * 0.80), W // 2
        for sign in (-1, 1):
            arm = np.array([[mid - thickness // 2, apex_y], [mid + thickness // 2, apex_y],
                            [mid + sign * (W // 2) + thickness // 2 * (1 if sign > 0 else -1) + (thickness if sign > 0 else -thickness), H],
                            [mid + sign * (W // 2), H]], np.int32)
            cv2.fillPoly(img, [arm], color)
    if opp is not None:
        color, x0, y0, x1, y1 = opp
        img[y0:y1, x0:x1] = color
    if apron:
        img[H - apron:, :] = APRON
    img *= gain
    img[..., 0] *= tint[0]
    img[..., 2] *= tint[1]
    img += rng.normal(0, noise, img.shape)
    return np.clip(img, 0, 255).astype(np.uint8)


GAINS = (0.75, 1.0, 1.25)                                              # exposure drifting +-25%
TINTS = ((1.0, 1.0), (1.12, 0.90), (0.90, 1.12))                       # white balance drifting toward blue / toward red
GRADIENTS = (-0.25, 0.0, 0.25)                                         # far floor darker / same / brighter than near floor
TAPE_BAND = (200, 215)                                                 # rows of the tape band (inside the strip in front)


def opponent_box(size, centre_x):
    """A rectangle covering `size` of the ground area, bottom edge near the robot."""
    area = size * (H - HORIZON_ROW) * W
    h = int(np.sqrt(area / 1.4)); w = int(1.4 * h)
    x0 = int(centre_x * W - w / 2); y1 = min(H - 4, 215)
    return x0, y1 - h, x0 + w, y1


def evaluate(det, seed=1, floor=FLOOR):
    """Run every scene on the floor of this colour. Returns {description: fraction}."""
    rng = np.random.default_rng(seed)
    out = {}
    opponents = dict(OPPONENT)
    opponents["floor-coloured"] = tuple(min(255, c + 4) for c in floor)      # a robot the colour of this floor
    # 1. empty ring: nothing should be reported
    fp_opp = fp_tape = n = 0
    for grad in GRADIENTS:
        det.reset_tape()
        det.learn_floor([scene(rng, floor=floor, gradient=grad) for _ in range(20)])  # learned in nominal conditions, in this room
        for gain in GAINS:
            for tint in TINTS:
                for _ in range(4):
                    tape, seen, _, _ = det.look(scene(rng, floor=floor, gradient=grad, gain=gain, tint=tint))
                    fp_opp += seen
                    fp_tape += tape > 0.08
                    n += 1
    out["empty ring: reports an opponent that isn't there"] = fp_opp / n
    out["empty ring: reports tape that isn't there"] = fp_tape / n
    # 1b. a ring corner in view (a V of tape right in front, like the real camera showed: tape_ahead about 0.21), no opponent
    CORNER, BIG_CORNER = (TAPE["normal blue"], 80), (TAPE["normal blue"], 110)
    det.reset_tape()
    det.learn_floor([scene(rng, floor=floor) for _ in range(20)])
    hits = n = 0
    for grad in GRADIENTS:
        for gain in GAINS:
            for tint in TINTS:
                for _ in range(2):
                    hits += det.look(scene(rng, floor=floor, gradient=grad, gain=gain, tint=tint, vee=CORNER))[1]
                    n += 1
    out["empty ring with a ring corner in view: reports an opponent"] = hits / n
    # 1c. the floor RELEARNED with a big ring corner in view (tape_ahead about 0.24), then the same ring with no tape in view
    if det.new:
        det.reset_tape()
        det.learn_floor([scene(rng, floor=floor, vee=BIG_CORNER) for _ in range(20)])
        hits = n = 0
        for grad in GRADIENTS:
            for gain in GAINS:
                for tint in TINTS:
                    for _ in range(2):
                        hits += det.look(scene(rng, floor=floor, gradient=grad, gain=gain, tint=tint))[1]
                        n += 1
        out["floor relearned with a big ring corner in view, then no tape: reports an opponent"] = hits / n
    # 2. the robot's own apron in view (3% of the picture's rows)
    det.reset_tape()
    det.learn_floor([scene(rng, floor=floor) for _ in range(20)])
    out["own apron in 3% of the picture: false tape"] = np.mean([det.look(scene(rng, floor=floor, apron=8))[0] > 0.08 for _ in range(20)])
    if det.new:
        det.set("SELF_MASK_BOTTOM", 0.10)
        det.learn_floor([scene(rng, floor=floor) for _ in range(20)])
        out["  same, with SELF_MASK_BOTTOM = 0.10: false tape"] = np.mean([det.look(scene(rng, floor=floor, apron=8))[0] > 0.08 for _ in range(20)])
        out["  same, with SELF_MASK_BOTTOM = 0.10: false opponent"] = np.mean([det.look(scene(rng, floor=floor, apron=8))[1] for _ in range(20)])
        det.set("SELF_MASK_BOTTOM", 0.0)
        det.learn_floor([scene(rng, floor=floor) for _ in range(20)])
    # 3. opponents of different colours
    for name, color in opponents.items():
        hit = n = 0
        for grad in (0.0, 0.25):
            det.reset_tape()
            det.learn_floor([scene(rng, floor=floor, gradient=grad) for _ in range(20)])
            for gain in GAINS:
                for size in (0.08, 0.25):
                    for cx in (0.3, 0.5, 0.7):
                        _, seen, _, _ = det.look(scene(rng, floor=floor, gradient=grad, gain=gain, opp=(color, *opponent_box(size, cx))))
                        hit += seen
                        n += 1
        out[f"opponent found: {name}"] = hit / n
    # 4. tape that the camera renders differently. The new detector is taught by ONE click at normal exposure; then the light drifts.
    det.reset_tape()
    det.learn_floor([scene(rng, floor=floor) for _ in range(20)])
    for name, color in TAPE.items():
        default_hits = clicked_hits = n = 0
        for gain in GAINS:
            for tint in TINTS:
                det.reset_tape()
                frame = scene(rng, floor=floor, gain=gain, tint=tint, tape=(color, *TAPE_BAND))
                default_hits += det.look(frame)[0] > 0.08
                if det.new:
                    det.learn_tape(scene(rng, floor=floor, tape=(color, *TAPE_BAND)), 0.5, (TAPE_BAND[0] + TAPE_BAND[1]) / 2 / (H - 1))
                    clicked_hits += det.look(frame)[0] > 0.08
                n += 1
        out[f"tape seen: {name} (starting range)"] = default_hits / n
        if det.new:
            out[f"tape seen: {name} (after one click)"] = clicked_hits / n
    if det.new:                                                        # a careless click on the very edge of the tape (the patch includes floor)
        sloppy = n = 0
        for gain in GAINS:
            for tint in TINTS:
                det.reset_tape()
                det.learn_tape(scene(rng, floor=floor, tape=(TAPE["normal blue"], *TAPE_BAND)), 0.5, (TAPE_BAND[0] + 1) / (H - 1))
                sloppy += det.look(scene(rng, floor=floor, gain=gain, tint=tint, tape=(TAPE["normal blue"], *TAPE_BAND)))[0] > 0.08
                sloppy_fp = det.look(scene(rng, floor=floor, gain=gain, tint=tint))[0] > 0.08
                out["tape: false tape in an empty ring after a careless click"] = max(out.get("tape: false tape in an empty ring after a careless click", 0), float(sloppy_fp))
                n += 1
        out["tape seen: normal blue (after a careless click on its edge)"] = sloppy / n
    det.reset_tape()
    # 5. a LEGO teal robot in the strip in front must not read as tape
    det.reset_tape()
    det.learn_floor([scene(rng, floor=floor) for _ in range(20)])
    teal = np.mean([det.look(scene(rng, floor=floor, gain=g, tint=t, opp=(OPPONENT["LEGO teal"], *opponent_box(0.2, 0.5))))[0] > 0.08
                    for g in GAINS for t in TINTS])
    out["a teal opponent misread as tape (starting range)"] = teal
    if det.new:                                                        # after one click on normal blue
        det.learn_tape(scene(rng, floor=floor, tape=(TAPE["normal blue"], *TAPE_BAND)), 0.5, (TAPE_BAND[0] + TAPE_BAND[1]) / 2 / (H - 1))
        out["a teal opponent misread as tape (after one click)"] = np.mean(
            [det.look(scene(rng, floor=floor, gain=g, tint=t, opp=(OPPONENT["LEGO teal"], *opponent_box(0.2, 0.5))))[0] > 0.08
             for g in GAINS for t in TINTS])
        det.reset_tape()
    if det.new:                                                        # a stray click on the plain floor must not become the tape colour
        det.reset_tape()
        det.learn_floor([scene(rng, floor=floor) for _ in range(20)])
        before = (det.ns["BLUE_LO"].copy(), det.ns["BLUE_HI"].copy())
        refused = []
        for x, y in ((0.5, 0.75), (0.2, 0.9), (0.8, 0.6)):
            det.learn_tape(scene(rng, floor=floor), x, y)
            refused.append(bool((det.ns["BLUE_LO"] == before[0]).all() and (det.ns["BLUE_HI"] == before[1]).all()))
            det.reset_tape()
        out["a click on the plain floor is refused"] = float(np.mean(refused))
    import time
    det.learn_floor([scene(rng, floor=floor) for _ in range(20)])
    frames = [scene(rng, floor=floor) for _ in range(60)]
    t = time.perf_counter()
    for f in frames:
        det.look(f)
    out["__ms"] = (time.perf_counter() - t) / len(frames) * 1000.0
    return out


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    floors = FLOORS if "--all-floors" in sys.argv else {"grey floor": FLOOR}
    if not args:
        sys.exit(__doc__)
    for fname, fcolor in floors.items():
        results = {p: evaluate(Detector(p), floor=fcolor) for p in args}
        labels = [k for k in results[args[0]] if not k.startswith("__")]
        for p in args[1:]:
            for k in results[p]:
                if k not in labels and not k.startswith("__"):
                    labels.append(k)
        print(f"\n=== on a {fname} ===")
        print(f"{'':<60}" + "".join(f"{('detector ' + str(i + 1)):>13}" for i in range(len(args))))
        for k in labels:
            row = "".join(f"{results[p][k]:>12.0%} " if k in results[p] else f"{'-':>13}" for p in args)
            print(f"{k:<60}{row}")
        print("time per picture on this computer (relative, not the UNO Q): " + ", ".join(f"{results[p]['__ms']:.1f} ms" for p in args))
    print("\nrows about false reports (empty ring, own apron, teal misread): lower is better. Rows about 'found', 'seen' and 'refused': higher is better.")


if __name__ == "__main__":
    main()
