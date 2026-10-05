# Project 4 - Two Minifig Parking, UNO Q Linux side (runs in Arduino App Lab)
#
# Runs BOTH minifig detectors (green + blue) on the UNO Q's own USB camera,
# shows both minifigs on the LED matrix, and drives forward/backward until the
# two minifigs sit on either side of the camera image -- i.e. their midpoint
# is in the middle of the screen, so the car is parked between them.
#
# No MQTT and no Mac needed: detection runs on the board.
from arduino.app_utils import *
from arduino.app_peripherals.camera import Camera
from pathlib import Path
import time, threading

from detector import MinifigDetector

MODELS = Path(__file__).parent / "models"

# ---- Camera ----
CAMERA_SOURCE = "usb:0"     # first USB camera; try "usb:1" if it opens the wrong one
RESOLUTION = (640, 480)
CONFIDENCE = 0.5            # ignore detections less sure than this
SAME_FIGURE_IOU = 0.5       # green + blue boxes overlapping this much = one figure seen twice

# ---- Display ----
FLIP_X = False
FLIP_Y = True
GREEN_BRIGHTNESS = 255      # green minifig = bright dot
BLUE_BRIGHTNESS = 40        # blue minifig = dim dot

# ---- Driving ----
TARGET_X = 0.5      # where the midpoint should end up (0.5 = middle of the screen)
DEADBAND = 0.04     # "close enough" -> stop
RESUME = 0.08       # once stopped, only move again if it drifts this far (prevents twitching)
DIRECTION = 1       # set to -1 if the car drives AWAY from the center
MIN_SPEED = 110     # slowest PWM that actually moves the car
MAX_SPEED = 150     # lower than the MQTT app: detection on the board is slower -> more overshoot
GAIN = 600          # speed per unit of error
TIMEOUT = 1.5       # no fresh detection for this long -> stop

green_detector = MinifigDetector(MODELS / "green_minifig.onnx", CONFIDENCE)
blue_detector = MinifigDetector(MODELS / "blue_minifig.onnx", CONFIDENCE)

latest = None       # (green_box, blue_box, time); boxes are (x, y, w, h, conf) normalized 0-1, or None
lock = threading.Lock()


def iou(a, b):
    """Overlap of two (x, y, w, h, ...) center boxes: 0 = apart, 1 = identical."""
    ax1, ay1, ax2, ay2 = a[0] - a[2] / 2, a[1] - a[3] / 2, a[0] + a[2] / 2, a[1] + a[3] / 2
    bx1, by1, bx2, by2 = b[0] - b[2] / 2, b[1] - b[3] / 2, b[0] + b[2] / 2, b[1] + b[3] / 2
    iw = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    ih = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = iw * ih
    union = a[2] * a[3] + b[2] * b[3] - inter
    return inter / union if union > 0 else 0.0


def detection_thread():
    """Grab frames and run both models as fast as the board allows."""
    global latest
    with Camera(CAMERA_SOURCE, resolution=RESOLUTION) as camera:
        print(f"Camera {CAMERA_SOURCE} started")
        while True:
            frame = camera.capture()
            if frame is None:
                time.sleep(0.05)
                continue
            t0 = time.time()
            green = green_detector.detect(frame)
            blue = blue_detector.detect(frame)
            # Both models fired on the same spot -> it's one figure; keep the surer one
            if green and blue and iou(green, blue) > SAME_FIGURE_IOU:
                if green[4] >= blue[4]:
                    blue = None
                else:
                    green = None
            with lock:
                latest = (green, blue, time.time())
            print(f"green={fmt(green)}  blue={fmt(blue)}  ({(time.time() - t0) * 1000:.0f} ms)")


def fmt(box):
    return "-" if box is None else f"x={box[0]:.2f} ({box[4]:.2f})"


def to_cell(box):
    if box is None:
        return (-1, -1)
    x = min(max(box[0], 0.0), 1.0)
    y = min(max(box[1], 0.0), 1.0)
    dx = 1 - x if FLIP_X else x
    dy = 1 - y if FLIP_Y else y
    return (round(dx * 12), round(dy * 7))


threading.Thread(target=detection_thread, daemon=True).start()

last_cells = None
stopped = True
last_mode = None


def loop():
    global last_cells, stopped, last_mode
    with lock:
        cur = latest

    if cur is None or time.time() - cur[2] > TIMEOUT:
        green = blue = None
    else:
        green, blue = cur[0], cur[1]

    if green and blue:
        mid = (green[0] + blue[0]) / 2
        error = mid - TARGET_X
        limit = RESUME if stopped else DEADBAND
        if abs(error) <= limit:
            speed, stopped = 0, True
            mode = "PARKED BETWEEN BOTH"
        else:
            stopped = False
            mag = min(MAX_SPEED, max(MIN_SPEED, int(abs(error) * GAIN)))
            speed = DIRECTION * (mag if error < 0 else -mag)
            mode = "FORWARD" if speed > 0 else "BACKWARD"
        left, right = ("green", "blue") if green[0] < blue[0] else ("blue", "green")
        mode += f"  ({left} left, {right} right, midpoint x={mid:.2f})"
    else:
        speed, stopped = 0, True
        if green or blue:
            mode = f"ONLY {'GREEN' if green else 'BLUE'} -> STOP"
        else:
            mode = "NO MINIFIGS -> STOP"

    Bridge.call("drive", speed, speed)     # resent every loop so the sketch watchdog stays fed

    cells = to_cell(green) + to_cell(blue)
    if cells != last_cells:
        Bridge.call("dots", cells[0], cells[1], GREEN_BRIGHTNESS, cells[2], cells[3], BLUE_BRIGHTNESS)
        last_cells = cells

    if mode.split("  ")[0] != (last_mode or "").split("  ")[0]:   # print only when the behavior changes
        print(mode)
    last_mode = mode

    time.sleep(0.05)                        # ~20 updates per second


App.run(user_loop=loop)
