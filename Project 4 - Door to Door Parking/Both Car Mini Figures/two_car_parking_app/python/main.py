# Project 4 - Two Car Parking, UNO Q Linux side (runs in Arduino App Lab on the GREEN car)
#
# The computer opens this app's web page (https://<board-ip>:7000) and the page streams
# the computer's webcam to the board. Here, both trained YOLO models (green + blue,
# exported to ONNX) run on every frame:
#   - GREEN minifig: this car drives forward/backward until the green minifig is on the
#     LEFT of the image (x = 0.25), and shows its position as a dot on the LED matrix.
#   - BLUE minifig: its box is sent over MQTT to the blue car (car_blue_app), which drives
#     until the blue minifig is on the RIGHT (x = 0.75).
# The page draws both boxes and both stopping lines on the computer screen.
from arduino.app_utils import *
from arduino.app_bricks.web_ui import WebUI
import paho.mqtt.client as mqtt
from pathlib import Path
import base64, json, time, threading
import cv2
import numpy as np

from detector import MinifigDetector

MODELS = Path(__file__).parent / "models"

# ---- Detection ----
CONFIDENCE = 0.5            # ignore detections less sure than this
SAME_FIGURE_IOU = 0.5       # green + blue boxes overlapping this much = one figure seen twice

# ---- MQTT (to the blue car) ----
BROKER = "test.mosquitto.org"
PORT = 1883
BLUE_TOPIC = "ME193/minifig/tashamia/blue"   # must match car_blue_app
CLIENT_ID = "ME193-two-car-parking"          # unique (car_blue_app uses ME193-minifig-blue-car)

# ---- Display ----
FLIP_X = False
FLIP_Y = True

# ---- Driving (this car = green) ----
TARGET_X = 0.25     # where the green minifig should stop: left side of the screen
DEADBAND = 0.04     # "close enough" -> stop
RESUME = 0.08       # once stopped, only move again if it drifts this far (prevents twitching)
DIRECTION = -1      # set to -1 if the car drives AWAY from the target (same as unoq_app)
MIN_SPEED = 110     # slowest PWM that actually moves the car
MAX_SPEED = 150     # a bit lower than unoq_app: detection on the board is slower -> overshoot
GAIN = 600          # speed per unit of error
TIMEOUT = 1.0       # no fresh green detection for this long -> stop

green_detector = MinifigDetector(MODELS / "green_minifig.onnx", CONFIDENCE)
blue_detector = MinifigDetector(MODELS / "blue_minifig.onnx", CONFIDENCE)

latest_jpeg = None          # newest frame from the browser (bytes), waiting to be processed
frame_ready = threading.Event()
green_seen = None           # (x, y, time) of the green minifig, normalized 0-1
lock = threading.Lock()

client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id=CLIENT_ID)
client.connect(BROKER, PORT, keepalive=30)
client.loop_start()

ui = WebUI(use_tls=True)    # HTTPS: browsers only allow webcam access on secure pages


def on_frame(sid, data):
    """Browser sent a webcam frame (JPEG bytes, or a base64 data URL string). Just keep the newest."""
    global latest_jpeg
    if isinstance(data, str):
        data = base64.b64decode(data.split(",", 1)[-1])
    latest_jpeg = data
    frame_ready.set()


ui.on_message("frame", on_frame)


def iou(a, b):
    """Overlap of two (x, y, w, h, ...) center boxes: 0 = apart, 1 = identical."""
    iw = min(a[0] + a[2] / 2, b[0] + b[2] / 2) - max(a[0] - a[2] / 2, b[0] - b[2] / 2)
    ih = min(a[1] + a[3] / 2, b[1] + b[3] / 2) - max(a[1] - a[3] / 2, b[1] - b[3] / 2)
    inter = max(0.0, iw) * max(0.0, ih)
    union = a[2] * a[3] + b[2] * b[3] - inter
    return inter / union if union > 0 else 0.0


def as_dict(box):
    return None if box is None else {"x": box[0], "y": box[1], "w": box[2], "h": box[3], "conf": box[4]}


def detection_thread():
    """Run both models on the newest browser frame, as fast as the board allows."""
    global green_seen
    while True:
        frame_ready.wait()
        frame_ready.clear()
        frame = cv2.imdecode(np.frombuffer(latest_jpeg, np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
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
        ms = (time.time() - t0) * 1000

        if green:
            with lock:
                green_seen = (green[0], green[1], time.time())
        if blue:
            # Same JSON as minifig_mqtt.py, in pixels, for car_blue_app
            h, w = frame.shape[:2]
            client.publish(BLUE_TOPIC, json.dumps({
                "x": round(blue[0] * w), "y": round(blue[1] * h),
                "width": round(blue[2] * w), "height": round(blue[3] * h),
                "img_w": w, "img_h": h,
            }))

        # Tells the page to draw the boxes AND that it can send the next frame
        ui.send_message("detections", {"green": as_dict(green), "blue": as_dict(blue), "ms": round(ms)})


threading.Thread(target=detection_thread, daemon=True).start()

last_cell = None
stopped = True
last_mode = None


def loop():
    global last_cell, stopped, last_mode
    with lock:
        cur = green_seen

    if cur is None or time.time() - cur[2] > TIMEOUT:
        speed, cell, mode, x = 0, (-1, -1), "NO GREEN MINIFIG -> STOP", None
        stopped = True
    else:
        x, y, _ = cur
        x = min(max(x, 0.0), 1.0)
        y = min(max(y, 0.0), 1.0)
        dx = 1 - x if FLIP_X else x
        dy = 1 - y if FLIP_Y else y
        cell = (round(dx * 12), round(dy * 7))

        error = x - TARGET_X
        limit = RESUME if stopped else DEADBAND
        if abs(error) <= limit:
            speed, mode, stopped = 0, "PARKED ON THE LEFT -> STOP", True
        else:
            stopped = False
            mag = min(MAX_SPEED, max(MIN_SPEED, int(abs(error) * GAIN)))
            speed = DIRECTION * (mag if error < 0 else -mag)
            mode = "FORWARD" if speed > 0 else "BACKWARD"

    Bridge.call("drive", speed, speed)     # resent every loop so the sketch watchdog stays fed

    if cell != last_cell:
        Bridge.call("dot", cell[0], cell[1])
        last_cell = cell

    if mode != last_mode:                  # print only when the behavior changes
        print(mode if x is None else f"{mode}  (x={x:.2f}, speed={speed})")
        last_mode = mode

    time.sleep(0.05)                        # ~20 updates per second


App.run(user_loop=loop)
