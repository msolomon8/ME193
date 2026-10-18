# Robot Sumo - UNO Q vision sensor (Linux side, runs in Arduino App Lab)
# The UNO Q is only a SENSOR here: it never drives. It reads the Logitech webcam on the
# robot, looks for two things, and publishes them over MQTT for the laptop's behavior tree:
#   1. the blue tape ring  -> an early warning that the edge is coming (the color sensor
#                             only finds out once the robot is already ON the tape)
#   2. the opponent        -> anything near the ground that isn't ring floor or tape
# The laptop fuses this with the LEGO color sensor (see "Project 5 - Project 5 - Robot Sumo/sumo_brain.py" in ME193).
#
# Message (topic ME193/Rogers/MK), about 10 per second:
#   {"opp_seen": true, "opp_x": -0.3, "opp_size": 0.12, "tape_ahead": 0.05, "tape_x": 0.4}
#   opp_x / tape_x: -1 = far left of the image, 0 = center, +1 = far right
#   opp_size:       fraction of the ground area the opponent covers (bigger = closer)
#   tape_ahead:     fraction of the bottom of the image that is blue tape
#
# Floor calibration: for the first FLOOR_FRAMES frames, the robot should be sitting in the
# ring looking at empty floor. The app learns the floor's color from them.
from arduino.app_utils import *
import paho.mqtt.client as mqtt
import cv2
import numpy as np
import json, time
from pathlib import Path

# ---- MQTT ----
BROKER = "test.mosquitto.org"
PORT = 1883
TOPIC = "ME193/Rogers/MK"
CLIENT_ID = "ME193-MK-sumo-vision"

# ---- Camera ----
# /dev/video0 and video1 are the board's built-in video encoder/decoder (qcom-venus), not
# cameras, so the webcam is found by name instead of by number.
FRAME_W, FRAME_H = 320, 240   # small frames keep the loop fast on the board
FLIP_X = False       # set True if left/right come out mirrored

# ---- Image regions (fractions of the image height, 0 = top) ----
HORIZON = 0.45       # above this is past the ring (walls, people) -> ignored
TAPE_STRIP = 0.70    # tape_ahead only looks below this (the floor right in front)

# ---- Blue tape (OpenCV HSV: H 0-179, S 0-255, V 0-255) - TUNE on the real tape ----
BLUE_LO = np.array([95, 80, 40])
BLUE_HI = np.array([130, 255, 255])

# ---- Opponent ----
FLOOR_FRAMES = 20    # frames used to learn the floor color at startup
FLOOR_DIST = 60      # how different from the floor (in BGR) a pixel must be to count
MIN_OPP_AREA = 0.02  # ignore blobs smaller than this fraction of the ground area

floor_samples = []
floor_bgr = None
last_seen = None


def webcam_devices():
    """/dev/videoN paths that aren't the board's built-in qcom-venus encoder/decoder."""
    found = []
    for d in sorted(Path("/sys/class/video4linux").glob("video*"), key=lambda p: int(p.name[5:])):
        name = (d / "name").read_text().strip() if (d / "name").exists() else ""
        if "venus" not in name:
            found.append((f"/dev/{d.name}", name))
    return found


def open_camera():
    """Open the first webcam device that actually gives a frame (a USB webcam also
    makes a metadata device that can't)."""
    for path, name in webcam_devices():
        cam = cv2.VideoCapture(path, cv2.CAP_V4L2)
        cam.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_W)
        cam.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)
        if cam.isOpened() and cam.read()[0]:
            print(f"Camera connected: {path} ({name})")
            return cam
        cam.release()
    return None


def x_of(mask):
    """Center of the white pixels in a mask, as -1 (left) .. +1 (right)."""
    m = cv2.moments(mask, binaryImage=True)
    if m["m00"] == 0:
        return 0.0
    x = (m["m10"] / m["m00"]) / mask.shape[1] * 2 - 1
    return -x if FLIP_X else x


def find_tape(hsv):
    """Blue tape near the bottom of the image."""
    h = hsv.shape[0]
    strip = cv2.inRange(hsv[int(h * TAPE_STRIP):], BLUE_LO, BLUE_HI)
    return float(np.count_nonzero(strip)) / strip.size, x_of(strip)


def find_opponent(frame, hsv):
    """Biggest blob below the horizon that is neither floor nor blue tape."""
    h = frame.shape[0]
    top = int(h * HORIZON)
    ground = frame[top:].astype(np.int16)
    diff = np.linalg.norm(ground - floor_bgr, axis=2)
    mask = ((diff > FLOOR_DIST) * 255).astype(np.uint8)
    mask[cv2.inRange(hsv[top:], BLUE_LO, BLUE_HI) > 0] = 0      # tape is not an opponent
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return False, 0.0, 0.0
    biggest = max(contours, key=cv2.contourArea)
    size = cv2.contourArea(biggest) / mask.size
    if size < MIN_OPP_AREA:
        return False, 0.0, 0.0
    blob = np.zeros_like(mask)
    cv2.drawContours(blob, [biggest], -1, 255, cv2.FILLED)
    return True, x_of(blob), size


def show_on_matrix(seen, x):
    """Opponent position as a dot on the LED matrix (handy when the laptop is far away)."""
    if seen:
        Bridge.call("dot", round((x + 1) / 2 * 12), 3)
    else:
        Bridge.call("dot", -1, -1)


client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id=CLIENT_ID)
client.connect(BROKER, PORT, keepalive=30)
client.loop_start()
print(f"MQTT connected, publishing to {TOPIC}")

cam = open_camera()
if cam is None:
    print("No webcam found yet (plug in the Logitech). Checking every 2 s...")


def loop():
    global cam, floor_bgr, last_seen
    if cam is None:                             # no webcam yet: keep trying every 2 s
        time.sleep(2)
        cam = open_camera()
        return
    ok, frame = cam.read()
    if not ok:
        print("Lost the camera (unplugged?). Waiting for it...")
        cam.release()
        cam = None
        return
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    if floor_bgr is None:                       # still learning the floor color
        h = frame.shape[0]
        floor_samples.append(np.median(frame[int(h * TAPE_STRIP):].reshape(-1, 3), axis=0))
        if len(floor_samples) >= FLOOR_FRAMES:
            floor_bgr = np.median(floor_samples, axis=0)
            print(f"Floor color learned (BGR): {floor_bgr.round()}")
        return

    tape_ahead, tape_x = find_tape(hsv)
    opp_seen, opp_x, opp_size = find_opponent(frame, hsv)

    client.publish(TOPIC, json.dumps({
        "opp_seen": opp_seen, "opp_x": round(opp_x, 2), "opp_size": round(opp_size, 3),
        "tape_ahead": round(tape_ahead, 3), "tape_x": round(tape_x, 2),
    }))

    if opp_seen != last_seen:                   # print only when it changes
        print("OPPONENT SEEN" if opp_seen else "no opponent")
        last_seen = opp_seen
    show_on_matrix(opp_seen, opp_x)
    time.sleep(0.1)                             # ~10 messages per second


App.run(user_loop=loop)
