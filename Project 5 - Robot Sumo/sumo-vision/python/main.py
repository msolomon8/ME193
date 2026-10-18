# Robot Sumo - UNO Q vision + approval (Linux side, runs in Arduino App Lab)
# The UNO Q never drives. It reads the Logitech webcam on the FRONT of the robot, finds
# the opponent and the blue tape, and asks YOU (on its web page) before anything happens:
#
#   Web page: http://<board-ip>:7000  (shows the camera, what it detects, and the questions)
#     "Opponent?"  Approve (Enter) -> the app tells the laptop to charge while it still
#                                     sees the opponent
#                  Reject (Delete) -> ignored; asks again after IGNORE_TIME if still seen
#     "We're OUT?" Approve (Enter) -> the app tells the laptop we lost (stop + sad song)
#                  Reject (Delete) -> ignored for IGNORE_TIME
#     Buttons for "Stop charge", "We're out" (any time) and "Relearn floor" are there too.
#
# MQTT commands to the laptop (topic ME193/Rogers/MK, see sumo_brain.py in ME193):
#   {"charge": true}   about 10 per second while an approved opponent is in view
#   {"charge": false}  approved opponent lost for LOST_TIME, rejected, or stopped
#   {"out": true}      you approved "we're out"
# Plus the raw numbers, about 10 per second (the laptop ignores these):
#   {"opp_seen": true, "opp_x": -0.3, "opp_size": 0.12, "tape_ahead": 0.05, "tape_x": 0.4}
#   opp_x / tape_x: -1 = far left of the image, 0 = center, +1 = far right
#   opp_size:       fraction of the ground area the opponent covers (bigger = closer)
#   tape_ahead:     fraction of the bottom of the image that is blue tape
#
# Floor calibration: for the first FLOOR_FRAMES frames, the robot should be sitting in the
# ring looking at empty floor. The app learns the floor's color from them ("Relearn floor"
# on the web page does it again).
from arduino.app_utils import *
from arduino.app_bricks.web_ui import WebUI
import paho.mqtt.client as mqtt
import cv2
import numpy as np
import base64, json, time
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

# ---- Approval ----
LOST_TIME = 1.0      # approved opponent out of view this long (s) -> stop charging, ask again next time
IGNORE_TIME = 2.0    # after Reject, don't ask the same question again for this long (s)
OUT_TAPE = 0.5       # tape covers this much of the bottom strip -> ask "we're out?"
OUT_CONFIRM = 0.3    # ...for this long (s) in a row

# ---- Web page ----
VIEW_EVERY = 0.2     # seconds between pictures sent to the web page

floor_samples = []
floor_bgr = None
last_seen = None
view_sent_at = 0.0

# Approval state (changed by the web page's buttons)
opp_state = "none"          # "none", "asking", "approved"
opp_ignore_until = 0.0
opp_last_seen = 0.0
out_state = "none"          # "none", "asking", "sent"
out_ignore_until = 0.0
out_since = None


client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id=CLIENT_ID)
client.connect(BROKER, PORT, keepalive=30)
client.loop_start()
print(f"MQTT connected, publishing to {TOPIC}")

ui = WebUI()


def command(msg):
    client.publish(TOPIC, json.dumps(msg), qos=1)
    print("->", msg)


def on_answer(sid, data):
    """A button (or Enter / Delete) on the web page."""
    global opp_state, opp_ignore_until, out_state, out_ignore_until, floor_bgr
    action = data.get("action") if isinstance(data, dict) else data
    now = time.monotonic()
    if action in ("approve_out", "out_now"):
        if out_state != "sent":
            out_state = "sent"
            opp_state = "none"
            command({"charge": False})
            command({"out": True})
    elif action == "reject_out":
        out_state, out_ignore_until = "none", now + IGNORE_TIME
    elif action == "approve_opponent":
        if opp_state == "asking":
            opp_state = "approved"
            command({"charge": True})
    elif action in ("reject_opponent", "stop_charge"):
        if opp_state == "approved":
            command({"charge": False})
        opp_state, opp_ignore_until = "none", now + IGNORE_TIME
    elif action == "relearn_floor":
        floor_samples.clear()
        floor_bgr = None
        print("Relearning the floor color...")


ui.on_message("answer", on_answer)


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
    """Biggest blob below the horizon that is neither floor nor blue tape.
    Returns (seen, x, size, mask, biggest contour or None)."""
    h = frame.shape[0]
    top = int(h * HORIZON)
    ground = frame[top:].astype(np.int16)
    diff = np.linalg.norm(ground - floor_bgr, axis=2)
    mask = ((diff > FLOOR_DIST) * 255).astype(np.uint8)
    mask[cv2.inRange(hsv[top:], BLUE_LO, BLUE_HI) > 0] = 0      # tape is not an opponent
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return False, 0.0, 0.0, mask, None
    biggest = max(contours, key=cv2.contourArea)
    size = cv2.contourArea(biggest) / mask.size
    if size < MIN_OPP_AREA:
        return False, 0.0, 0.0, mask, biggest
    blob = np.zeros_like(mask)
    cv2.drawContours(blob, [biggest], -1, 255, cv2.FILLED)
    return True, x_of(blob), size, mask, biggest


def update_approval(opp_seen, tape_ahead):
    """Raise questions for the web page, and keep an approved charge going."""
    global opp_state, opp_last_seen, out_state, out_since
    now = time.monotonic()
    if out_state == "sent":
        return
    # Opponent
    if opp_seen:
        opp_last_seen = now
    if opp_state == "none" and opp_seen and now > opp_ignore_until:
        opp_state = "asking"
        print("Opponent? (waiting for approve / reject on the web page)")
    elif opp_state == "asking" and now - opp_last_seen > LOST_TIME:
        opp_state = "none"                      # it went away before you answered
    elif opp_state == "approved":
        if now - opp_last_seen > LOST_TIME:
            opp_state = "none"
            command({"charge": False})
        elif opp_seen:
            client.publish(TOPIC, json.dumps({"charge": True}))   # keep the charge going
    # Out
    if tape_ahead > OUT_TAPE:
        out_since = out_since or now
    else:
        out_since = None
    if (out_state == "none" and out_since is not None and now - out_since > OUT_CONFIRM
            and now > out_ignore_until):
        out_state = "asking"
        print("We're out? (waiting for approve / reject on the web page)")
    elif out_state == "asking" and out_since is None:
        out_state = "none"                      # tape went away before you answered


def send_view(frame, hsv, opp_mask, biggest, opp_seen, opp_size, tape_ahead):
    """The camera picture for the web page, marked up with what the app sees:
    red = "not floor" pixels (opponent candidates), blue = tape, green outline = the
    opponent (yellow = too small to count), white line = horizon, orange line = top of the
    tape strip, top-left square = the learned floor color."""
    h, w = frame.shape[:2]
    top = int(h * HORIZON)
    pic = frame.copy()
    if opp_mask is not None:
        ground = pic[top:]
        ground[opp_mask > 0] = (0.5 * ground[opp_mask > 0] + (0, 0, 127)).astype(np.uint8)
    tape = cv2.inRange(hsv, BLUE_LO, BLUE_HI) > 0
    pic[tape] = (0.5 * pic[tape] + (127, 64, 0)).astype(np.uint8)
    if biggest is not None:
        color = (0, 255, 0) if opp_seen else (0, 255, 255)
        cv2.drawContours(pic, [biggest + (0, top)], -1, color, 2)
    cv2.line(pic, (0, top), (w, top), (255, 255, 255), 1)
    cv2.line(pic, (0, int(h * TAPE_STRIP)), (w, int(h * TAPE_STRIP)), (0, 128, 255), 1)
    if floor_bgr is not None:
        cv2.rectangle(pic, (0, 0), (20, 20), tuple(int(c) for c in floor_bgr), cv2.FILLED)
        cv2.rectangle(pic, (0, 0), (20, 20), (255, 255, 255), 1)
    ok, jpg = cv2.imencode(".jpg", pic, [cv2.IMWRITE_JPEG_QUALITY, 70])
    if not ok:
        return
    ui.send_message("view", {
        "jpg": base64.b64encode(jpg.tobytes()).decode("ascii"),
        "learning": floor_bgr is None,
        "opp_seen": opp_seen, "opp_size": round(opp_size, 2), "tape_ahead": round(tape_ahead, 2),
        "opp_state": opp_state, "out_state": out_state,
    })


def show_on_matrix(seen, x):
    """Opponent position as a dot on the LED matrix (handy when the laptop is far away)."""
    if seen:
        Bridge.call("dot", round((x + 1) / 2 * 12), 3)
    else:
        Bridge.call("dot", -1, -1)


cam = open_camera()
if cam is None:
    print("No webcam found yet (plug in the Logitech). Checking every 2 s...")


def loop():
    global cam, floor_bgr, last_seen, view_sent_at
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
    now = time.monotonic()

    if floor_bgr is None:                       # still learning the floor color
        h = frame.shape[0]
        floor_samples.append(np.median(frame[int(h * TAPE_STRIP):].reshape(-1, 3), axis=0))
        if len(floor_samples) >= FLOOR_FRAMES:
            floor_bgr = np.median(floor_samples, axis=0)
            print(f"Floor color learned (BGR): {floor_bgr.round()}")
        if now - view_sent_at > VIEW_EVERY:
            send_view(frame, hsv, None, None, False, 0.0, 0.0)
            view_sent_at = now
        return

    tape_ahead, tape_x = find_tape(hsv)
    opp_seen, opp_x, opp_size, opp_mask, biggest = find_opponent(frame, hsv)

    client.publish(TOPIC, json.dumps({
        "opp_seen": opp_seen, "opp_x": round(opp_x, 2), "opp_size": round(opp_size, 3),
        "tape_ahead": round(tape_ahead, 3), "tape_x": round(tape_x, 2),
    }))
    update_approval(opp_seen, tape_ahead)

    if opp_seen != last_seen:                   # print only when it changes
        print("OPPONENT SEEN" if opp_seen else "no opponent")
        last_seen = opp_seen
    show_on_matrix(opp_seen, opp_x)
    if now - view_sent_at > VIEW_EVERY:
        send_view(frame, hsv, opp_mask, biggest, opp_seen, opp_size, tape_ahead)
        view_sent_at = now
    time.sleep(0.1)                             # ~10 messages per second


App.run(user_loop=loop)
