# Robot Sumo - UNO Q vision + approval (Linux side, runs in Arduino App Lab)
# The UNO Q never drives. It reads the Logitech webcam on the FRONT of the robot, finds
# the opponent and the blue tape, and asks YOU before anything happens. You answer with
# Enter / Delete in the laptop's sumo_brain.py terminal (works on any network, through MQTT),
# or on this app's web page if your laptop can reach the board:
#
#   Web page: http://<board-ip>:7000  (shows the camera, what it detects, and the questions;
#             needs a network where the laptop can reach the board - Tufts WiFi can't)
#     "Opponent?"  Approve (Enter) -> the app tells the laptop to charge while it still
#                                     sees the opponent
#                  Reject (Delete) -> ignored; asks again after IGNORE_TIME if still seen
#     "We're OUT?" asked when the LAPTOP says {"out_detected": true}: its back color sensor
#                  crossed the tape and this camera now sees tape ahead (sensor fusion), so
#                  the robot already stopped on its own.
#                  Approve (Enter) -> the app tells the laptop we lost (sad song)
#                  Reject (Delete) -> false alarm: the laptop keeps wrestling
#     Buttons for "Stop charge", "We're out" (any time) and "Relearn floor" are there too.
#
# MQTT commands to the laptop (topic ME193/Rogers/MK, see sumo_brain.py in ME193):
#   {"charge": true}   about 10 per second while an approved opponent is in view
#   {"charge": false}  approved opponent lost for LOST_TIME, rejected, or stopped
#   {"out": true}      you approved "we're out" (or pressed "We're out")
#   {"out": false}     you rejected "we're out"
# and listens on the same topic for the laptop's messages:
#   {"out_detected": true}             the laptop's sensors say we're out -> ask "We're OUT?"
#   {"answer": "approve_opponent"}     your Enter / Delete in the laptop terminal (same
#                                      actions as the web page's buttons)
# Plus the raw numbers and the open question, about 10 per second:
#   {"opp_seen": true, "opp_x": -0.3, "opp_size": 0.12, "tape_ahead": 0.05, "tape_x": 0.4,
#    "ask": "opponent"}               ask: "opponent", "out" or null (no question open)
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
import base64, json, sys, time, warnings
from pathlib import Path

try:
    sys.stdout.reconfigure(line_buffering=True)    # show each print at once (a container holds its output back, so the console can look empty)
except Exception:
    pass

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

# ---- Blue tape (OpenCV HSV: H 0-179, S 0-255, V 0-255) ----
# These are only the STARTING values. CLICK ON THE TAPE in the web page's picture and the app measures its real colour
# under the real light and uses that instead (it remembers it in tape_calibration.json, next to this file).
BLUE_LO = np.array([95, 80, 40])
BLUE_HI = np.array([130, 255, 255])

# ---- The robot's own body ----
# The robot's own blue apron (and the white track under it) can show along the bottom of the picture. This fraction of
# the image height, counted up from the bottom, is ignored everywhere, so the robot never mistakes its own apron for the
# ring's tape or for an opponent. 0 = ignore nothing. Raise it (0.10, 0.20 ...) until tape_ahead reads 0 in an empty ring.
SELF_MASK_BOTTOM = 0.0

# ---- Opponent ----
FLOOR_FRAMES = 20    # frames used to learn the floor color at startup
# A pixel counts as "not floor" when it differs from the floor colour AT ITS OWN ROW (the floor looks different far away
# than near) by more than FLOOR_DIST, in Lab colour units. Before comparing, the picture's overall drift (the camera's
# exposure and white balance wandering) is measured on the floor itself and taken out, so only real objects are left.
# LUMA_WEIGHT: how much a brightness difference counts compared with a colour difference.
FLOOR_DIST = 25
LUMA_WEIGHT = 0.6
DRIFT_LIMIT = (60, 14, 14)   # the most brightness, a and b drift (Lab units) that is taken out
MIN_OPP_AREA = 0.04  # ignore blobs smaller than this fraction of the ground area
OPP_CONFIRM = 3      # frames in a row the opponent must be seen before the app asks you "Opponent?"

# ---- Camera ----
LOCK_CAMERA = False  # True: once the floor is learned, freeze auto exposure and white balance (best effort, see lock_camera)

# ---- Approval ----
LOST_TIME = 1.0      # approved opponent out of view this long (s) -> stop charging, ask again next time
IGNORE_TIME = 2.0    # after Reject, don't ask the same question again for this long (s)

# ---- Web page ----
VIEW_EVERY = 0.2     # seconds between pictures sent to the web page

floor_samples = []
floor_bgr = None            # median colour of the strip in front (also the "learned" flag and the swatch on the web page)
floor_rows = None           # the floor colour of every row of the ground, in Lab (what opponents are compared with)
last_seen = None
view_sent_at = 0.0
latest_frame = None         # the newest picture (clicking on the tape uses it)
opp_streak = 0              # frames in a row the opponent has been seen
ui_note, ui_note_until = "", 0.0    # a short message for the web page (what clicking on the tape found)
try:
    TAPE_FILE = Path(__file__).resolve().with_name("tape_calibration.json")
except NameError:                                       # no __file__ when pasted into a console
    TAPE_FILE = Path("tape_calibration.json")

# Approval state (changed by your answers)
opp_state = "none"          # "none", "asking", "approved"
opp_ignore_until = 0.0
opp_last_seen = 0.0
out_state = "none"          # "none", "asking" (laptop detected out), "sent"


client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id=CLIENT_ID)
ui = WebUI()


def command(msg):
    client.publish(TOPIC, json.dumps(msg), qos=1)
    print("->", msg)


def on_answer(sid, data):
    """Your answer: a button (or Enter / Delete) on the web page, or Enter / Delete in the
    laptop terminal (arrives over MQTT, see on_mqtt)."""
    global opp_state, opp_ignore_until, out_state, floor_bgr, floor_rows, ui_note, ui_note_until
    action = data.get("action") if isinstance(data, dict) else data
    now = time.monotonic()
    if action in ("approve_out", "out_now"):
        if out_state != "sent":
            out_state = "sent"
            opp_state = "none"
            command({"charge": False})
            command({"out": True})
    elif action == "reject_out":
        if out_state == "asking":
            out_state = "none"
            command({"out": False})
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
        floor_bgr = floor_rows = None
        print("Relearning the floor color...")
    elif action == "learn_tape":                    # you clicked on the tape in the picture (x, y: 0..1 across the picture)
        if latest_frame is None or not isinstance(data, dict):
            ui_note = "No picture yet."
        else:
            ui_note = learn_tape(latest_frame, float(data.get("x", 0.5)), float(data.get("y", 0.5)))
        ui_note_until = now + 8.0
        print(ui_note)


ui.on_message("answer", on_answer)


def on_mqtt(client, userdata, msg):
    """Messages from the laptop: its sensors think we're out (it has already stopped), or
    your Enter / Delete from its terminal."""
    global out_state
    try:
        data = json.loads(msg.payload)
    except ValueError:
        return
    if not isinstance(data, dict):
        return
    if data.get("out_detected") and out_state == "none":
        out_state = "asking"
        print("Laptop: we're out? (waiting for your answer)")
    if isinstance(data.get("answer"), str):
        print("Laptop answered:", data["answer"])
        on_answer(None, {"action": data["answer"]})


def open_question():
    """What the app is asking right now: "out" wins over "opponent"."""
    if out_state == "asking":
        return "out"
    if opp_state == "asking":
        return "opponent"
    return None


client.on_message = on_mqtt
client.on_connect = lambda c, u, f, rc, p=None: c.subscribe(TOPIC)
client.reconnect_delay_set(min_delay=1, max_delay=10)
client.connect_async(BROKER, PORT, keepalive=30)     # keep retrying in the background if the Wi-Fi or broker is not ready yet
client.loop_start()
print(f"MQTT: publishing to {TOPIC} as soon as {BROKER} is reachable")


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


def ground_rows(h, top):
    """(first row, last row + 1) to analyze: from `top` (a fraction of the height) down to just above the robot's own apron."""
    first = int(h * top)
    return first, max(first + 1, int(h * (1.0 - SELF_MASK_BOTTOM)))


def masked_row_median(lab, tape, min_clean=20):
    """The median colour of each row of `lab`, ignoring the pixels flagged in `tape` (blue tape is not floor).
    Returns (medians, enough): `enough` says which rows kept at least min_clean clean pixels."""
    work = np.where(tape[..., None], np.inf, lab)
    work.sort(axis=1)                                   # the tape pixels (inf) sort to the end of each row
    clean = (~tape).sum(axis=1)
    mid = (np.maximum(clean, 1) - 1) // 2
    medians = np.take_along_axis(work, np.broadcast_to(mid[:, None, None], (len(mid), 1, 3)), axis=1)[:, 0, :]
    return medians, clean >= min_clean


def floor_sample(frame):
    """What one picture adds to the floor model: the median colour of the strip in front (BGR) and the median colour of
    every ground row (Lab), leaving out blue tape (a ring edge in view when you learn must not become 'floor')."""
    first, last = ground_rows(frame.shape[0], TAPE_STRIP)
    strip = np.median(frame[first:last].reshape(-1, 3), axis=0)
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB).astype(np.float32)
    top, bottom = ground_rows(frame.shape[0], HORIZON)
    tape = cv2.inRange(cv2.cvtColor(frame[top:bottom], cv2.COLOR_BGR2HSV), BLUE_LO, BLUE_HI) > 0
    medians, enough = masked_row_median(lab[top:bottom], tape)
    if enough.mean() < 0.2:                             # nearly all tape: nothing to learn the floor from, keep the plain medians
        return strip, np.median(lab[top:bottom], axis=1)
    medians[~enough] = np.nan                           # rows that were mostly tape: filled in from their neighbours in finish_floor
    return strip, medians


def finish_floor(samples):
    """The learning pictures -> (floor_bgr, floor_rows)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # a row that was tape in every picture has no median
        rows = np.nanmedian([s[1] for s in samples], axis=0)
    good = ~np.isnan(rows).any(axis=1)
    if good.any() and not good.all():                   # fill those rows from the nearest rows that were floor
        index = np.arange(len(rows))
        for c in range(rows.shape[1]):
            rows[:, c] = np.interp(index, index[good], rows[good, c])
    return np.median([s[0] for s in samples], axis=0), rows


def learn_tape(frame, x, y, radius=6):
    """Learn the tape's colour from the small patch of the picture you clicked on (x, y: 0..1 across the picture) and use it
    from now on. Returns a sentence about what it found."""
    global BLUE_LO, BLUE_HI
    h, w = frame.shape[:2]
    cx, cy = int(min(max(x, 0.0), 1.0) * (w - 1)), int(min(max(y, 0.0), 1.0) * (h - 1))
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    patch = hsv[max(0, cy - radius):cy + radius + 1, max(0, cx - radius):cx + radius + 1].reshape(-1, 3).astype(float)
    if np.median(patch[:, 1]) < 35:
        return "That spot has almost no colour (floor? white track?). Click on the middle of the blue tape."
    hue = np.median(patch[:, 0])
    if not 80 <= hue <= 145:                         # blue is about 95-130; a wooden floor or a box is nowhere near
        return f"That spot isn't blue (its hue is {hue:.0f}; blue tape is about 95 to 130). Click on the middle of the blue tape."
    # A click near the tape's edge also catches floor. Keep only the clearly coloured pixels that agree with the middle one.
    keep = (patch[:, 1] >= 0.6 * np.median(patch[:, 1])) & (np.abs(patch[:, 0] - np.median(patch[:, 0])) <= 15)
    patch = patch[keep]
    if len(patch) < 20:
        return "Couldn't find a clear patch of colour there. Click on the middle of the blue tape."
    # The clicked colour, widened to allow for the light drifting afterwards (exposure, white balance, shadows).
    lo = [max(0, np.percentile(patch[:, 0], 2) - 10), max(25, 0.5 * np.percentile(patch[:, 1], 2)), max(15, 0.55 * np.percentile(patch[:, 2], 2))]
    hi = [min(179, np.percentile(patch[:, 0], 98) + 10), 255, 255]
    BLUE_LO, BLUE_HI = np.array(lo, dtype=int), np.array(hi, dtype=int)
    try:
        TAPE_FILE.write_text(json.dumps({"lo": BLUE_LO.tolist(), "hi": BLUE_HI.tolist()}))
    except OSError:
        pass
    return f"Tape learned: hue {BLUE_LO[0]}-{BLUE_HI[0]}, saturation from {BLUE_LO[1]}, brightness from {BLUE_LO[2]}."      # the numbers actually kept


def load_tape_calibration():
    """Use the tape colour learned last time, if there is one, and say so on the web page (the console is easy to miss)."""
    global BLUE_LO, BLUE_HI, ui_note, ui_note_until
    try:
        saved = json.loads(TAPE_FILE.read_text())
        lo, hi = np.array(saved["lo"], dtype=int), np.array(saved["hi"], dtype=int)
        if lo.shape != (3,) or hi.shape != (3,):
            raise ValueError("expected three numbers for each end of the range")
        BLUE_LO, BLUE_HI = lo, hi
        ui_note = f"Tape colour loaded from {TAPE_FILE.name}: hue {lo[0]}-{hi[0]}, saturation from {lo[1]}, brightness from {lo[2]}."
    except FileNotFoundError:
        ui_note = "No saved tape colour yet: using the default range. Click on the tape to teach it."
    except (OSError, ValueError, KeyError, TypeError):
        ui_note = f"{TAPE_FILE.name} could not be read: using the default range. Click on the tape to teach it again."
    ui_note_until = time.monotonic() + 90.0      # the web page shows it for 90 s after the app starts
    print(ui_note)


def lock_camera(cam):
    """Freeze auto exposure and auto white balance at what the camera chose, so colours stop drifting as the robot moves.
    Best effort: not every camera and driver accepts every setting, so it prints what it ended up with."""
    exposure, balance = cam.get(cv2.CAP_PROP_EXPOSURE), cam.get(cv2.CAP_PROP_WB_TEMPERATURE)
    cam.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)          # 1 = manual on the Linux (V4L2) backend
    cam.set(cv2.CAP_PROP_EXPOSURE, exposure)
    cam.set(cv2.CAP_PROP_AUTO_WB, 0)
    cam.set(cv2.CAP_PROP_WB_TEMPERATURE, balance)
    print(f"Camera locked: exposure {cam.get(cv2.CAP_PROP_EXPOSURE)}, white balance {cam.get(cv2.CAP_PROP_WB_TEMPERATURE)}")


def find_tape(frame, hsv):
    """Blue tape near the bottom of the image (not counting the robot's own apron)."""
    first, last = ground_rows(hsv.shape[0], TAPE_STRIP)
    strip = cv2.inRange(hsv[first:last], BLUE_LO, BLUE_HI)
    return float(np.count_nonzero(strip)) / strip.size, x_of(strip)


def find_opponent(frame, hsv):
    """Biggest blob below the horizon that is neither floor nor blue tape.
    Returns (seen, x, size, mask, biggest contour or None)."""
    top, bottom = ground_rows(frame.shape[0], HORIZON)
    lab = cv2.cvtColor(frame[top:bottom], cv2.COLOR_BGR2LAB).astype(np.float32)
    # How far has this picture's floor drifted from the learned floor (exposure, white balance)? The median over rows of each
    # row's median colour: an opponent narrower than half the picture cannot pull it. Take that drift out first.
    tape_px = cv2.inRange(hsv[top:bottom], BLUE_LO, BLUE_HI) > 0
    medians, enough = masked_row_median(lab, tape_px)             # blue tape is not floor: it must not pull the drift estimate
    drift = np.median((medians - floor_rows)[enough], axis=0) if enough.any() else np.zeros(3, np.float32)
    lab -= np.clip(drift, [-d for d in DRIFT_LIMIT], DRIFT_LIMIT)
    delta = lab - floor_rows[:, None, :]                # each pixel against the floor colour of ITS OWN row
    delta[..., 0] *= LUMA_WEIGHT
    mask = ((np.sqrt((delta ** 2).sum(axis=2)) > FLOOR_DIST) * 255).astype(np.uint8)
    mask[cv2.inRange(hsv[top:bottom], BLUE_LO, BLUE_HI) > 0] = 0      # tape is not an opponent
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


def update_approval(opp_seen, opp_confirmed):
    """Raise the opponent question for the web page, and keep an approved charge going.
    opp_confirmed: the opponent has been seen OPP_CONFIRM frames in a row (one stray frame does not ask)."""
    global opp_state, opp_last_seen
    now = time.monotonic()
    if out_state == "sent":
        return
    # Opponent
    if opp_seen:
        opp_last_seen = now
    if opp_state == "none" and opp_confirmed and now > opp_ignore_until:
        opp_state = "asking"
        print("Opponent? (waiting for your answer)")
    elif opp_state == "asking" and now - opp_last_seen > LOST_TIME:
        opp_state = "none"                      # it went away before you answered
    elif opp_state == "approved":
        if now - opp_last_seen > LOST_TIME:
            opp_state = "none"
            command({"charge": False})
        elif opp_seen:
            client.publish(TOPIC, json.dumps({"charge": True}))   # keep the charge going


def send_view(frame, hsv, opp_mask, biggest, opp_seen, opp_size, tape_ahead):
    """The camera picture for the web page, marked up with what the app sees:
    red = "not floor" pixels (opponent candidates), blue = tape, green outline = the
    opponent (yellow = too small to count), white line = horizon, orange line = top of the
    tape strip, top-left square = the learned floor color."""
    h, w = frame.shape[:2]
    top, bottom = ground_rows(h, HORIZON)
    pic = frame.copy()
    if opp_mask is not None:
        ground = pic[top:bottom]
        ground[opp_mask > 0] = (0.5 * ground[opp_mask > 0] + (0, 0, 127)).astype(np.uint8)
    tape = cv2.inRange(hsv, BLUE_LO, BLUE_HI) > 0
    pic[tape] = (0.5 * pic[tape] + (127, 64, 0)).astype(np.uint8)
    if biggest is not None:
        color = (0, 255, 0) if opp_seen else (0, 255, 255)
        cv2.drawContours(pic, [biggest + (0, top)], -1, color, 2)
    cv2.line(pic, (0, top), (w, top), (255, 255, 255), 1)
    cv2.line(pic, (0, int(h * TAPE_STRIP)), (w, int(h * TAPE_STRIP)), (0, 128, 255), 1)
    if SELF_MASK_BOTTOM > 0:
        cv2.line(pic, (0, bottom), (w, bottom), (255, 0, 255), 1)       # magenta: everything below this line is ignored
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
        "note": ui_note if time.monotonic() < ui_note_until else "",
    })


def show_on_matrix(seen, x):
    """Opponent position as a dot on the LED matrix (handy when the laptop is far away)."""
    if seen:
        Bridge.call("dot", round((x + 1) / 2 * 12), 3)
    else:
        Bridge.call("dot", -1, -1)


load_tape_calibration()
cam = open_camera()
if cam is None:
    print("No webcam found yet (plug in the Logitech). Checking every 2 s...")


def loop():
    global cam, floor_bgr, floor_rows, last_seen, view_sent_at, latest_frame, opp_streak
    started = time.monotonic()
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
    latest_frame = frame

    if floor_bgr is None:                       # still learning the floor color
        floor_samples.append(floor_sample(frame))
        if len(floor_samples) >= FLOOR_FRAMES:
            floor_bgr, floor_rows = finish_floor(floor_samples)
            print(f"Floor color learned (BGR): {floor_bgr.round()}")
            if LOCK_CAMERA:
                lock_camera(cam)
        if now - view_sent_at > VIEW_EVERY:
            send_view(frame, hsv, None, None, False, 0.0, 0.0)
            view_sent_at = now
        return

    tape_ahead, tape_x = find_tape(frame, hsv)
    opp_seen, opp_x, opp_size, opp_mask, biggest = find_opponent(frame, hsv)
    opp_streak = opp_streak + 1 if opp_seen else 0

    client.publish(TOPIC, json.dumps({
        "opp_seen": opp_seen, "opp_x": round(opp_x, 2), "opp_size": round(opp_size, 3),
        "tape_ahead": round(tape_ahead, 3), "tape_x": round(tape_x, 2),
        "ask": open_question(),
    }))
    update_approval(opp_seen, opp_streak >= OPP_CONFIRM)

    if opp_seen != last_seen:                   # print only when it changes
        print("OPPONENT SEEN" if opp_seen else "no opponent")
        last_seen = opp_seen
    show_on_matrix(opp_seen, opp_x)
    if now - view_sent_at > VIEW_EVERY:
        send_view(frame, hsv, opp_mask, biggest, opp_seen, opp_size, tape_ahead)
        view_sent_at = now
    time.sleep(max(0.005, 0.1 - (time.monotonic() - started)))   # ~10 messages per second, however long the work took


App.run(user_loop=loop)