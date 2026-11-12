"""
The Impenetrable Safe - running entirely on the Arduino UNO Q.

Linux side (this file):  webcam -> crop inside red frame -> CNN (numpy) -> lock policy -> webpage
MCU side (sketch.ino):   button (D7), 3 LEDs (D2 D3 D4), servo (D9)
They talk through the Bridge:
    sketch -> Python:  Bridge.notify("button_pressed")
    Python -> sketch:  Bridge.call("set_state", mode, digits_entered, servo_open)
The webpage (assets/index.html) is served by the WebUI brick on port 7000 and asks
/api/state a few times per second.

Flow: press D7 -> LEDs flash, page says "Ready" -> show digit 1 -> LED D2 + page -> blank ->
digit 2 -> LED D3 -> blank -> digit 3 -> LED D4 -> all LEDs flash, page says UNLOCKED,
servo turns left to open.

Extras:
  * Change the code: while UNLOCKED, hold D7 for 3 s, then show 3 new digits (blank between).
    The new code is saved to data/secret_code.json, so it survives a restart.
  * Intruder photos: every wrong code takes a picture from the camera. The webpage shows them.
"""
import json
import re
import threading
import time
from collections import deque
from pathlib import Path

import cv2

from arduino.app_bricks.web_ui import WebUI
from arduino.app_utils import App, Bridge
from lock_policy import LockPolicy
from safe_net import NumpyCNN

# ---------------------------------------------------------------------------
# Settings (same meaning as in the laptop control panel)
# ---------------------------------------------------------------------------
SETTINGS = {
    "secret_code": [2, 4, 5],   # starting code, digits 0-5 (0 = fist); a code set with the button replaces it
    "min_confidence": 0.80,     # network must be at least this sure
    "stable_seconds": 1.0,      # hold a digit this long
    "clear_seconds": 0.5,       # blank box this long between digits
    "digit_timeout": 30,        # give up after this many seconds without progress
    "unlock_seconds": 20,       # servo stays open this long (or until the button is pressed)
    "denied_seconds": 3,
    "max_failures": 3,
    "lockout_seconds": 30,
    "saved_seconds": 3,         # LEDs show "saved" this long after a new code is set
}
FPS = 8                         # frames per second to process
CLASS_NAMES = ["0", "1", "2", "3", "4", "5", "blank"]
# mode numbers sent to the sketch (it turns them into LED patterns)
MODES = {"IDLE": 0, "READING": 1, "WAIT_CLEAR": 1, "UNLOCKED": 2, "DENIED": 3, "LOCKOUT": 4,
         "NEW_CODE": 5, "NEW_CODE_CLEAR": 5, "CODE_SAVED": 6}

HERE = Path(__file__).parent
DATA_DIR = HERE.parent / "data"              # = ~/ArduinoApps/impenetrable-safe/data on the board
CODE_FILE = DATA_DIR / "secret_code.json"
INTRUDER_DIR = DATA_DIR / "intruders"
MAX_INTRUDER_PHOTOS = 30

try:   # a code chosen with the button (saved last time) replaces the starting code
    saved_code = json.loads(CODE_FILE.read_text())
    if isinstance(saved_code, list) and len(saved_code) == 3 and all(d in range(6) for d in saved_code):
        SETTINGS["secret_code"] = saved_code
        print("[safe] using the code saved with the button", flush=True)
except FileNotFoundError:
    pass
except Exception as e:
    print(f"[safe] couldn't read {CODE_FILE.name} ({e}); using the starting code", flush=True)
net = NumpyCNN(HERE / "finger_cnn.npz")
print(f"[safe] network loaded ({net.val_acc:.0%} validation accuracy), crop = {net.roi}", flush=True)


# ---------------------------------------------------------------------------
# Button: the sketch notifies us; we only count presses here (handlers must be quick)
#   short press -> "button_pressed"      held 3 s -> "button_held"
# ---------------------------------------------------------------------------
presses = {"n": 0, "held": 0}
presses_lock = threading.Lock()


def button_pressed():
    with presses_lock:
        presses["n"] += 1


def button_held():
    with presses_lock:
        presses["held"] += 1


Bridge.provide("button_pressed", button_pressed)
Bridge.provide("button_held", button_held)


# ---------------------------------------------------------------------------
# Camera: a background thread keeps only the newest frame
# ---------------------------------------------------------------------------
class FrameGrabber:
    """Opens the webcam in the BACKGROUND and keeps retrying, so the webpage and button keep
    working even if the camera is unplugged or drops out (e.g. a power dip when the servo moves)."""

    def __init__(self):
        self.frame = None
        self.frame_t = 0.0
        self.lock = threading.Lock()
        threading.Thread(target=self._run, daemon=True).start()

    def _open(self):
        try:   # Arduino's own camera class (finds the USB webcam for us)
            from arduino.app_peripherals.camera import Camera
            # 1920 x 1080 = the camera's full view, same as the training photos.
            # MJPG = compressed, so it's smooth over USB.
            try:
                cam = Camera(resolution=(1920, 1080), fps=15, codec="MJPG", auto_reconnect=False)
            except TypeError:
                cam = Camera(resolution=(1920, 1080), fps=15)
            cam.start()
            print("[safe] camera started (Arduino Camera)", flush=True)
            return cam.capture, cam.stop
        except Exception as e:
            print(f"[safe] Arduino Camera didn't start ({e}); trying OpenCV directly", flush=True)
        for idx in range(6):
            cap = cv2.VideoCapture(idx, cv2.CAP_V4L2)
            if cap.isOpened():
                cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc(*"MJPG"))
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
                ok, _ = cap.read()
                if ok:
                    print(f"[safe] camera /dev/video{idx} (OpenCV)", flush=True)
                    return (lambda c=cap: c.read()[1]), cap.release
            cap.release()
        return None, None

    def _run(self):
        while True:
            read_fn, close_fn = self._open()
            if read_fn is None:
                print("[safe] NO CAMERA FOUND - is the webcam plugged into the USB-C hub? Retrying...", flush=True)
                time.sleep(3)
                continue
            failures, warned = 0, False
            while failures < 30:                    # ~3 s of failed reads = camera is gone, reopen it
                try:
                    f = read_fn()
                except Exception:
                    f = None
                if f is None:
                    failures += 1
                    time.sleep(0.1)
                    continue
                failures = 0
                if not warned:
                    warned = True
                    h, w = f.shape[:2]
                    print(f"[safe] camera picture: {w} x {h}", flush=True)
                with self.lock:
                    self.frame, self.frame_t = f, time.monotonic()
            print("[safe] camera stopped sending pictures - reconnecting", flush=True)
            try:
                close_fn()
            except Exception:
                pass
            time.sleep(1)

    def read(self):
        with self.lock:   # an old picture (camera dropped out) counts as no picture
            return self.frame if time.monotonic() - self.frame_t < 2.0 else None


camera = FrameGrabber()


# ---------------------------------------------------------------------------
# Lock policy
# ---------------------------------------------------------------------------
events = deque(maxlen=30)
event_id = [0]


intruder = {"pending": None}    # set on a wrong code; the main loop takes the photo


def on_event(name, **info):
    event_id[0] += 1
    events.append({"id": event_id[0], "name": name, "info": info, "t": time.time()})
    shown = {k: v for k, v in info.items() if k not in ("digit", "entered")}   # never print codes
    print(f"[safe] event: {name} {shown if shown else ''}", flush=True)
    if name in ("denied", "lockout"):
        intruder["pending"] = "INTRUDER HAND! " + ("LOCKED OUT" if name == "lockout" else f"WRONG CODE ({info.get('failures', '?')} of {SETTINGS['max_failures']})")
    if name == "code_changed":
        try:
            DATA_DIR.mkdir(parents=True, exist_ok=True)
            CODE_FILE.write_text(json.dumps(SETTINGS["secret_code"]))
        except Exception as e:
            print(f"[safe] couldn't save the new code ({e}); it works until the app restarts", flush=True)


def save_intruder_photo(frame, label):
    """Full camera picture with the time and reason stamped on it; keeps the newest 30."""
    INTRUDER_DIR.mkdir(parents=True, exist_ok=True)
    img = cv2.resize(frame, (960, 540), interpolation=cv2.INTER_AREA)
    stamp = time.strftime("%a %b %d  %I:%M:%S %p")
    cv2.rectangle(img, (0, 0), (960, 44), (0, 0, 0), -1)
    cv2.putText(img, f"{label}   {stamp}", (14, 31), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (80, 80, 255), 2, cv2.LINE_AA)
    name = time.strftime("%Y%m%d-%H%M%S") + ".jpg"
    cv2.imwrite(str(INTRUDER_DIR / name), img, [cv2.IMWRITE_JPEG_QUALITY, 85])
    for old in sorted(INTRUDER_DIR.glob("*.jpg"))[:-MAX_INTRUDER_PHOTOS]:
        old.unlink()
    print(f"[safe] intruder photo saved: {name}", flush=True)


policy = LockPolicy(SETTINGS, on_event=on_event)
recent = deque(maxlen=3)        # average the last few frames (one odd frame can't flip it)
shared = {"probs": None, "status": policy.status(time.monotonic()), "jpeg": None, "ms": 0}
shared_lock = threading.Lock()
sent = {"key": None, "t": 0.0}


def send_to_sketch(status):
    key = (MODES[status["state"]], status["progress_leds"], status["servo_open"])
    now = time.monotonic()
    if key == sent["key"] and now - sent["t"] < 2.0:     # re-send every 2 s in case the sketch restarted
        return
    try:
        Bridge.call("set_state", key[0], key[1], key[2], timeout=2)
        sent["key"], sent["t"] = key, now
    except Exception as e:
        print(f"[safe] couldn't reach the sketch: {e}", flush=True)


# ---------------------------------------------------------------------------
# Webpage (WebUI brick serves ../assets/index.html; these are the data routes)
# ---------------------------------------------------------------------------
ui = WebUI()


def api_state():
    with shared_lock:
        st = dict(shared["status"])
        probs = shared["probs"]
        ms = shared["ms"]
        evs = list(events)
    top = None if probs is None else max(range(7), key=lambda i: probs[i])
    st["digits"] = len(st["digits"])          # don't send the actual digits (keeps the code secret)
    st.pop("candidate", None)
    return {
        "lock": st,
        "sees": None if top is None else CLASS_NAMES[top],
        "sure": None if top is None else round(probs[top], 3),
        "need": SETTINGS["min_confidence"],
        "camera_ok": camera.read() is not None,
        "capture": {"busy": capture["busy"], "message": capture["message"]},
        "counts": photo_counts(),
        "ms": ms,
        "events": [{**e, "info": {k: v for k, v in e["info"].items() if k not in ("digit", "entered")}}
                   for e in evs],
    }


def api_snapshot():
    from fastapi import Response
    with shared_lock:
        jpeg = shared["jpeg"]
    return Response(content=jpeg or b"", media_type="image/jpeg", headers={"Cache-Control": "no-store"})


def api_button():                       # the page's "Start" button works like D7
    button_pressed()
    return {"ok": True}


def api_button_hold():                  # the page's "Change code" button = holding D7 for 3 s
    button_held()
    return {"ok": True}


# ---- intruder photos ----
PHOTO_NAME = re.compile(r"^\d{8}-\d{6}\.jpg$")


def api_intruders():
    if not INTRUDER_DIR.exists():
        return {"photos": []}
    return {"photos": [p.name for p in sorted(INTRUDER_DIR.glob("*.jpg"), reverse=True)]}


def api_intruder(name: str):
    from fastapi import Response
    p = INTRUDER_DIR / name
    if not PHOTO_NAME.match(name) or not p.exists():
        return Response(status_code=404)
    return Response(content=p.read_bytes(), media_type="image/jpeg")


def api_intruders_clear():
    n = 0
    for p in INTRUDER_DIR.glob("*.jpg") if INTRUDER_DIR.exists() else []:
        p.unlink()
        n += 1
    return {"ok": True, "deleted": n}


# ---------------------------------------------------------------------------
# Collecting training photos THROUGH THE BOARD'S CAMERA (so training matches what it sees)
# Saved to /app/data/<class>/ = ~/ArduinoApps/impenetrable-safe/data/ on the board,
# same size and naming as the laptop control panel (224 x 128, <burst>_<frame>.png).
# ---------------------------------------------------------------------------
capture = {"busy": False, "last": [], "message": ""}


def _burst(label, n):
    try:
        folder = DATA_DIR / CLASS_NAMES[label]
        folder.mkdir(parents=True, exist_ok=True)
        burst = "board-" + time.strftime("%Y%m%d-%H%M%S") + f"-{int(time.time() * 1000) % 1000:03d}"
        saved = []
        for i in range(n):
            f = camera.read()
            if f is not None:
                p = folder / f"{burst}_{i:03d}.png"
                cv2.imwrite(str(p), cv2.resize(f, (224, 128), interpolation=cv2.INTER_AREA))
                saved.append(p)
            time.sleep(0.15)
        capture["last"] = saved
        name = "blank" if label == 6 else f"{label} finger{'s' if label != 1 else ''}"
        capture["message"] = f"Saved {len(saved)} photos of {name}"
    finally:
        capture["busy"] = False


def api_capture(label: int, count: int = 20):
    if not 0 <= label <= 6:
        return {"error": "bad label"}
    if capture["busy"]:
        return {"error": "already taking photos"}
    if camera.read() is None:
        return {"error": "no camera"}
    capture["busy"], capture["message"] = True, ""
    threading.Thread(target=_burst, args=(label, max(1, min(60, count))), daemon=True).start()
    return {"ok": True}


def api_undo():
    n = sum(1 for p in capture["last"] if p.exists() and (p.unlink() or True))
    capture["last"], capture["message"] = [], f"Deleted the last {n} photos"
    return {"ok": True}


def photo_counts():
    return {name: len(list((DATA_DIR / name).glob("*.png"))) if (DATA_DIR / name).exists() else 0
            for name in CLASS_NAMES}


ui.expose_api("POST", "/api/capture", api_capture)
ui.expose_api("POST", "/api/undo", api_undo)
ui.expose_api("GET", "/api/state", api_state)
ui.expose_api("GET", "/api/snapshot", api_snapshot)
ui.expose_api("POST", "/api/button", api_button)
ui.expose_api("POST", "/api/button_hold", api_button_hold)
ui.expose_api("GET", "/api/intruders", api_intruders)
ui.expose_api("GET", "/api/intruder", api_intruder)
ui.expose_api("POST", "/api/intruders/clear", api_intruders_clear)


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
loop_state = {"last_print": 0.0, "last_jpeg": 0.0}


def loop():
    t0 = time.monotonic()

    with presses_lock:
        n, presses["n"] = presses["n"], 0
        held, presses["held"] = presses["held"], 0
    for _ in range(n):
        policy.press_button(t0)
    for _ in range(held):
        policy.hold_button(t0)

    probs = None
    frame = camera.read()
    if frame is not None:
        recent.append(net.probabilities(net.prepare(frame)))
        probs = [sum(p[i] for p in recent) / len(recent) for i in range(7)]
    else:
        recent.clear()

    now = time.monotonic()
    policy.update(probs, now)
    status = policy.status(now)
    send_to_sketch(status)

    if intruder["pending"] and frame is not None:
        try:
            save_intruder_photo(frame, intruder["pending"])
        except Exception as e:
            print(f"[safe] couldn't save the intruder photo: {e}", flush=True)
        intruder["pending"] = None

    jpeg = None
    if frame is not None and now - loop_state["last_jpeg"] > 0.4:     # small picture for the webpage
        loop_state["last_jpeg"] = now
        view = cv2.resize(frame, (480, 270))
        if net.roi:
            x0, y0, x1, y1 = net.roi
            cv2.rectangle(view, (int(x0 * 480), int(y0 * 270)), (int(x1 * 480), int(y1 * 270)), (80, 255, 80), 2)
        jpeg = cv2.imencode(".jpg", view, [cv2.IMWRITE_JPEG_QUALITY, 70])[1].tobytes()

    with shared_lock:
        shared["probs"], shared["status"], shared["ms"] = probs, status, int((now - t0) * 1000)
        if jpeg is not None:
            shared["jpeg"] = jpeg

    if now - loop_state["last_print"] > 2.0:     # one status line every 2 s in the App Lab console
        loop_state["last_print"] = now
        if probs:
            top = max(range(7), key=lambda i: probs[i])
            sees = f"sees {CLASS_NAMES[top]} ({probs[top]:.0%})"
        else:
            sees = "no camera picture"
        print(f"[safe] {status['state']:<10} digits entered: {len(status['digits'])}   {sees}   "
              f"({(now - t0) * 1000:.0f} ms per frame)", flush=True)

    time.sleep(max(0.0, 1 / FPS - (time.monotonic() - t0)))


App.run(user_loop=loop)
