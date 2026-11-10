"""
The Impenetrable Safe - control panel.

Start it:
    cd finger_lock
    ~/.venvs/impenetrable-safe/bin/python app.py
(the Python environment lives outside Documents because iCloud offloads its files)
then open http://localhost:5050 in your browser.

Tabs:  1 Collect (take labeled photos)  2 Train (train the CNN)  3 Test (live finger count)
       4 Lock (the full safe, with sounds)
"""
import threading
import time
from collections import deque

import cv2
from flask import Flask, Response, jsonify, render_template, request, send_from_directory

import config
from camera import Camera, find_red_frame, for_saving, network_view
from config import BLANK, CLASSES, DATA_DIR, MODEL_DIR, MODEL_PATH
from dataset import class_dir, counts
from hardware import Hardware
from lock_policy import LockPolicy

app = Flask(__name__)
cfg = config.load()
camera = Camera(cfg["camera_index"])
hardware = Hardware()
lock = threading.Lock()          # protects everything below that threads share

shared = {"jpeg": None, "net_jpeg": None, "probs": None}
events = deque(maxlen=200)
event_counter = [0]
capture = {"busy": False, "last_burst": [], "message": ""}
training = {"running": False, "stop": False, "epoch": 0, "epochs": 0, "history": None,
            "best_acc": None, "confusion": None, "log": deque(maxlen=200), "error": None}
predictor = [None]


def describe(label):
    if label == BLANK:
        return "blank (no hand)"
    return f"{label} finger{'s' if label != 1 else ''}"


def record_event(name, **info):
    event_counter[0] += 1
    events.append({"id": event_counter[0], "name": name, "info": info, "t": time.time()})
    hardware.sound(name, **info)


policy = LockPolicy(cfg, on_event=record_event)


def load_predictor():
    if not MODEL_PATH.exists():
        predictor[0] = None
        return
    from predict import Predictor
    try:
        predictor[0] = Predictor(MODEL_PATH)
    except Exception as e:  # e.g. model from an older version of model.py
        predictor[0] = None
        training["error"] = f"Could not load model: {e}"


# ---------------------------------------------------------------------------
# Main loop: read camera -> network -> lock policy -> draw the picture
# ---------------------------------------------------------------------------
STATE_COLORS = {"IDLE": (200, 200, 200), "READING": (0, 220, 255), "WAIT_CLEAR": (255, 180, 0),
                "UNLOCKED": (0, 200, 0), "DENIED": (0, 0, 255), "LOCKOUT": (0, 0, 180)}


recent_probs = deque(maxlen=5)   # last ~1/3 second of predictions


def smooth(probs):
    """Average the last few frames' probabilities so one odd frame can't flip the answer."""
    if probs is None:
        recent_probs.clear()
        return None
    recent_probs.append(probs)
    n = len(recent_probs)
    return [sum(p[i] for p in recent_probs) / n for i in range(len(probs))]


def processing_loop():
    while True:
        t0 = time.monotonic()
        frame = camera.read()
        probs = None
        if frame is not None:
            if predictor[0] is not None:
                try:
                    probs = smooth(predictor[0].probabilities(frame))
                except Exception:
                    probs = None
            # what the network actually sees: the whole frame at 112 x 64, shown enlarged
            big = cv2.resize(network_view(frame, cfg.get("roi")), (336, 192), interpolation=cv2.INTER_NEAREST)
            net_jpeg = cv2.imencode(".jpg", big)[1].tobytes()

        now = time.monotonic()
        with lock:
            policy.update(probs, now)
            status = policy.status(now)
            hardware.apply(status)
            shared["probs"] = probs

        if frame is not None:
            view = frame.copy()
            scale = 960 / view.shape[1]
            view = cv2.resize(view, None, fx=scale, fy=scale)
            h, w = view.shape[:2]
            color = STATE_COLORS.get(status["state"], (255, 255, 255))
            cv2.rectangle(view, (2, 2), (w - 3, h - 3), color, 5)   # border = lock state
            roi = cfg.get("roi")
            if roi:   # dashed-looking green box = the part the network reads
                cv2.rectangle(view, (int(roi[0] * w), int(roi[1] * h)), (int(roi[2] * w), int(roi[3] * h)),
                              (80, 255, 80), 2)
            if probs is not None:
                p = max(range(len(probs)), key=lambda i: probs[i])
                for thick, col in ((7, (0, 0, 0)), (2, color)):
                    cv2.putText(view, f"{describe(p)}  {probs[p]:.0%}", (16, 42),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.0, col, thick)
            jpeg = cv2.imencode(".jpg", view, [cv2.IMWRITE_JPEG_QUALITY, 80])[1].tobytes()
            with lock:
                shared["jpeg"], shared["net_jpeg"] = jpeg, net_jpeg

        time.sleep(max(0.0, 1 / 15 - (time.monotonic() - t0)))   # ~15 frames per second


# ---------------------------------------------------------------------------
# Pages and video
# ---------------------------------------------------------------------------
@app.route("/")
def index():
    return render_template("index.html")


def mjpeg(key):
    while True:
        with lock:
            jpeg = shared[key]
        if jpeg:
            yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n"
        time.sleep(1 / 15)


@app.route("/video")
def video():
    return Response(mjpeg("jpeg"), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/net_video")
def net_video():
    return Response(mjpeg("net_jpeg"), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/photos/<int:label>/<name>")
def photo(label, name):
    return send_from_directory(class_dir(label), name)


@app.route("/models/<name>")
def model_file(name):
    return send_from_directory(MODEL_DIR, name, max_age=0)


# ---------------------------------------------------------------------------
# API
# ---------------------------------------------------------------------------
@app.route("/api/status")
def status():
    since = int(request.args.get("since", 0))
    now = time.monotonic()
    with lock:
        lock_status = policy.status(now)
        probs = shared["probs"]
        new_events = [e for e in events if e["id"] > since]
    meta = predictor[0].meta if predictor[0] is not None else None
    return jsonify({
        "camera_ok": camera.ok, "camera_index": camera.index, "roi": cfg.get("roi"),
        "model_roi": (meta or {}).get("roi") if meta else None,
        "counts": counts(), "capture": {"busy": capture["busy"], "message": capture["message"],
                                         "can_undo": bool(capture["last_burst"])},
        "model": meta, "probs": probs, "lock": lock_status, "events": new_events,
        "settings": {k: cfg[k] for k in ("secret_code", "min_confidence", "stable_seconds",
                                         "clear_seconds", "digit_timeout", "unlock_seconds",
                                         "max_failures", "lockout_seconds")},
        "training": {k: (list(v) if isinstance(v, deque) else v) for k, v in training.items()},
    })


@app.route("/api/capture", methods=["POST"])
def api_capture():
    body = request.get_json(force=True)
    label, n = int(body["label"]), max(1, min(100, int(body.get("count", 1))))
    if label not in CLASSES:
        return jsonify(error="bad label"), 400
    if capture["busy"]:
        return jsonify(error="already capturing"), 409
    if not camera.ok:
        return jsonify(error="no camera"), 409
    capture["busy"] = True
    threading.Thread(target=_burst, args=(label, n, float(body.get("interval", 0.12))), daemon=True).start()
    return jsonify(ok=True)


def _burst(label, n, interval):
    try:
        folder = class_dir(label)
        folder.mkdir(parents=True, exist_ok=True)
        burst = time.strftime("%Y%m%d-%H%M%S") + f"-{int(time.time() * 1000) % 1000:03d}"
        saved = []
        for i in range(n):
            frame = camera.read()
            if frame is not None:
                path = folder / f"{burst}_{i:03d}.png"
                cv2.imwrite(str(path), for_saving(frame))
                saved.append(path)
            time.sleep(interval)
        capture["last_burst"] = saved
        capture["message"] = f"Saved {len(saved)} photos of {describe(label)}"
    finally:
        capture["busy"] = False


@app.route("/api/undo", methods=["POST"])
def api_undo():
    n = 0
    for p in capture["last_burst"]:
        if p.exists():
            p.unlink()
            n += 1
    capture["last_burst"] = []
    capture["message"] = f"Deleted the last {n} photos"
    return jsonify(ok=True, deleted=n)


@app.route("/api/photos/<int:label>")
def api_photos(label):
    files = sorted(class_dir(label).glob("*.png"), reverse=True)[:48]
    return jsonify([f.name for f in files])


@app.route("/api/photos/<int:label>/<name>", methods=["DELETE"])
def api_delete_photo(label, name):
    p = class_dir(label) / name
    if p.parent == class_dir(label) and p.exists():
        p.unlink()
    return jsonify(ok=True)


@app.route("/api/find_frame", methods=["POST"])
def api_find_frame():
    frame = camera.read()
    if frame is None:
        return jsonify(error="no camera picture"), 409
    roi = find_red_frame(frame)
    if roi is None:
        return jsonify(error="Couldn't find the white area inside the red frame. Empty the box and try again."), 409
    cfg["roi"] = roi
    config.save(cfg)
    return jsonify(ok=True, roi=roi)


@app.route("/api/camera", methods=["POST"])
def api_camera():
    idx = int(request.get_json(force=True)["index"])
    camera.open(idx)
    cfg["camera_index"] = idx
    config.save(cfg)
    return jsonify(ok=True)


@app.route("/api/settings", methods=["POST"])
def api_settings():
    body = request.get_json(force=True)
    if "secret_code" in body:
        code = [int(d) for d in body["secret_code"]]
        if not config.valid_code(code):
            return jsonify(error="The code must be 3 different digits from 0 to 5 (0 = fist)"), 400
        cfg["secret_code"] = code
    for k in ("min_confidence", "stable_seconds", "clear_seconds", "digit_timeout",
              "unlock_seconds", "lockout_seconds"):
        if k in body:
            cfg[k] = float(body[k])
    if "max_failures" in body:
        cfg["max_failures"] = int(body["max_failures"])
    config.save(cfg)
    return jsonify(ok=True)


@app.route("/api/button", methods=["POST"])
def api_button():
    with lock:
        policy.press_button(time.monotonic())
    return jsonify(ok=True)


@app.route("/api/train", methods=["POST"])
def api_train():
    if training["running"]:
        return jsonify(error="already training"), 409
    body = request.get_json(force=True)
    epochs = max(1, min(300, int(body.get("epochs", 30))))
    training.update(running=True, stop=False, epoch=0, epochs=epochs, history=None,
                    best_acc=None, confusion=None, error=None)
    training["log"].clear()
    threading.Thread(target=_train, args=(epochs,), daemon=True).start()
    return jsonify(ok=True)


def _train(epochs):
    from train import train

    def progress(p):
        training.update(epoch=p["epoch"], history=p["history"], best_acc=p["best_acc"],
                        confusion=p["confusion"])

    try:
        train(epochs=epochs, progress=progress, should_stop=lambda: training["stop"],
              log=training["log"].append)
        load_predictor()
    except Exception as e:
        training["error"] = str(e)
    finally:
        training["running"] = False


@app.route("/api/dedupe", methods=["POST"])
def api_dedupe():
    if training["running"] or capture["busy"]:
        return jsonify(error="wait until training/capturing is finished"), 409
    from dedupe import dedupe
    moved = dedupe(log=lambda s: None)
    return jsonify(ok=True, moved=moved, total=sum(moved.values()))


@app.route("/api/models")
def api_models():
    from train import RUNS_DIR
    files = sorted(RUNS_DIR.glob("*.pt"), reverse=True) if RUNS_DIR.exists() else []
    return jsonify([f.name for f in files])


@app.route("/api/models/restore", methods=["POST"])
def api_models_restore():
    import shutil
    from train import RUNS_DIR, archive_current_model
    name = request.get_json(force=True).get("name", "")
    src = RUNS_DIR / name
    if training["running"] or src.parent != RUNS_DIR or not src.exists():
        return jsonify(error="can't restore that model now"), 400
    archive_current_model(log=lambda s: None)
    shutil.copy2(src, MODEL_PATH)
    load_predictor()
    return jsonify(ok=True)


@app.route("/api/train/stop", methods=["POST"])
def api_train_stop():
    training["stop"] = True
    return jsonify(ok=True)


if __name__ == "__main__":
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    load_predictor()
    camera.open()
    threading.Thread(target=processing_loop, daemon=True).start()
    print("\n  The Impenetrable Safe control panel:  http://localhost:5050")
    print("  Leave this window open while you use it. Press Ctrl+C here to stop.\n")
    import webbrowser
    threading.Timer(1.5, lambda: webbrowser.open("http://localhost:5050")).start()
    # 127.0.0.1 = only this computer can open the page (it streams your camera)
    app.run(host="127.0.0.1", port=5050, threaded=True, debug=False)
