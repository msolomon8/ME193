"""Detect the green AND blue LEGO minifigures in the camera feed and send each car its box over MQTT.

Same as minifig_mqtt.py, but runs both trained models on every frame:
    green minifig -> ME193/minifig/tashamia/green   (car_green_app parks it on the LEFT line)
    blue minifig  -> ME193/minifig/tashamia/blue    (car_blue_app parks it on the RIGHT line)

Each published message is JSON, same as minifig_mqtt.py:

    {"x": 412, "y": 260, "width": 85, "height": 140, "img_w": 1280, "img_h": 720}

x, y are the CENTER of the bounding box, and width, height its size, all in
pixels of the camera frame; img_w, img_h are the frame's size. Nothing is sent
for a minifig that isn't found, so that car stops.

Usage:
    python both_minifigs_mqtt.py                # default camera (0)
    python both_minifigs_mqtt.py --camera 1     # another camera, e.g. iPhone Continuity Camera
    python both_minifigs_mqtt.py --no-preview   # no video window

Press q in the preview window (or Ctrl+C) to stop.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
from ultralytics import YOLO

# mqttlib.py lives in ME193/MQTT/ (this script is in ME193/Project 4/Both Car Mini Figures/).
PROJECT4 = Path(__file__).resolve().parent.parent
ME193 = PROJECT4.parent
sys.path.insert(0, str(ME193 / "MQTT"))
from mqttlib import MQTTClient

TOPIC_BASE = "ME193/minifig/tashamia"
CONFIDENCE = 0.5          # ignore detections less sure than this
SAME_FIGURE_IOU = 0.5     # green + blue boxes overlapping this much = one figure seen twice
PUBLISH_INTERVAL = 0.1    # seconds between messages per car (~10 per second max)

# name -> (trained weights, MQTT topic, TARGET_X line in the preview, BGR color)
# TARGET_X must match TARGET_X in that car's app.
MINIFIGS = {
    "green": (PROJECT4 / "green_minifig.pt", f"{TOPIC_BASE}/green", 0.25, (0, 200, 0)),
    "blue": (PROJECT4 / "blue_minifig" / "blue_minifig.pt", f"{TOPIC_BASE}/blue", 0.75, (255, 80, 0)),
}


def best_box(model, frame):
    """Most confident (x, y, w, h, conf) box in pixels, or None."""
    boxes = model(frame, conf=CONFIDENCE, device="mps", verbose=False)[0].boxes
    if len(boxes) == 0:
        return None
    i = boxes.conf.argmax()
    return (*boxes.xywh[i].tolist(), float(boxes.conf[i]))


def iou(a, b):
    """Overlap of two (x, y, w, h, ...) center boxes: 0 = apart, 1 = identical."""
    iw = min(a[0] + a[2] / 2, b[0] + b[2] / 2) - max(a[0] - a[2] / 2, b[0] - b[2] / 2)
    ih = min(a[1] + a[3] / 2, b[1] + b[3] / 2) - max(a[1] - a[3] / 2, b[1] - b[3] / 2)
    inter = max(0.0, iw) * max(0.0, ih)
    union = a[2] * a[3] + b[2] * b[3] - inter
    return inter / union if union > 0 else 0.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--no-preview", action="store_true")
    args = parser.parse_args()

    models = {name: YOLO(str(spec[0])) for name, spec in MINIFIGS.items()}
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        sys.exit(f"Could not open camera {args.camera}")

    last_publish = {name: 0.0 for name in MINIFIGS}
    with MQTTClient() as client:
        for name, (_, topic, target_x, _) in MINIFIGS.items():
            print(f"{name} minifig -> '{topic}' (parks at x = {target_x})")
        print("Press q or Ctrl+C to stop.")
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    print("Camera frame failed; stopping.")
                    break

                found = {name: best_box(model, frame) for name, model in models.items()}

                # Both models fired on the same spot -> it's one figure; keep the surer one
                g, b = found["green"], found["blue"]
                if g and b and iou(g, b) > SAME_FIGURE_IOU:
                    found["blue" if g[4] >= b[4] else "green"] = None

                preview = frame.copy()
                for name, (_, topic, target_x, color) in MINIFIGS.items():
                    # Where this car parks its minifig
                    line_x = round(target_x * preview.shape[1])
                    cv2.line(preview, (line_x, 0), (line_x, preview.shape[0]), color, 2)

                    box = found[name]
                    if box is None:
                        continue
                    x, y, w, h, conf = box

                    if time.time() - last_publish[name] >= PUBLISH_INTERVAL:
                        message = json.dumps({
                            "x": round(x), "y": round(y),
                            "width": round(w), "height": round(h),
                            "img_w": frame.shape[1], "img_h": frame.shape[0],
                        })
                        client.publish(topic, message)
                        last_publish[name] = time.time()
                        print(f"{name:5} {message}")

                    x1, y1 = round(x - w / 2), round(y - h / 2)
                    cv2.rectangle(preview, (x1, y1), (round(x + w / 2), round(y + h / 2)), color, 3)
                    cv2.putText(preview, f"{name.capitalize()} Minifig {conf:.2f}", (x1, max(20, y1 - 8)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

                if not args.no_preview:
                    cv2.imshow("both minifigs", preview)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
        except KeyboardInterrupt:
            pass
        finally:
            cap.release()
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
