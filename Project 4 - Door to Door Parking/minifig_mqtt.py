"""Detect the green LEGO minifigure in the camera feed and publish its box over MQTT.

Uses the weights from train_minifig.py. Each published message is JSON:

    {"x": 412, "y": 260, "width": 85, "height": 140, "img_w": 1280, "img_h": 720}

x, y are the CENTER of the bounding box, and width, height its size, all in
pixels of the camera frame; img_w, img_h are the frame's size. Only the most
confident detection is sent, and nothing is sent on frames where no minifig
is found.

Usage:
    python minifig_mqtt.py                # default camera (0)
    python minifig_mqtt.py --camera 1     # another camera, e.g. iPhone Continuity Camera
    python minifig_mqtt.py --no-preview   # no video window

Press q in the preview window (or Ctrl+C) to stop.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
from ultralytics import YOLO

# mqttlib.py lives in ME193/MQTT/.
ME193 = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ME193 / "MQTT"))
from mqttlib import MQTTClient

# Trained weights, copied here from runs/ by train_minifig.py (tracked in git).
WEIGHTS = Path(__file__).resolve().parent / "green_minifig.pt"
TOPIC = f"ME193/minifig/tashamia"
CONFIDENCE = 0.5          # ignore detections less sure than this
PUBLISH_INTERVAL = 0.1    # seconds between messages (~10 per second max)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--no-preview", action="store_true")
    args = parser.parse_args()

    model = YOLO(str(WEIGHTS))
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        sys.exit(f"Could not open camera {args.camera}")

    last_publish = 0.0
    with MQTTClient() as client:
        print(f"Publishing minifig boxes to '{TOPIC}'. Press q or Ctrl+C to stop.")
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    print("Camera frame failed; stopping.")
                    break

                result = model(frame, conf=CONFIDENCE, device="mps", verbose=False)[0]

                if len(result.boxes) > 0 and time.time() - last_publish >= PUBLISH_INTERVAL:
                    best = result.boxes.conf.argmax()
                    x, y, w, h = result.boxes.xywh[best].tolist()
                    message = json.dumps({
                        "x": round(x), "y": round(y),
                        "width": round(w), "height": round(h),
                        "img_w": frame.shape[1], "img_h": frame.shape[0],
                    })
                    client.publish(TOPIC, message)
                    last_publish = time.time()
                    print(message)

                if not args.no_preview:
                    cv2.imshow("minifig", result.plot())
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
        except KeyboardInterrupt:
            pass
        finally:
            cap.release()
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
