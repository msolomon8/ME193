"""Browser camera to Mac to LEGO AprilTag centering.

The Mac serves a page. Open that page in iPhone Safari, allow the back camera,
and Safari sends JPEG frames to the Mac. The Mac detects the AprilTag and drives
the LEGO Double Motor. No camera app or camera URL is required.

Camera-only test:
    python phone-browser-control.py --test --https --port 8443

Robot mode:
    python phone-browser-control.py --https --port 8443
"""

import argparse
import base64
import json
import ssl
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2
import legoeducation as le
import numpy as np

from lelib import doubleMotor


CARD_COLOR = le.LEGO_COLOR_GREEN
CARD_SERIAL = 997
TURN_SPEED = 30
FORWARD_SPEED = 35
REVERSE_SPEED = 25
CENTER_TOLERANCE = 0.08
MIN_AREA = 800
MAX_AREA = 9000
WINDOW_NAME = "iPhone AprilTag Centering"

latest_frame = None
frame_lock = threading.Lock()


def detector():
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36H11)
    parameters = cv2.aruco.DetectorParameters()
    parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_APRILTAG
    return cv2.aruco.ArucoDetector(dictionary, parameters)


def page():
    return """<!doctype html>
<html><head><meta name="viewport" content="width=device-width,initial-scale=1"></head>
<body style="background:#111;color:white;font-family:sans-serif;text-align:center">
<h3>iPhone camera is sending to the Mac</h3>
<video id="video" autoplay playsinline muted style="width:95%;max-width:600px"></video>
<p id="status">Requesting back-camera permission...</p>
<script>
const video = document.getElementById('video');
const status = document.getElementById('status');
const canvas = document.createElement('canvas');
const context = canvas.getContext('2d');
async function start() {
  try {
    const stream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: { ideal: 'environment' }, width: { ideal: 640 }, height: { ideal: 480 } },
      audio: false
    });
    video.srcObject = stream;
    await video.play();
    status.textContent = 'Camera active. Keep the AprilTag visible.';
    setInterval(sendFrame, 150);
  } catch (error) {
    status.textContent = 'Camera permission failed: ' + error.name;
  }
}
function sendFrame() {
  if (!video.videoWidth) return;
  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;
  context.drawImage(video, 0, 0);
  canvas.toBlob(blob => {
    if (!blob) return;
    fetch('/frame', {method:'POST', body:blob, headers:{'Content-Type':'image/jpeg'}})
      .catch(() => { status.textContent = 'Mac connection lost'; });
  }, 'image/jpeg', 0.7);
}
start();
</script></body></html>"""


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/":
            body = page().encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        self.send_response(404)
        self.end_headers()

    def do_POST(self):
        if self.path != "/frame":
            self.send_response(404)
            self.end_headers()
            return
        length = int(self.headers.get("Content-Length", "0"))
        data = self.rfile.read(length)
        image = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is not None:
            global latest_frame
            with frame_lock:
                latest_frame = image
        self.send_response(204)
        self.end_headers()

    def log_message(self, *_args):
        return


def command_for_tag(center_x, width, area):
    offset = (center_x - width / 2) / max(width / 2, 1)
    left = right = 0
    if abs(offset) > CENTER_TOLERANCE:
        turn = TURN_SPEED if abs(offset) > 0.30 else TURN_SPEED * 0.55
        if offset < 0:
            left, right = turn, -turn
        else:
            left, right = -turn, turn
    if area < MIN_AREA:
        left += FORWARD_SPEED
        right += FORWARD_SPEED
    elif area > MAX_AREA:
        left -= REVERSE_SPEED
        right -= REVERSE_SPEED
    return int(max(-100, min(100, left))), int(max(-100, min(100, right)))


def control(motor):
    tag_detector = detector()
    last_command = None

    def send(left, right):
        nonlocal last_command
        command = (left, right)
        if command != last_command:
            if motor is not None:
                motor.movement_move_tank(left, right, blocking=False)
            last_command = command

    try:
        while True:
            with frame_lock:
                frame = None if latest_frame is None else latest_frame.copy()
            if frame is None:
                send(0, 0)
                time.sleep(0.02)
                continue

            height, width = frame.shape[:2]
            corners, ids, _ = tag_detector.detectMarkers(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
            if ids is None:
                send(0, 0)
                message = "NO APRILTAG"
                color = (0, 0, 255)
            else:
                best = max(
                    range(len(corners)),
                    key=lambda i: cv2.contourArea(corners[i][0].astype(np.float32)),
                )
                points = corners[best][0]
                tag_area = cv2.contourArea(points.astype(np.float32))
                center_x = float(points[:, 0].mean())
                left, right = command_for_tag(center_x, width, tag_area)
                send(left, right)
                cv2.aruco.drawDetectedMarkers(frame, [corners[best]], ids[best:best + 1])
                message = f"ID {int(ids[best][0])}  L={left} R={right}"
                color = (0, 255, 0)

            cv2.line(frame, (width // 2, 0), (width // 2, height), (255, 255, 255), 1)
            cv2.putText(frame, message, (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            cv2.imshow(WINDOW_NAME, frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        send(0, 0)
        cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--test",
        action="store_true",
        help="Receive and display the iPhone feed without connecting to the car",
    )
    parser.add_argument(
        "--https",
        action="store_true",
        help="Serve the phone page over HTTPS for iPhone camera permission",
    )
    parser.add_argument("--cert", default="phone-camera-cert.pem")
    parser.add_argument("--key", default="phone-camera-key.pem")
    args = parser.parse_args()

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    if args.https:
        context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        context.load_cert_chain(args.cert, args.key)
        server.socket = context.wrap_socket(server.socket, server_side=True)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    protocol = "https" if args.https else "http"
    print(f"Phone page ready at {protocol}://<MAC-IP>:{args.port}/")

    if args.test:
        print("Test mode: no LEGO motor connection will be attempted.")
        print("Open the phone page in Safari and allow camera access.")
        control(None)
        server.shutdown()
        return

    motor = doubleMotor()
    print("Connecting to LEGO Double Motor on green 0997 card...")
    motor.connect(card_serial=CARD_SERIAL, card_color=CARD_COLOR)
    print("Connected. Open the phone page in Safari and allow camera access.")
    control(motor)
    motor.disconnect()
    server.shutdown()


if __name__ == "__main__":
    main()
