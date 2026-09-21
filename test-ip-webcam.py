"""Test that the iPhone Safari camera feed is reaching the Mac.

This mirrors the exact camera transport used by updatedphone-apriltag.py:
the Mac runs an HTTP(S) server, the iPhone opens the page in Safari, and
Safari's JavaScript POSTs JPEG frames to /frame. This script does not
connect to the LEGO robot and does not run AprilTag detection -- it only
proves the phone -> Mac video path works before running the full
controller.

Example:
    python test-ip-webcam.py --https --host 0.0.0.0 --port 8443

Then open https://<MAC-IP>:8443/ on the iPhone in Safari and accept the
certificate warning. Press q in the video window to quit.
"""

import argparse
import os
import platform
import socket
import ssl
import subprocess
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2
import numpy as np

latest_frame = None
frame_lock = threading.Lock()


class CameraServer(ThreadingHTTPServer):
    allow_reuse_address = True


def camera_page():
    return """<!doctype html>
<html><head><meta name="viewport" content="width=device-width,initial-scale=1"></head>
<body style="background:#111;color:#fff;text-align:center;font-family:sans-serif">
<h3>iPhone camera to Mac</h3>
<video id="video" autoplay playsinline muted style="width:95%;max-width:640px"></video>
<p id="status">Requesting back-camera permission...</p>
<script>
const video=document.getElementById('video');
const status=document.getElementById('status');
const canvas=document.createElement('canvas');
const ctx=canvas.getContext('2d');
let sending=false;
async function start(){
  try{
    const stream=await navigator.mediaDevices.getUserMedia({
      video:{
        facingMode:{ideal:'environment'},
        width:{ideal:1280},
        height:{ideal:720},
        focusMode:{ideal:'continuous'}
      },
      audio:false
    });
    video.srcObject=stream;
    await video.play();
    status.textContent='Camera connected. Keep the AprilTag visible.';
    setInterval(sendFrame,50);
  }catch(error){
    status.textContent='Camera permission failed: '+error.name;
  }
}
function sendFrame(){
  if(sending || !video.videoWidth) return;
  sending=true;
  canvas.width=video.videoWidth;
  canvas.height=video.videoHeight;
  ctx.drawImage(video,0,0);
  canvas.toBlob(blob=>{
    if(!blob){sending=false;return;}
    fetch('/frame',{method:'POST',body:blob,headers:{'Content-Type':'image/jpeg'}})
      .then(()=>{status.textContent='Camera connected. Frames reaching Mac.';})
      .catch(()=>{status.textContent='Mac connection lost';})
      .finally(()=>{sending=false;});
  },'image/jpeg',0.82);
}
start();
</script></body></html>"""


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path != "/":
            self.send_response(404)
            self.end_headers()
            return
        body = camera_page().encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        if self.path != "/frame":
            self.send_response(404)
            self.end_headers()
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            data = self.rfile.read(length)
            image = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError("invalid JPEG")
            global latest_frame
            with frame_lock:
                latest_frame = image
            self.send_response(204)
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
        except Exception:
            self.send_response(400)
            self.end_headers()

    def log_message(self, format, *args):
        return


def local_ip():
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            sock.connect(("192.0.2.1", 80))
            return sock.getsockname()[0]
    except OSError:
        return "<MAC-IP>"


def prevent_sleep():
    """Keep the Mac awake for as long as this process runs. macOS suspends
    the network stack on sleep, which kills the HTTPS server and shows up
    on the iPhone as "network connection was lost" even though the URL
    never changed."""
    if platform.system() != "Darwin":
        return None
    try:
        return subprocess.Popen(["caffeinate", "-dis", "-w", str(os.getpid())])
    except OSError:
        return None


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--https", action="store_true", help="Use HTTPS for iPhone camera permission")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8443)
    parser.add_argument("--cert", default="phone-camera-cert.pem")
    parser.add_argument("--key", default="phone-camera-key.pem")
    return parser.parse_args()


def main():
    args = parse_args()

    caffeinate_process = prevent_sleep()
    if caffeinate_process is not None:
        print("Preventing Mac sleep for the duration of this run (caffeinate).")

    server = CameraServer((args.host, args.port), Handler)
    if args.https:
        context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        context.load_cert_chain(args.cert, args.key)
        server.socket = context.wrap_socket(server.socket, server_side=True)

    threading.Thread(target=server.serve_forever, daemon=True).start()
    scheme = "https" if args.https else "http"
    print(f"Open {scheme}://{local_ip()}:{args.port}/ on the iPhone in Safari")
    print("Press q in the video window to stop.")

    frame_count = 0
    started_at = time.monotonic()
    last_report = started_at

    try:
        while True:
            with frame_lock:
                frame = None if latest_frame is None else latest_frame.copy()

            if frame is None:
                blank = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(blank, "WAITING FOR IPHONE CAMERA", (35, 220),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 200, 255), 2)
                cv2.putText(blank, "Open the URL above in Safari", (80, 265),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                cv2.imshow("IP Webcam Test", blank)
            else:
                frame_count += 1
                cv2.putText(
                    frame,
                    f"iPhone camera connected | frames: {frame_count}",
                    (15, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2,
                )
                cv2.imshow("IP Webcam Test", frame)

                now = time.monotonic()
                if now - last_report >= 2:
                    elapsed = now - started_at
                    fps = frame_count / elapsed if elapsed else 0
                    print(f"Receiving frames: {frame_count} total ({fps:.1f} FPS)")
                    last_report = now

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cv2.destroyAllWindows()
        server.shutdown()

    if frame_count:
        print(f"Success: received {frame_count} video frames from the phone.")
    else:
        print("Failure: received zero video frames.")


if __name__ == "__main__":
    main()
