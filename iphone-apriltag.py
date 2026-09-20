"""iPhone Safari camera to Mac AprilTag preview and LEGO control.

The iPhone opens the HTTPS page in Safari and sends camera frames to this Mac.
The Mac preview starts immediately, even if the LEGO motor is disconnected.
Use --test to verify the phone feed without connecting to the car.
"""

import argparse
import base64
import json
import math
import os
import socket
import ssl
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2
import legoeducation as le
import numpy as np

from lelib import doubleMotor, singleMotor


CARD_COLOR = le.LEGO_COLOR_GREEN
CARD_SERIAL = "0997"
TAG_SIZE_METERS = 2.025 * 0.0254
TARGET_DISTANCE_METERS = 0.45
DISTANCE_TOLERANCE_METERS = 0.03
DISTANCE_KP = 80.0
DRIVE_DIRECTION_SIGN = -1
CENTER_TOLERANCE_PIXELS = 35
CENTERING_KP = 0.025
YAW_ENTER_TOLERANCE_DEG = 3.0
YAW_EXIT_TOLERANCE_DEG = 6.0
KP = 0.10
KI = 0.0
KD = 0.01
MAX_SPEED = 25
YAW_KP = -0.08
YAW_KI = 0.0
YAW_KD = 0.002
MAX_YAW_CORRECTION = 6
MAX_ACCEL_PER_SEC = 30
INVERT_RIGHT_MOTOR = True
CAMERA_PAN_SPEED = 25
CAMERA_MOTOR_SIGN = 1
CAMERA_SCAN_SPEED = 3
CAMERA_SCAN_DIRECTION = 1
CAMERA_SCAN_ACCELERATION = 20
TAG_LOST_HOLD_SECONDS = 1.0
CAMERA_REACQUIRE_SPEED = 3
CAMERA_REACQUIRE_INTERVAL_SECONDS = 4.0
CAMERA_YAW_ALIGN_SPEED = 5
CAMERA_YAW_TOLERANCE_DEG = 5.0
CAMERA_YAW_SIGN = 1
WINDOW_NAME = "iPhone AprilTag"

latest_frame = None
frame_lock = threading.Lock()
motor = None
camera_motor = None
motor_lock = threading.Lock()
camera_motor_lock = threading.Lock()
stop_event = threading.Event()


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

    def log_message(self, *_args):
        return


class PID:
    def __init__(self, kp, ki, kd, output_limit):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.output_limit = output_limit
        self.integral = 0.0
        self.previous_error = 0.0
        self.previous_time = None

    def reset(self):
        self.integral = 0.0
        self.previous_error = 0.0
        self.previous_time = None

    def compute(self, error):
        now = time.monotonic()
        dt = 0.0 if self.previous_time is None else now - self.previous_time
        self.previous_time = now
        self.integral += error * dt
        if self.ki:
            max_integral = self.output_limit / self.ki
            self.integral = max(-max_integral, min(max_integral, self.integral))
        derivative = 0.0 if dt <= 0 else (error - self.previous_error) / dt
        self.previous_error = error
        output = self.kp * error + self.ki * self.integral + self.kd * derivative
        return max(-self.output_limit, min(self.output_limit, output))


class SlewLimiter:
    def __init__(self, max_change_per_second):
        self.max_change_per_second = max_change_per_second
        self.value = 0.0
        self.previous_time = None

    def step(self, target):
        now = time.monotonic()
        dt = 0.0 if self.previous_time is None else now - self.previous_time
        self.previous_time = now
        if dt <= 0:
            self.value = target
            return self.value
        maximum_delta = self.max_change_per_second * dt
        delta = max(-maximum_delta, min(maximum_delta, target - self.value))
        self.value += delta
        return self.value


def camera_matrix(width, height):
    focal_length = width
    return np.array(
        [[focal_length, 0, width / 2], [0, focal_length, height / 2], [0, 0, 1]],
        dtype=np.float32,
    )


def marker_object_points(tag_size):
    half = tag_size / 2.0
    return np.array(
        [[-half, half, 0], [half, half, 0], [half, -half, 0], [-half, -half, 0]],
        dtype=np.float32,
    )


def estimate_pose(points, tag_size, camera, distortion):
    success, rotation, translation = cv2.solvePnP(
        marker_object_points(tag_size), points.astype(np.float32), camera, distortion
    )
    if not success:
        return None
    return rotation, translation


def yaw_from_rotation(rotation):
    matrix, _ = cv2.Rodrigues(rotation)
    return math.degrees(math.atan2(matrix[0][2], matrix[2][2]))


def angle_difference(angle, target):
    return (angle - target + 180) % 360 - 180


def send_motor(left, right):
    with motor_lock:
        active_motor = motor
    if active_motor is None:
        return False
    right_command = -right if INVERT_RIGHT_MOTOR else right
    active_motor.motor_run(motor=le.MOTOR_LEFT, speed=int(round(left)), blocking=False)
    active_motor.motor_run(motor=le.MOTOR_RIGHT, speed=int(round(right_command)), blocking=False)
    return True


def drive_car(left_speed, right_speed, last_sent, send_threshold=1):
    """Send the two wheel commands used while aligning and parking."""
    left_int = int(round(left_speed))
    right_int = int(round(right_speed))
    if (
        last_sent is not None
        and abs(left_int - last_sent[0]) < send_threshold
        and abs(right_int - last_sent[1]) < send_threshold
    ):
        return last_sent
    if send_motor(left_int, right_int):
        return left_int, right_int
    return last_sent


def send_camera_motor(speed):
    with camera_motor_lock:
        active_motor = camera_motor
    if active_motor is None:
        return False
    active_motor.run(speed=int(round(CAMERA_MOTOR_SIGN * speed)))
    return True


def stop_camera_motor():
    with camera_motor_lock:
        active_motor = camera_motor
    if active_motor is not None:
        active_motor.motor_stop()


def connect_motor():
    global camera_motor, motor
    try:
        print("Connecting to camera pan motor on green 0997 card...")
        camera_candidate = singleMotor()
        camera_candidate.connect(card_serial=CARD_SERIAL, card_color=CARD_COLOR)
        camera_candidate.motor_reset_relative_position(position=0)
        camera_candidate.motor_set_acceleration(
            CAMERA_SCAN_ACCELERATION,
            CAMERA_SCAN_ACCELERATION,
            blocking=False,
        )
        with camera_motor_lock:
            camera_motor = camera_candidate
        print("Camera pan motor connected.")

        print("Connecting to LEGO Double Motor on green 0997 card...")
        car_candidate = doubleMotor()
        car_candidate.connect(card_serial=CARD_SERIAL, card_color=CARD_COLOR)
        with motor_lock:
            motor = car_candidate
        print("Car motors connected. Robot control is enabled.")
    except Exception as exc:
        print(f"Motor unavailable: {exc}")
        print("Camera preview remains active, but motor control is disabled.")


def choose_largest_tag(corners, ids):
    if ids is None or len(corners) == 0:
        return None
    index = max(
        range(len(corners)),
        key=lambda item: cv2.contourArea(corners[item][0].astype(np.float32)),
    )
    return corners[index][0], int(ids[index])


def preview_loop(enable_motor):
    parameters = cv2.aruco.DetectorParameters()
    parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_APRILTAG
    tag_detector = cv2.aruco.ArucoDetector(
        cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36H11),
        parameters,
    )
    distance_pid = PID(DISTANCE_KP, 0.0, 0.0, MAX_SPEED)
    yaw_pid = PID(YAW_KP, YAW_KI, YAW_KD, MAX_YAW_CORRECTION)
    left_slew = SlewLimiter(MAX_ACCEL_PER_SEC)
    right_slew = SlewLimiter(MAX_ACCEL_PER_SEC)
    distortion = np.zeros((5, 1), dtype=np.float32)
    camera = None
    yaw_parked = False
    car_aligned = False
    tag_acquired = False
    tag_lost_since = None
    reacquiring_camera = False
    reacquire_direction = 1
    last_reacquire_switch = time.monotonic()
    camera_scanning = False
    last_command = None

    while not stop_event.is_set():
        with frame_lock:
            frame = None if latest_frame is None else latest_frame.copy()

        if frame is None:
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(frame, "WAITING FOR IPHONE CAMERA", (35, 220), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 200, 255), 2)
            cv2.putText(frame, "Open the HTTPS URL in Safari", (80, 265), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        else:
            height, width = frame.shape[:2]
            corners, ids, _ = tag_detector.detectMarkers(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
            tag = choose_largest_tag(corners, ids)
            if tag is None:
                distance_pid.reset()
                yaw_pid.reset()
                left_slew.value = 0.0
                right_slew.value = 0.0
                now = time.monotonic()
                if tag_acquired:
                    if tag_lost_since is None:
                        tag_lost_since = now
                    send_motor(0, 0)
                    if (
                        now - tag_lost_since >= TAG_LOST_HOLD_SECONDS
                    ):
                        reacquiring_camera = True
                    if reacquiring_camera and enable_motor:
                        if now - last_reacquire_switch >= CAMERA_REACQUIRE_INTERVAL_SECONDS:
                            reacquire_direction *= -1
                            last_reacquire_switch = now
                        send_camera_motor(reacquire_direction * CAMERA_REACQUIRE_SPEED)
                        label, color = "REACQUIRING APRILTAG", (0, 200, 255)
                    else:
                        stop_camera_motor()
                        camera_scanning = False
                        label, color = "TAG LOST - CAR STOPPED", (0, 165, 255)
                else:
                    if enable_motor and not camera_scanning:
                        send_camera_motor(CAMERA_SCAN_DIRECTION * CAMERA_SCAN_SPEED)
                        camera_scanning = True
                    label, color = "SEARCHING FOR APRILTAG", (0, 200, 255)
                command = (0, 0)
            else:
                points, tag_id = tag
                tag_acquired = True
                tag_lost_since = None
                if reacquiring_camera:
                    stop_camera_motor()
                    reacquiring_camera = False
                    last_reacquire_switch = time.monotonic()
                camera_scanning = False
                center_x = float(points[:, 0].mean())
                if enable_motor:
                    stop_camera_motor()
                if camera is None:
                    camera = camera_matrix(width, height)
                pose = estimate_pose(points, TAG_SIZE_METERS, camera, distortion)
                if pose is None:
                    stop_camera_motor()
                    command = (0, 0)
                    label, color = "POSE FAILED - STOPPED", (0, 165, 255)
                else:
                    rotation, translation = pose
                    distance = float(translation[2][0])
                    distance_error = distance - TARGET_DISTANCE_METERS
                    center_error = center_x - width / 2
                    yaw_raw = yaw_from_rotation(rotation)
                    yaw_error = angle_difference(yaw_raw, 180.0)
                    if enable_motor and not car_aligned:
                        yaw_aligned = abs(yaw_error) <= YAW_ENTER_TOLERANCE_DEG
                        center_aligned = abs(center_error) <= CENTER_TOLERANCE_PIXELS
                        if yaw_aligned and center_aligned:
                            car_aligned = True
                            stop_camera_motor()
                        else:
                            yaw_correction = yaw_pid.compute(yaw_error)
                            yaw_correction += CENTERING_KP * center_error
                            yaw_correction = max(
                                -MAX_YAW_CORRECTION,
                                min(MAX_YAW_CORRECTION, yaw_correction),
                            )
                            left = left_slew.step(yaw_correction)
                            right = right_slew.step(-yaw_correction)
                            command = (int(round(left)), int(round(right)))
                            last_command = drive_car(
                                command[0], command[1], last_command
                            )
                        if car_aligned:
                            last_command = drive_car(0, 0, last_command)
                        distance_pid.reset()
                        if car_aligned:
                            yaw_pid.reset()
                            left_slew.value = 0.0
                            right_slew.value = 0.0
                            command = (0, 0)
                            label = "CAR PARALLEL AND CENTERED - PARKING"
                        else:
                            label = "ALIGNING CAR PARALLEL TO TAG"
                        color = (0, 200, 255)
                    else:
                        stop_camera_motor()
                        distance_ok = abs(distance_error) <= DISTANCE_TOLERANCE_METERS
                        center_ok = abs(center_error) <= CENTER_TOLERANCE_PIXELS
                        if yaw_parked:
                            yaw_parked = abs(yaw_error) <= YAW_EXIT_TOLERANCE_DEG
                        else:
                            yaw_parked = abs(yaw_error) <= YAW_ENTER_TOLERANCE_DEG
                        yaw_ok = yaw_parked

                        if distance_ok:
                            translation_speed = 0.0
                            distance_pid.reset()
                        else:
                            translation_speed = DRIVE_DIRECTION_SIGN * distance_pid.compute(
                                distance_error
                            )
                        if yaw_ok:
                            yaw_correction = 0.0
                            yaw_pid.reset()
                        else:
                            yaw_correction = yaw_pid.compute(yaw_error)
                        if not center_ok:
                            yaw_correction += CENTERING_KP * center_error
                        yaw_correction = max(
                            -MAX_YAW_CORRECTION,
                            min(MAX_YAW_CORRECTION, yaw_correction),
                        )

                        target_left = translation_speed + yaw_correction
                        target_right = translation_speed - yaw_correction
                        left = left_slew.step(target_left)
                        right = right_slew.step(target_right)
                        command = (int(round(left)), int(round(right)))
                        if distance_ok and yaw_ok and center_ok:
                            label, color = "STOP - PARKED", (0, 220, 0)
                        elif distance_ok and not center_ok:
                            label, color = "CENTERING TAG", (255, 200, 0)
                        elif distance_ok:
                            label, color = "ALIGNING CAR YAW", (255, 200, 0)
                        elif translation_speed > 0:
                            label, color = "DRIVE FORWARD", (0, 255, 255)
                        else:
                            label, color = "DRIVE BACKWARD", (0, 165, 255)
                    label = f"ID {tag_id} {label} L={command[0]} R={command[1]}"
                    cv2.putText(
                        frame,
                        f"dist={distance:.2f}m err={distance_error:+.2f}m yaw={yaw_error:+.1f}deg",
                        (15, 62),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        (255, 255, 255),
                        2,
                    )
                cv2.aruco.drawDetectedMarkers(
                    frame,
                    [points.reshape(1, 4, 2)],
                    np.array([[tag_id]], dtype=np.int32),
                )
            if enable_motor and command != last_command:
                if send_motor(*command):
                    last_command = command
            cv2.line(frame, (width // 2, 0), (width // 2, height), (255, 255, 255), 1)
            cv2.putText(frame, label, (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        cv2.imshow(WINDOW_NAME, frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            stop_event.set()
            break

    if enable_motor:
        send_motor(0, 0)
    cv2.destroyAllWindows()


def local_ip():
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            sock.connect(("192.0.2.1", 80))
            return sock.getsockname()[0]
    except OSError:
        return "<MAC-IP>"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test", action="store_true", help="Disable motor connection and commands")
    parser.add_argument("--https", action="store_true", help="Use HTTPS for iPhone camera permission")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8443)
    parser.add_argument("--cert", default="phone-camera-cert.pem")
    parser.add_argument("--key", default="phone-camera-key.pem")
    args = parser.parse_args()

    server = CameraServer((args.host, args.port), Handler)
    if args.https:
        context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        context.load_cert_chain(args.cert, args.key)
        server.socket = context.wrap_socket(server.socket, server_side=True)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    scheme = "https" if args.https else "http"
    print(f"Open {scheme}://{local_ip()}:{args.port}/ on the iPhone in Safari")

    if not args.test:
        threading.Thread(target=connect_motor, daemon=True).start()
    else:
        print("Test mode: no motor connection will be attempted.")

    try:
        preview_loop(enable_motor=not args.test)
    finally:
        stop_event.set()
        send_motor(0, 0)
        stop_camera_motor()
        with motor_lock:
            active_motor = motor
        if active_motor is not None:
            active_motor.disconnect()
        with camera_motor_lock:
            active_camera_motor = camera_motor
        if active_camera_motor is not None:
            active_camera_motor.disconnect()
        server.shutdown()


if __name__ == "__main__":
    main()
