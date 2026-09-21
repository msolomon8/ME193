"""iPhone Safari camera to Mac AprilTag preview and LEGO control.

Camera is mounted RIGIDLY to the car (no independent pan/search motor).
That's the key difference from updatedphone-apriltag.py: with a fixed
camera, "camera-forward" and "car-forward" are always the same direction,
so there's no hidden pan-angle offset for the steering math to get wrong.
The iPhone opens the HTTPS page in Safari and sends camera frames to this
Mac. The Mac preview starts immediately, even if the LEGO motor is
disconnected. Use --test to verify the phone feed without connecting to
the car.

Search/park behavior:
  SEARCHING - the Double Motor slowly rotates the whole car in place
    (there is no separate camera motor to pan) until a tag is glimpsed.
    The instant a tag appears, the car HOLDS STILL (rather than
    continuing to spin) while it's confirmed for
    CONFIRMATION_FRAMES_REQUIRED consecutive frames -- this is the
    false-positive double-check, and it also keeps the tag from panning
    back out of frame before confirmation completes. If it disappears
    again before being confirmed, the car resumes spinning.
  FOUND - the Double Motor drives forward/backward to reach the target
    distance AND steers to face the tag squarely (yaw), at the same
    time, using PID + hysteresis + DivergenceGuard, with the pose
    readings smoothed (EMA) to avoid chasing camera noise. If the tag is
    lost for TAG_LOST_HOLD_SECONDS, the car stops and the state goes
    back to SEARCHING.
"""

import argparse
import math
import os
import platform
import socket
import ssl
import subprocess
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2
import legoeducation as le
import numpy as np
from lelib import doubleMotor

CARD_COLOR = le.LEGO_COLOR_GREEN
CARD_SERIAL = "0997"
TAG_SIZE_METERS = 2.025 * 0.0254

TARGET_DISTANCE_METERS = 0.45
DISTANCE_TOLERANCE_METERS = 0.03
DRIVE_DIRECTION_SIGN = 1  # flips forward/backward driving direction

CENTER_TOLERANCE_PIXELS = 35
CENTERING_KP = 0.025  # nudges forward/backward drive to re-center the tag
                       # horizontally (the side-facing camera means left/right
                       # in frame is a driving-position error, not a heading
                       # error, so this drives -- it does not turn)
CENTERING_DIRECTION_SIGN = -1  # flips which way centering drives (fwd/back)

YAW_ENTER_TOLERANCE_DEG = 3.0   # must get this close to be considered "parked"
YAW_EXIT_TOLERANCE_DEG = 6.0    # must drift this far out to resume correcting

TARGET_YAW_DEG = 180.0  # with this file's marker_object_points, a tag
                         # squarely facing the camera reads yaw ~= 180

# --- Distance PID (forward/backward) --------------------------------------
DISTANCE_KP = 80.0
DISTANCE_KI = 0.0
DISTANCE_KD = 0.0
MAX_SPEED = 12

# --- Yaw PID (steering, to face the tag squarely) -------------------------
YAW_KP = -0.15
YAW_KI = 0.0
YAW_KD = 0.01
MAX_YAW_CORRECTION = 6

MAX_ACCEL_PER_SEC = 30
INVERT_RIGHT_MOTOR = True

# --- Searching (rotate the whole car; no camera motor) ---------------------
SEARCH_SPIN_YAW = 3              # slow, constant in-place spin while searching
SEARCH_SPIN_DIRECTION_SIGN = 1   # flips which way the car spins while searching
CONFIRMATION_FRAMES_REQUIRED = 5  # consecutive good frames before trusting a tag

# --- Losing the tag while parked/driving ----------------------------------
TAG_LOST_HOLD_SECONDS = 1.0  # how long to wait before giving up and re-searching

# --- Pose smoothing and frame freshness ------------------------------------
POSE_SMOOTHING_ALPHA = 0.25  # lower = smoother but slower to react to real motion
STALE_FRAME_SECONDS = 0.3    # hold the car if the phone feed falls this far behind

WINDOW_NAME = "iPhone AprilTag"

latest_frame = None
latest_frame_time = 0.0
frame_lock = threading.Lock()
motor = None
motor_lock = threading.Lock()
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
            global latest_frame, latest_frame_time
            with frame_lock:
                latest_frame = image
                latest_frame_time = time.monotonic()
            self.send_response(204)
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
        except Exception:
            self.send_response(400)
            self.end_headers()

    def log_message(self, format, *args, **kwargs):
        return


class PID:
    """Minimal PID controller. compute(error) returns a clamped output;
    call reset() whenever control should stop accumulating (e.g. the car
    is parked, or the tag drops out of view)."""

    def __init__(self, kp, ki, kd, output_limit):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.output_limit = output_limit
        self.integral = 0.0
        self.prev_error = 0.0
        self.prev_time = None

    def reset(self):
        self.integral = 0.0
        self.prev_error = 0.0
        self.prev_time = None

    def compute(self, error):
        now = time.monotonic()
        dt = 0.0 if self.prev_time is None else now - self.prev_time
        self.prev_time = now

        self.integral += error * dt
        if self.ki:
            max_integral = self.output_limit / self.ki
            self.integral = max(-max_integral, min(max_integral, self.integral))

        derivative = 0.0 if dt <= 0 else (error - self.prev_error) / dt
        self.prev_error = error

        output = self.kp * error + self.ki * self.integral + self.kd * derivative
        return max(-self.output_limit, min(self.output_limit, output))


class EMA:
    """Exponential moving average filter for a noisy measurement. Smooths
    out per-frame pose jitter (from a low-quality/laggy camera feed) so
    the PID controllers react to the real trend instead of chasing noise.
    Lower alpha means more smoothing but slower to reflect real motion."""

    def __init__(self, alpha):
        self.alpha = alpha
        self.value = None

    def reset(self):
        self.value = None

    def update(self, sample):
        if self.value is None:
            self.value = sample
        else:
            self.value = self.alpha * sample + (1 - self.alpha) * self.value
        return self.value


class SlewLimiter:
    """Limits how fast a value is allowed to change per second, regardless
    of how much the raw input jumps between calls."""

    def __init__(self, max_change_per_sec):
        self.max_change_per_sec = max_change_per_sec
        self.value = 0.0
        self.prev_time = None

    def reset(self, value=0.0):
        self.value = value
        self.prev_time = None

    def step(self, target):
        now = time.monotonic()
        dt = 0.0 if self.prev_time is None else now - self.prev_time
        self.prev_time = now

        if dt <= 0:
            self.value = target
            return self.value

        max_delta = self.max_change_per_sec * dt
        delta = max(-max_delta, min(max_delta, target - self.value))
        self.value += delta
        return self.value


class DivergenceGuard:
    """Watches whether a control loop's |error| is shrinking over time
    while active correction is applied. If it's consistently getting
    WORSE instead, that's the signature of a wrong-sign gain -- this
    flips the sign automatically instead of guessing and re-testing by
    hand. Multiply your PID's output by .sign before sending it:
    `output = guard.sign * pid.compute(error)`."""

    def __init__(self, window_seconds=0.4, min_error_to_monitor=8.0,
                 worsening_margin=2.0, cooldown_seconds=1.0):
        self.window_seconds = window_seconds
        self.min_error_to_monitor = min_error_to_monitor
        self.worsening_margin = worsening_margin
        self.cooldown_seconds = cooldown_seconds
        self.samples = []
        self.sign = 1
        self.last_flip_time = None

    def record(self, error):
        now = time.monotonic()

        if abs(error) < self.min_error_to_monitor:
            self.samples.clear()
            return

        self.samples.append((now, abs(error)))
        cutoff = now - self.window_seconds
        self.samples = [(t, e) for (t, e) in self.samples if t >= cutoff]

        if (self.last_flip_time is not None
                and now - self.last_flip_time < self.cooldown_seconds):
            return

        if len(self.samples) < 4:
            return

        oldest_error = self.samples[0][1]
        newest_error = self.samples[-1][1]

        if newest_error - oldest_error > self.worsening_margin:
            self.sign *= -1
            self.last_flip_time = now
            self.samples.clear()
            print(f"[DivergenceGuard] Error grew from {oldest_error:.1f} to "
                  f"{newest_error:.1f} despite active correction -- "
                  f"flipping sign to {self.sign:+d}")


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


def draw_debug_panel(frame, lines, margin=10, line_height=22, font_scale=0.55):
    """Solid black rectangle in the top-right corner with each string in
    `lines` printed inside it, for readability against a busy background."""
    if not lines:
        return
    frame_height, frame_width = frame.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    thickness = 1

    text_sizes = [cv2.getTextSize(line, font, font_scale, thickness)[0] for line in lines]
    max_text_width = max(size[0] for size in text_sizes)

    panel_width = max_text_width + 2 * margin
    panel_height = line_height * len(lines) + margin

    x2 = frame_width - 10
    x1 = x2 - panel_width
    y1 = 10
    y2 = y1 + panel_height

    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), -1)
    for i, line in enumerate(lines):
        text_y = y1 + margin + (i + 1) * line_height - 6
        cv2.putText(
            frame, line, (x1 + margin, text_y),
            font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA,
        )


def send_motor(left, right):
    with motor_lock:
        active_motor = motor
    if active_motor is None:
        return False
    right_command = -right if INVERT_RIGHT_MOTOR else right
    active_motor.motor_run(motor=le.MOTOR_LEFT, speed=int(round(left)), blocking=False)
    active_motor.motor_run(motor=le.MOTOR_RIGHT, speed=int(round(right_command)), blocking=False)
    return True


def connect_motor():
    global motor
    try:
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

    distance_pid = PID(DISTANCE_KP, DISTANCE_KI, DISTANCE_KD, MAX_SPEED)
    yaw_pid = PID(YAW_KP, YAW_KI, YAW_KD, MAX_YAW_CORRECTION)
    yaw_guard = DivergenceGuard()
    center_guard = DivergenceGuard(min_error_to_monitor=40.0, worsening_margin=15.0)
    distance_error_filter = EMA(POSE_SMOOTHING_ALPHA)
    yaw_error_filter = EMA(POSE_SMOOTHING_ALPHA)
    center_error_filter = EMA(POSE_SMOOTHING_ALPHA)
    left_slew = SlewLimiter(MAX_ACCEL_PER_SEC)
    right_slew = SlewLimiter(MAX_ACCEL_PER_SEC)
    yaw_parked = False

    distortion = np.zeros((5, 1), dtype=np.float32)
    camera = None

    state = "SEARCHING"
    confirmations = 0
    last_seen_tag_id = None
    tag_lost_since = None
    last_command = None
    last_printed_label = None

    while not stop_event.is_set():
        with frame_lock:
            frame = None if latest_frame is None else latest_frame.copy()
            frame_time = latest_frame_time

        if frame is None:
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(frame, "WAITING FOR IPHONE CAMERA", (35, 220),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 200, 255), 2)
            cv2.putText(frame, "Open the HTTPS URL in Safari", (80, 265),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.imshow(WINDOW_NAME, frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                stop_event.set()
                break
            continue

        height, width = frame.shape[:2]
        if camera is None:
            camera = camera_matrix(width, height)

        corners, ids, _ = tag_detector.detectMarkers(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
        tag = choose_largest_tag(corners, ids)

        pose = None
        if tag is not None:
            points, tag_id = tag
            pose = estimate_pose(points, TAG_SIZE_METERS, camera, distortion)

        label, color = "", (255, 255, 255)
        debug_lines = []
        command = (0, 0)

        # --- SEARCHING: rotate the whole car slowly in place (there is no
        # separate camera motor). The spin holds still the instant a tag is
        # glimpsed so it doesn't spin back out of frame before enough
        # confirmations accumulate, and resumes if the tag disappears again
        # before it's fully confirmed.
        if state == "SEARCHING":
            tag_detected = tag is not None and pose is not None

            if tag_detected:
                if tag_id == last_seen_tag_id:
                    confirmations += 1
                else:
                    confirmations = 1
                last_seen_tag_id = tag_id
            else:
                confirmations = 0
                last_seen_tag_id = None

            if confirmations >= CONFIRMATION_FRAMES_REQUIRED:
                state = "FOUND"
                tag_lost_since = None
                distance_pid.reset()
                yaw_pid.reset()
                distance_error_filter.reset()
                yaw_error_filter.reset()
                center_error_filter.reset()
                left_slew.reset()
                right_slew.reset()
                yaw_parked = False
                label, color = "TAG CONFIRMED - SWITCHING TO PARK", (0, 220, 0)
            else:
                if tag_detected:
                    target_left, target_right = 0.0, 0.0
                    label, color = (
                        f"HOLDING - CONFIRMING ({confirmations}/{CONFIRMATION_FRAMES_REQUIRED})",
                        (0, 200, 255),
                    )
                else:
                    target_left = SEARCH_SPIN_DIRECTION_SIGN * SEARCH_SPIN_YAW
                    target_right = -SEARCH_SPIN_DIRECTION_SIGN * SEARCH_SPIN_YAW
                    label, color = "SEARCHING - ROTATING CAR", (0, 200, 255)

                left = left_slew.step(target_left)
                right = right_slew.step(target_right)
                command = (int(round(left)), int(round(right)))

        # --- FOUND: drive forward/backward to reach the target distance
        # and steer to face the tag squarely, at the same time.
        elif state == "FOUND":
            if tag is None or pose is None:
                now = time.monotonic()
                if tag_lost_since is None:
                    tag_lost_since = now
                send_motor(0, 0)
                distance_pid.reset()
                yaw_pid.reset()
                distance_error_filter.reset()
                yaw_error_filter.reset()
                center_error_filter.reset()
                left_slew.reset()
                right_slew.reset()

                if now - tag_lost_since >= TAG_LOST_HOLD_SECONDS:
                    state = "SEARCHING"
                    confirmations = 0
                    last_seen_tag_id = None
                    label, color = "TAG LOST - SEARCHING AGAIN", (0, 165, 255)
                else:
                    label, color = "TAG LOST - HOLDING", (0, 165, 255)
            else:
                points, tag_id = tag
                tag_lost_since = None
                rotation, translation = pose

                center_x = float(points[:, 0].mean())
                distance = float(translation[2][0])
                distance_error = distance_error_filter.update(distance - TARGET_DISTANCE_METERS)
                center_error = center_error_filter.update(center_x - width / 2)
                yaw_raw = yaw_from_rotation(rotation)
                yaw_error = yaw_error_filter.update(angle_difference(yaw_raw, TARGET_YAW_DEG))

                distance_ok = abs(distance_error) <= DISTANCE_TOLERANCE_METERS
                center_ok = abs(center_error) <= CENTER_TOLERANCE_PIXELS

                # Hysteresis: once parked, must exceed the wider EXIT
                # threshold before correction resumes -- stops noise at
                # a single boundary from rapidly toggling states.
                if yaw_parked:
                    yaw_parked = abs(yaw_error) <= YAW_EXIT_TOLERANCE_DEG
                else:
                    yaw_parked = abs(yaw_error) <= YAW_ENTER_TOLERANCE_DEG
                yaw_ok = yaw_parked

                if distance_ok:
                    translation_speed = 0.0
                    distance_pid.reset()
                else:
                    translation_speed = DRIVE_DIRECTION_SIGN * distance_pid.compute(distance_error)

                if yaw_ok:
                    yaw_correction = 0.0
                    yaw_pid.reset()
                else:
                    yaw_guard.record(yaw_error)
                    yaw_correction = yaw_guard.sign * yaw_pid.compute(yaw_error)
                yaw_correction = max(-MAX_YAW_CORRECTION, min(MAX_YAW_CORRECTION, yaw_correction))

                # The camera looks out the SIDE of the car, not out the front.
                # So a tag that's off-center left/right in the image needs to
                # be fixed by driving forward/backward along the car's own
                # lane (which slides the view sideways past the tag), not by
                # turning -- turning only fixes heading (yaw), not position.
                if not center_ok:
                    center_guard.record(center_error)
                    translation_speed += (
                        center_guard.sign * CENTERING_DIRECTION_SIGN * CENTERING_KP * center_error
                    )
                translation_speed = max(-MAX_SPEED, min(MAX_SPEED, translation_speed))

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
                    label, color = "ALIGNING YAW", (255, 200, 0)
                elif translation_speed > 0:
                    label, color = "DRIVE FORWARD", (0, 255, 255)
                elif translation_speed < 0:
                    label, color = "DRIVE BACKWARD", (0, 165, 255)
                else:
                    label, color = "ALIGNING", (255, 200, 0)

                debug_lines = [
                    f"ID {tag_id}",
                    f"dist: {distance:.2f} m err: {distance_error:+.2f}",
                    f"center_err: {center_error:+.0f} px",
                    f"yaw_raw: {yaw_raw:+.1f}",
                    f"yaw_err: {yaw_error:+.1f}",
                    f"yaw_sign: {yaw_guard.sign:+d}",
                ]

                cv2.aruco.drawDetectedMarkers(
                    frame,
                    [points.reshape(1, 4, 2)],
                    np.array([[tag_id]], dtype=np.int32),
                )

        if time.monotonic() - frame_time > STALE_FRAME_SECONDS:
            command = (0, 0)
            label, color = "STALE FEED - HOLDING", (0, 165, 255)

        if label != last_printed_label:
            print(f"[STATE] {label} (command={command})")
            last_printed_label = label

        if enable_motor and command != last_command:
            if send_motor(*command):
                last_command = command

        draw_debug_panel(frame, debug_lines)
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

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test", action="store_true", help="Disable motor connection and commands")
    parser.add_argument("--https", action="store_true", help="Use HTTPS for iPhone camera permission")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8443)
    parser.add_argument("--cert", default="phone-camera-cert.pem")
    parser.add_argument("--key", default="phone-camera-key.pem")
    args = parser.parse_args()

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

    if not args.test:
        threading.Thread(target=connect_motor, daemon=True).start()
    else:
        print("Test mode: no motor connection will be attempted.")

    try:
        preview_loop(enable_motor=not args.test)
    finally:
        stop_event.set()
        send_motor(0, 0)
        with motor_lock:
            active_motor = motor
        if active_motor is not None:
            active_motor.disconnect()
        server.shutdown()


if __name__ == "__main__":
    main()
