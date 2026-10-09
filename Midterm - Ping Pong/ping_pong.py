"""Midterm - virtual ping-pong with the LEGO Double Motor as your paddle.

A Wii-style table tennis game against the computer. First to 11 wins.
  - POSE (MediaPipe): your right arm's left/right position moves your paddle.
    The paddle is measured from your right shoulder (set on the start page, then
    locked for the game), so you can reach both sides of the table. The camera
    view shows your pose lines and the table edges.
  - IMU (Double Motor gyroscope): you only hit the ball if you SWING while the
    paddle is under it. Harder swings send the ball back faster.
  - HAPTICS (Double Motor): a short buzz when you hit, a long buzz when you
    lose a point, three pulses when the game ends.
  - Where you hit the ball on the paddle aims your return: hit it with the
    left edge and it goes left, the center goes straight.
  - The rally goes back and forth until someone misses. Serve switches every
    2 points.

Still to come: AprilTag start page (level), MQTT score, Q-learning opponent.

Usage (from the ME193 folder, venv python):
    le-venv/bin/python "Midterm - Ping Pong/ping_pong.py"               # Double Motor, red 1142 card
    le-venv/bin/python "Midterm - Ping Pong/ping_pong.py" --calibrate   # redo the swing calibration
    le-venv/bin/python "Midterm - Ping Pong/ping_pong.py" --no-motor    # no LEGO: SPACE = swing
    le-venv/bin/python "Midterm - Ping Pong/ping_pong.py" --camera 1
    le-venv/bin/python "Midterm - Ping Pong/ping_pong.py" --make-tag    # printable AprilTag

Keys:  SPACE swing (--no-motor only)   q quit
"""

import argparse
import json
import math
import random
import sys
import threading
import time
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import PoseLandmarker, PoseLandmarkerOptions, RunningMode

HERE = Path(__file__).resolve().parent
ME193 = HERE.parent
sys.path.insert(0, str(ME193))           # lelib.py is shared across projects

# ---- Camera ----
CAMERA_INDEX = 0
WINDOW_NAME = "Ping Pong"

# ---- Pose ----
POSE_MODEL = ME193 / "pose_landmarker_lite.task"
LEFT_WRIST, RIGHT_WRIST = 15, 16
LEFT_SHOULDER, RIGHT_SHOULDER, RIGHT_ELBOW = 11, 12, 14
PADDLE_WRIST = RIGHT_WRIST   # the hand holding the Double Motor
MIN_VISIBILITY = 0.5         # ignore landmarks MediaPipe isn't this sure it can see
ELBOW_BLEND = 0.3            # paddle = this much elbow + the rest wrist. The elbow moves less
                             # sideways during a swing, so the paddle doesn't jump when you hit
PADDLE_SMOOTHING = 0.85      # 0 = raw position, closer to 1 = smoother but laggier
BODY_SMOOTHING = 0.95        # shoulders barely move, so smooth them a lot
LANE_DEADBAND = 0.03         # ignore paddle wiggles smaller than this (table widths)
# The table is mapped onto your arm's reach, measured from your RIGHT shoulder in
# shoulder widths (so it works close to or far from the camera):
TABLE_CENTER = -0.15         # table center: a little left of your right shoulder (hand in front of you)
ARM_SPAN = 0.8               # each table edge is this many shoulder widths from the center
                             # (smaller = less arm movement needed)
MIN_SHOULDER_WIDTH = 0.06    # fraction of the camera; guards against turning sideways
REACH = 0.85                 # fallback when shoulders aren't visible: camera width per table width
# Lines drawn on the camera view: arms, shoulders, body
SKELETON = [(11, 12), (11, 13), (13, 15), (12, 14), (14, 16), (11, 23), (12, 24), (23, 24)]

# ---- Double Motor (paddle) ----
CARD_SERIAL = "1142"         # red Connection Card
IMU_INTERVAL_MS = 20         # ask for an IMU reading every 20 ms (default 100 ms misses quick swings)
CALIBRATION_FILE = HERE / "swing_calibration.json"
CALIBRATION_SWINGS = 3
MIN_THRESHOLD = 60           # never count less than this as a swing (swings read ~170-750,
                             # holding still under 10, small wrist wiggles under 30)
SWING_FRACTION = 0.45        # threshold = this fraction of your median calibration swing
SWING_END = 0.5              # a swing ends when the gyro drops below this fraction of the threshold
MAX_SWING_TIME = 0.6         # ...or after this long, so waving nonstop still counts as new swings
HIT_BUZZ_MS = 70
MISS_BUZZ_MS = 400
BUZZ_SPEED = 100

# ---- Game ----
WIN_SCORE = 11
SERVES_EACH = 2              # serve switches every 2 points
TRAVEL_TIME = 1.6            # seconds for the computer's shots to reach you (levels change this in step 5)
POINT_PAUSE = 1.3            # seconds the point banner shows before the next serve
GAME_OVER_TIME = 4.0         # seconds the YOU WIN! / YOU LOSE! screen shows
CPU_SERVE_DELAY = 0.8

# Your hits
HIT_WINDOW_START = 0.80      # you can hit once the ball is this far through its flight to you...
HIT_WINDOW_END = 1.15        # ...until this far (past 1 = it's going by you)
SWING_EARLY = 0.30           # a swing up to this many seconds before the window still counts
PADDLE_WIDTH = 0.22          # how close (in table widths) the paddle must be to the ball
AIM_SPREAD = 0.40            # hitting with the paddle's edge sends the ball this far from center
RETURN_SPEEDUP_MAX = 1.4     # a very hard swing makes your shot this much faster

# Computer opponent (the Q-learning policy in step 7 will pick where it aims)
CPU_REACTION = 0.25          # seconds before it starts moving toward your shot
CPU_SPEED = 0.55             # table widths per second it can move
CPU_REACH = 0.10             # how close its paddle must be to return the ball
CPU_ERROR = 0.06             # how far off (std dev) its guess of where the ball lands is

# ---- Game screen (pixels) ----
SCREEN_W, SCREEN_H = 1280, 720
PIP_W, PIP_H = 400, 225      # camera picture-in-picture, bottom-right corner

# Table in perspective: y of the far and near ends, and half widths there
TABLE_FAR_Y, TABLE_NEAR_Y = 250, 600
TABLE_FAR_HALF, TABLE_NEAR_HALF = 190, 400
TABLE_THICKNESS = 18

# ---- Ball flight (progress: 0 = computer's paddle, HIT_LINE = your paddle) ----
HIT_LINE = 0.95
BOUNCE_AT = 0.72             # fraction of each shot's flight where it bounces on the far side
ARC_HEIGHT = 110             # pixels the ball rises above the table over the net
PADDLE_LIFT = 70             # your paddle is drawn this far above the table

# Colors (BGR)
WHITE, BLACK = (255, 255, 255), (0, 0, 0)
GREEN, RED, BLUE = (60, 190, 60), (60, 60, 230), (200, 110, 30)
TEXT_DARK = (70, 50, 40)
TABLE_GREEN, TABLE_EDGE, TABLE_SIDE = (70, 140, 40), (245, 245, 245), (40, 80, 25)
BALL_COLOR = (240, 250, 255)
MY_PADDLE, CPU_PADDLE = (50, 50, 210), (200, 110, 30)   # red, blue
HANDLE = (60, 110, 170)


# ---------------------------------------------------------------------------
# Paddle hardware: Double Motor IMU (swings) + motor (haptics)
# ---------------------------------------------------------------------------

class MotorPaddle:
    """The LEGO Double Motor you hold. A background thread reads its gyroscope
    50 times a second:
      - swing_time = the last moment the gyro's total rotation speed was over the
        threshold, so any part of a swing that lands in the hit window counts
      - a swing ENDS when it drops below SWING_END x threshold (or after
        MAX_SWING_TIME); its peak is the swing's power
    The motor is also the haptic buzzer. Its own vibration would look like a
    swing, so swings are ignored while it buzzes."""

    def __init__(self):
        import legoeducation as le
        from lelib import doubleMotor
        self.le = le
        print(f"Connecting to LEGO Double Motor (red {CARD_SERIAL} card)...")
        self.motor = doubleMotor()
        self.motor.connect(card_serial=CARD_SERIAL, card_color=le.LEGO_COLOR_RED,
                           device_notification_delay=IMU_INTERVAL_MS)
        print("Connected.")
        self.threshold = None        # set by calibration
        self.gyro = 0.0              # latest total rotation speed
        self.swing_time = 0.0        # last time the gyro was over the threshold
        self.swing_peak = 0.0        # its peak gyro (so far)
        self.swing_peaks = []        # finished swings' peaks (for calibration)
        self.quiet_until = 0.0
        self._in_swing = False
        self._swing_start = 0.0
        self._running = True
        threading.Thread(target=self._read_imu, daemon=True).start()

    def _read_imu(self):
        last = None
        while self._running:
            d = self.motor.imu_device
            if d is last:
                time.sleep(0.003)
                continue
            last = d
            g = math.sqrt(d.gyroscopeX ** 2 + d.gyroscopeY ** 2 + d.gyroscopeZ ** 2)
            if math.isnan(g):
                continue
            self.gyro = g
            now = time.time()
            if self.threshold is None or now < self.quiet_until:
                continue
            if g > self.threshold:
                self.swing_time = now
            if not self._in_swing and g > self.threshold:
                self._in_swing = True
                self._swing_start, self.swing_peak = now, g
            elif self._in_swing:
                self.swing_peak = max(self.swing_peak, g)
                if g < self.threshold * SWING_END or now - self._swing_start > MAX_SWING_TIME:
                    self._in_swing = False
                    self.swing_peaks.append(self.swing_peak)

    def buzz(self, ms):
        self.motor.motor_run_for_time(ms, speed=BUZZ_SPEED, motor=self.le.MOTOR_LEFT, blocking=False)
        self.quiet_until = time.time() + ms / 1000 + 0.15
        self._in_swing = False

    def close(self):
        self._running = False
        try:
            self.motor.motor_stop(motor=self.le.MOTOR_BOTH)
        finally:
            self.motor.disconnect()


class KeyboardPaddle:
    """Stand-in for MotorPaddle when there's no LEGO: SPACE is a swing."""

    def __init__(self):
        self.threshold = 1.0
        self.gyro = 0.0
        self.swing_time = 0.0
        self.swing_peak = 0.0
        self.swing_peaks = []

    def swing(self):
        self.swing_time, self.swing_peak = time.time(), 1.5

    def buzz(self, ms):
        pass

    def close(self):
        pass


def threshold_from(peaks):
    return max(SWING_FRACTION * float(np.median(peaks)), MIN_THRESHOLD)


def load_calibration():
    """Threshold from the saved calibration swings (recomputed, so changing
    SWING_FRACTION / MIN_THRESHOLD applies without recalibrating), or None -> calibrate."""
    if CALIBRATION_FILE.exists():
        peaks = [x for x in json.loads(CALIBRATION_FILE.read_text()).get("swing_peaks", [])
                 if x >= MIN_THRESHOLD]
        if len(peaks) >= CALIBRATION_SWINGS:
            return threshold_from(peaks)
    return None


class Calibration:
    """First-run swing calibration, shown on screen:
      1. hold the paddle still for 2 s -> measures the gyro's resting noise
      2. swing CALIBRATION_SWINGS times (detected with a low, temporary threshold)
      3. threshold = SWING_FRACTION of your median swing peak, so real swings clear it easily
         but turning your wrist to line up the paddle doesn't.
    Nothing under MIN_THRESHOLD counts, so small wiggles can't be mistaken for swings."""

    STILL_TIME = 2.0

    def __init__(self, paddle):
        self.paddle = paddle
        self.t0 = time.time()
        self.rest = []
        self.phase = "still"

    def update(self):
        """Returns True when done (and paddle.threshold is set)."""
        p = self.paddle
        if self.phase == "still":
            self.rest.append(p.gyro)
            if time.time() - self.t0 > self.STILL_TIME:
                mean, std = float(np.mean(self.rest)), float(np.std(self.rest))
                p.threshold = max(mean + 6 * std, MIN_THRESHOLD)   # temporary
                p.swing_peaks.clear()
                self.phase = "swing"
            return False
        if len(p.swing_peaks) >= CALIBRATION_SWINGS:
            p.threshold = threshold_from(p.swing_peaks)
            CALIBRATION_FILE.write_text(json.dumps({"threshold": round(p.threshold, 1),
                                                    "swing_peaks": [round(x, 1) for x in p.swing_peaks]}))
            print(f"Swing threshold {p.threshold:.0f} (saved to {CALIBRATION_FILE.name})")
            return True
        return False

    def draw(self, img):
        rounded_panel(img, 240, 200, 800, 260)
        text(img, "Paddle calibration", SCREEN_W // 2, 260, 1.4, BLUE, 3, center=True)
        if self.phase == "still":
            text(img, "Hold the paddle still...", SCREEN_W // 2, 340, 1.1, center=True)
        else:
            n = len(self.paddle.swing_peaks)
            text(img, f"Now swing it like a ping-pong shot!  {n} / {CALIBRATION_SWINGS}",
                 SCREEN_W // 2, 340, 1.0, center=True)
        draw_gyro_meter(img, self.paddle, 440, 400, 400)


# ---------------------------------------------------------------------------
# Pose: right wrist -> paddle position
# ---------------------------------------------------------------------------

def create_pose():
    # On this Mac's mediapipe (0.10.35), CPU + plain 3-channel SRGB works;
    # GPU only works with 4-channel SRGBA.
    options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(POSE_MODEL),
                                 delegate=BaseOptions.Delegate.CPU),
        running_mode=RunningMode.VIDEO,
        num_poses=1,
    )
    return PoseLandmarker.create_from_options(options)


def find_body(pose, frame, timestamp_ms):
    """{landmark index: (x, y)} for the landmarks MediaPipe can see, as 0-1 fractions
    of the MIRRORED frame. Pose runs on the un-mirrored frame so MediaPipe's "right
    wrist" really is your right wrist; x is flipped afterwards to match the picture."""
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = pose.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), timestamp_ms)
    if not result.pose_landmarks:
        return {}
    return {i: (1 - lm.x, lm.y) for i, lm in enumerate(result.pose_landmarks[0])
            if lm.visibility >= MIN_VISIBILITY}


def paddle_point(body):
    """Where the paddle is: between your right wrist and elbow (or just the wrist)."""
    if PADDLE_WRIST not in body:
        return None
    wx, wy = body[PADDLE_WRIST]
    if RIGHT_ELBOW not in body:
        return wx, wy
    ex, ey = body[RIGHT_ELBOW]
    return ELBOW_BLEND * ex + (1 - ELBOW_BLEND) * wx, ELBOW_BLEND * ey + (1 - ELBOW_BLEND) * wy


class ArmTracker:
    """Turns your pose into a paddle lane (0 = table's left edge, 1 = right edge).
    The table is mapped onto your right arm's reach around your right shoulder,
    in shoulder widths, so the left side is just across your body instead of
    across the whole camera picture.
    The mapping follows your body only while unlocked (the start page). During a
    game it's locked: reaching across turns your shoulders, and a mapping that
    followed them would slide the left edge away from you."""

    def __init__(self):
        self.x = None              # smoothed paddle point x (mirrored camera fraction)
        self.center = None         # smoothed camera x of the table's center
        self.half = None           # smoothed camera width from center to a table edge
        self.lane = 0.5
        self.locked = False

    def update(self, body):
        if LEFT_SHOULDER in body and RIGHT_SHOULDER in body and \
                (not self.locked or self.center is None):
            width = max(abs(body[RIGHT_SHOULDER][0] - body[LEFT_SHOULDER][0]), MIN_SHOULDER_WIDTH)
            center = body[RIGHT_SHOULDER][0] + TABLE_CENTER * width
            half = ARM_SPAN * width
            if self.center is None:
                self.center, self.half = center, half
            else:
                self.center = BODY_SMOOTHING * self.center + (1 - BODY_SMOOTHING) * center
                self.half = BODY_SMOOTHING * self.half + (1 - BODY_SMOOTHING) * half
        point = paddle_point(body)
        if point is not None:
            self.x = point[0] if self.x is None else \
                PADDLE_SMOOTHING * self.x + (1 - PADDLE_SMOOTHING) * point[0]
        if self.x is not None:
            target = min(max(self.x_to_lane(self.x), -0.1), 1.1)
            gap = target - self.lane
            if abs(gap) > LANE_DEADBAND:                  # follow, but ignore tiny wiggles
                self.lane = target - math.copysign(LANE_DEADBAND, gap)
        return self.lane

    def x_to_lane(self, x):
        if self.center is None:                          # no shoulders yet: whole camera
            return 0.5 + (x - 0.5) / REACH
        return 0.5 + (x - self.center) / (2 * self.half)

    def lane_to_x(self, lane):
        if self.center is None:
            return 0.5 + (lane - 0.5) * REACH
        return self.center + (lane - 0.5) * 2 * self.half


# ---------------------------------------------------------------------------
# Ball flight
# ---------------------------------------------------------------------------

class Shot:
    """One shot across the table, from the hitter's paddle to the other paddle.
    flight() goes 0 -> 1 over travel_time (and keeps going if nobody hits it)."""

    def __init__(self, to_player, start_lane, end_lane, travel_time):
        self.to_player = to_player
        self.start_lane, self.end_lane = start_lane, end_lane
        self.p_from, self.p_to = (0.0, HIT_LINE) if to_player else (HIT_LINE, 0.0)
        self.travel_time = travel_time
        self.t0 = time.time()

    def flight(self):
        return (time.time() - self.t0) / self.travel_time

    def progress(self):
        return self.p_from + (self.p_to - self.p_from) * self.flight()

    def lane(self):
        return self.start_lane + (self.end_lane - self.start_lane) * self.flight()

    def height(self):
        return ball_height(self.flight())


def ball_height(f):
    """Pixels above the table at flight fraction f: an arc over the net that
    lands at BOUNCE_AT, then a smaller hop up to the other paddle."""
    if f <= BOUNCE_AT:
        t = f / BOUNCE_AT
        return ARC_HEIGHT * 4 * t * (1 - t) + PADDLE_LIFT * (1 - t)
    t = (f - BOUNCE_AT) / (1 - BOUNCE_AT)
    return max(0.0, ARC_HEIGHT * 0.6 * t * (2 - t))


# ---------------------------------------------------------------------------
# Computer opponent
# ---------------------------------------------------------------------------

class Opponent:
    """Moves its paddle toward where it thinks your shot will land, at a limited
    speed and after a short reaction time, so wide, fast shots can beat it."""

    def __init__(self):
        self.lane = 0.5
        self.target = 0.5
        self.move_after = 0.0

    def track(self, shot):
        """Your shot is on its way: guess where it lands (with some error)."""
        self.target = shot.end_lane + random.gauss(0, CPU_ERROR)
        self.move_after = time.time() + CPU_REACTION

    def go_home(self):
        self.target, self.move_after = 0.5, time.time()

    def update(self, dt):
        if time.time() < self.move_after:
            return
        step = CPU_SPEED * dt
        self.lane += max(-step, min(step, self.target - self.lane))

    def can_reach(self, lane):
        return abs(self.lane - lane) <= CPU_REACH

    def choose_aim(self):
        """Where its next shot goes. Step 7 replaces this with the Q-learning policy."""
        return random.uniform(0.12, 0.88)


# ---------------------------------------------------------------------------
# Drawing: the Wii-style scene
# ---------------------------------------------------------------------------

def table_point(lane, progress):
    """Pixel on the table surface. lane: 0 = left edge, 1 = right edge.
    progress: 0 = far (computer) end, 1 = near (your) end."""
    y = TABLE_FAR_Y + (TABLE_NEAR_Y - TABLE_FAR_Y) * progress
    half = TABLE_FAR_HALF + (TABLE_NEAR_HALF - TABLE_FAR_HALF) * progress
    return int(SCREEN_W / 2 + (lane - 0.5) * 2 * half), int(y)


def scale_at(progress):
    """How big things look at this depth (1 at your end)."""
    return (TABLE_FAR_HALF + (TABLE_NEAR_HALF - TABLE_FAR_HALF) * progress) / TABLE_NEAR_HALF


def make_background():
    """Sky-to-white gradient wall and a light wooden floor, drawn once."""
    bg = np.zeros((SCREEN_H, SCREEN_W, 3), np.uint8)
    horizon = 330
    for y in range(horizon):
        t = y / horizon
        bg[y] = (255 - 20 * t, 215 + 30 * t, 150 + 90 * t)          # light blue -> pale
    for y in range(horizon, SCREEN_H):
        t = (y - horizon) / (SCREEN_H - horizon)
        bg[y] = (150 - 40 * t, 190 - 40 * t, 220 - 30 * t)          # warm wood floor
    cv2.line(bg, (0, horizon), (SCREEN_W, horizon), (120, 160, 190), 2)
    return bg


def draw_table(img):
    far_l, far_r = table_point(0, 0), table_point(1, 0)
    near_r, near_l = table_point(1, 1), table_point(0, 1)
    # front face (thickness), legs, then the top
    cv2.fillPoly(img, [np.array([near_l, near_r, (near_r[0], near_r[1] + TABLE_THICKNESS),
                                 (near_l[0], near_l[1] + TABLE_THICKNESS)])], TABLE_SIDE)
    for x in (near_l[0] + 40, near_r[0] - 40):
        cv2.rectangle(img, (x - 6, near_l[1] + TABLE_THICKNESS), (x + 6, SCREEN_H), (50, 50, 50), -1)
    top = np.array([far_l, far_r, near_r, near_l])
    cv2.fillPoly(img, [top], TABLE_GREEN, cv2.LINE_AA)
    cv2.polylines(img, [top], True, TABLE_EDGE, 3, cv2.LINE_AA)
    cv2.line(img, table_point(0.5, 0), table_point(0.5, 1), TABLE_EDGE, 2, cv2.LINE_AA)
    # net
    nl, nr = table_point(-0.06, 0.5), table_point(1.06, 0.5)
    net_h = int(28 * scale_at(0.5))
    cv2.rectangle(img, (nl[0], nl[1] - net_h), (nr[0], nr[1]), (225, 225, 225), -1)
    for x in range(nl[0], nr[0], 8):
        cv2.line(img, (x, nl[1] - net_h), (x, nl[1]), (170, 170, 170), 1)
    cv2.line(img, (nl[0], nl[1] - net_h), (nr[0], nr[1] - net_h), WHITE, 3)


def draw_paddle(img, center, size, color, alpha=1.0):
    """Round Wii-style paddle with a wooden handle."""
    layer = img.copy()
    cx, cy = center
    r = int(size)
    cv2.rectangle(layer, (cx - r // 5, cy + r // 2), (cx + r // 5, cy + int(1.6 * r)), HANDLE, -1)
    cv2.circle(layer, (cx, cy), r, color, -1, cv2.LINE_AA)
    cv2.circle(layer, (cx, cy), r, BLACK, 2, cv2.LINE_AA)
    cv2.addWeighted(layer, alpha, img, 1 - alpha, 0, img)


def draw_ball(img, lane, progress, height):
    s = scale_at(min(max(progress, 0.0), 1.2))
    shadow = table_point(lane, progress)
    on_table = 0 <= progress <= 1 and 0 <= lane <= 1
    if on_table:
        cv2.ellipse(img, shadow, (int(12 * s), int(5 * s)), 0, 0, 360, (40, 90, 25), -1, cv2.LINE_AA)
    center = (shadow[0], int(shadow[1] - height * s))
    cv2.circle(img, center, int(11 * s), BALL_COLOR, -1, cv2.LINE_AA)
    cv2.circle(img, center, int(11 * s), (150, 150, 150), 1, cv2.LINE_AA)


def rounded_panel(img, x, y, w, h, color=WHITE, alpha=0.85, radius=18):
    layer = img.copy()
    cv2.rectangle(layer, (x + radius, y), (x + w - radius, y + h), color, -1)
    cv2.rectangle(layer, (x, y + radius), (x + w, y + h - radius), color, -1)
    for cx, cy in ((x + radius, y + radius), (x + w - radius, y + radius),
                   (x + radius, y + h - radius), (x + w - radius, y + h - radius)):
        cv2.circle(layer, (cx, cy), radius, color, -1, cv2.LINE_AA)
    cv2.addWeighted(layer, alpha, img, 1 - alpha, 0, img)


def text(img, s, x, y, scale=1.0, color=TEXT_DARK, thick=2, center=False):
    if center:
        x -= cv2.getTextSize(s, cv2.FONT_HERSHEY_DUPLEX, scale, thick)[0][0] // 2
    cv2.putText(img, s, (x, y), cv2.FONT_HERSHEY_DUPLEX, scale, color, thick, cv2.LINE_AA)


def banner(img, s, color, sub=None):
    """Big bubbly message at the top of the screen (POINT!, MISS, YOU WIN!)."""
    w = cv2.getTextSize(s, cv2.FONT_HERSHEY_DUPLEX, 2.2, 5)[0][0] + 80
    if sub:
        w = max(w, cv2.getTextSize(sub, cv2.FONT_HERSHEY_DUPLEX, 0.9, 2)[0][0] + 60)
    h = 150 if sub else 100
    rounded_panel(img, (SCREEN_W - w) // 2, 30, w, h, WHITE, 0.92, 30)
    text(img, s, SCREEN_W // 2, 102, 2.2, color, 5, center=True)
    if sub:
        text(img, sub, SCREEN_W // 2, 155, 0.9, TEXT_DARK, 2, center=True)


def draw_scoreboard(img, me, cpu, player_serving, streak, best):
    rounded_panel(img, 20, 20, 250, 110)
    text(img, "YOU", 40, 62, 0.9, MY_PADDLE)
    text(img, str(me), 200, 64, 1.3, MY_PADDLE, 3)
    text(img, "CPU", 40, 110, 0.9, CPU_PADDLE)
    text(img, str(cpu), 200, 112, 1.3, CPU_PADDLE, 3)
    cv2.circle(img, (150, 54 if player_serving else 102), 7, (0, 200, 255), -1, cv2.LINE_AA)  # serve dot
    rounded_panel(img, 20, 140, 250, 50)
    text(img, f"Streak {streak}   Best {best}", 36, 174, 0.7)


def draw_gyro_meter(img, paddle, x, y, w):
    """Bar showing the paddle's rotation speed, with the swing threshold marked."""
    if paddle.threshold is None:
        top = max(paddle.gyro * 1.5, 1.0)
    else:
        top = paddle.threshold * 3
    cv2.rectangle(img, (x, y), (x + w, y + 22), (220, 220, 220), -1)
    fill = int(w * min(paddle.gyro / top, 1.0))
    over = paddle.threshold is not None and paddle.gyro > paddle.threshold
    cv2.rectangle(img, (x, y), (x + fill, y + 22), GREEN if over else BLUE, -1)
    if paddle.threshold is not None:
        tx = x + int(w * paddle.threshold / top)
        cv2.line(img, (tx, y - 4), (tx, y + 26), RED, 2)
    cv2.rectangle(img, (x, y), (x + w, y + 22), TEXT_DARK, 1)
    text(img, f"swing {paddle.gyro:.0f}", x, y + 46, 0.55, TEXT_DARK, 1)


def draw_pip(img, frame, body, arm, swinging):
    """Mirrored camera view in the bottom-right corner, with:
      - yellow lines where the table's edges and center are (move your arm between them)
      - your pose: arms, shoulders and body as white lines
      - the paddle point: green while you're swinging, red otherwise"""
    small = cv2.resize(cv2.flip(frame, 1), (PIP_W, PIP_H))
    def px(pt):
        return int(pt[0] * PIP_W), int(pt[1] * PIP_H)
    for lane, label, dx in ((0, "left edge", -70), (0.5, "", 0), (1, "right edge", 5)):
        cx = int(arm.lane_to_x(lane) * PIP_W)
        cv2.line(small, (cx, 0), (cx, PIP_H), (0, 220, 255), 1 if lane == 0.5 else 2, cv2.LINE_AA)
        cv2.putText(small, label, (cx + dx, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 220, 255), 1, cv2.LINE_AA)
    for a, b in SKELETON:
        if a in body and b in body:
            cv2.line(small, px(body[a]), px(body[b]), WHITE, 2, cv2.LINE_AA)
    for i in (11, 12, 13, 14, 15, 16):
        if i in body:
            cv2.circle(small, px(body[i]), 3, WHITE, -1, cv2.LINE_AA)
    point = paddle_point(body)
    if point is not None:
        cv2.circle(small, px(point), 9, GREEN if swinging else MY_PADDLE, -1 if swinging else 2, cv2.LINE_AA)
    x, y = SCREEN_W - PIP_W - 20, SCREEN_H - PIP_H - 20
    rounded_panel(img, x - 6, y - 6, PIP_W + 12, PIP_H + 12, WHITE, 1.0, 12)
    img[y:y + PIP_H, x:x + PIP_W] = small
    if PADDLE_WRIST not in body:
        text(img, "Can't see your right wrist", x + 10, y + PIP_H - 12, 0.55, RED, 1)


# ---------------------------------------------------------------------------
# Make a printable AprilTag (same as Project 2)
# ---------------------------------------------------------------------------

def make_tag(tag_id=0, pixels=600):
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36H11)
    tag = cv2.aruco.generateImageMarker(dictionary, tag_id, pixels)
    quiet = pixels // 8                                  # white border so it can be detected
    marker = np.full((pixels + 2 * quiet, pixels + 2 * quiet), 255, np.uint8)
    marker[quiet:-quiet, quiet:-quiet] = tag
    path = HERE / f"apriltag-36h11-id{tag_id}.png"
    cv2.imwrite(str(path), marker)
    print(f"Saved {path}")


# ---------------------------------------------------------------------------
# The game
# ---------------------------------------------------------------------------

class Game:
    """Game states:
      start      waiting for you to swing to begin
      serve      someone is about to serve (you: swing to serve; CPU: after a pause)
      rally      the ball is in play
      point      someone just won a point (banner), then -> serve or game over
      game_over  YOU WIN! / YOU LOSE! for a few seconds, then -> start"""

    def __init__(self, paddle):
        self.paddle = paddle
        self.cpu = Opponent()
        self.best_streak = 0
        self.to_start()

    def set_state(self, state):
        self.state = state
        self.state_time = time.time()

    def swung_since(self, t):
        return self.paddle.swing_time > t

    def to_start(self):
        self.me = self.them = 0
        self.streak = 0
        self.shot = None
        self.message = None
        self.set_state("start")

    def player_serving(self):
        return ((self.me + self.them) // SERVES_EACH) % 2 == 0

    # --- shots ---

    def player_hits(self, paddle_lane, from_lane):
        """Send the ball back. Where it hit the paddle aims it; swing power sets the speed."""
        offset = max(-1.0, min(1.0, (from_lane - paddle_lane) / (PADDLE_WIDTH / 2)))
        aim = min(0.9, max(0.1, 0.5 + offset * AIM_SPREAD + random.gauss(0, 0.05)))
        power = self.paddle.swing_peak / self.paddle.threshold if self.paddle.threshold else 1.0
        speedup = min(RETURN_SPEEDUP_MAX, max(1.0, 0.8 + 0.2 * power))
        self.shot = Shot(False, from_lane, aim, TRAVEL_TIME / speedup)
        self.cpu.track(self.shot)
        self.paddle.buzz(HIT_BUZZ_MS)

    def cpu_hits(self, from_lane):
        self.shot = Shot(True, from_lane, self.cpu.choose_aim(), TRAVEL_TIME)
        self.cpu.go_home()
        self.hit_window_opened = None

    def point(self, player_won):
        if player_won:
            self.me += 1
            self.message = ("POINT!", GREEN)
        else:
            self.them += 1
            self.streak = 0
            self.message = ("MISS", RED)
            self.paddle.buzz(MISS_BUZZ_MS)
        self.set_state("point")

    # --- one frame ---

    def update(self, paddle_lane, dt):
        now = time.time()
        self.cpu.update(dt)

        if self.state == "start":
            if now - self.state_time > 0.5 and self.swung_since(self.state_time + 0.5):
                self.set_state("serve")

        elif self.state == "serve":
            if self.player_serving():
                if self.swung_since(self.state_time + 0.3):
                    self.player_hits(paddle_lane, paddle_lane)
                    self.set_state("rally")
            elif now - self.state_time > CPU_SERVE_DELAY:
                self.cpu_hits(self.cpu.lane)
                self.set_state("rally")

        elif self.state == "rally":
            f = self.shot.flight()
            if self.shot.to_player:
                if f >= HIT_WINDOW_START:
                    if self.hit_window_opened is None:
                        self.hit_window_opened = now
                    lined_up = abs(self.shot.lane() - paddle_lane) <= PADDLE_WIDTH / 2
                    swung = self.swung_since(self.hit_window_opened - SWING_EARLY) and \
                        self.swung_since(self.shot.t0)
                    if lined_up and swung:
                        self.streak += 1
                        self.best_streak = max(self.best_streak, self.streak)
                        self.player_hits(paddle_lane, self.shot.lane())
                    elif f > HIT_WINDOW_END:
                        self.point(player_won=False)
            elif f >= 1.0:
                if self.cpu.can_reach(self.shot.lane()):
                    self.cpu_hits(self.shot.lane())
                else:
                    self.point(player_won=True)

        elif self.state == "point":
            if now - self.state_time > POINT_PAUSE:
                if self.me >= WIN_SCORE or self.them >= WIN_SCORE:
                    self.message = ("YOU WIN!", GREEN) if self.me > self.them else ("YOU LOSE!", RED)
                    self.paddle.buzz(MISS_BUZZ_MS if self.me < self.them else HIT_BUZZ_MS * 4)
                    self.set_state("game_over")
                else:
                    self.shot = None
                    self.cpu.go_home()
                    self.set_state("serve")

        elif self.state == "game_over":
            if now - self.state_time > GAME_OVER_TIME:
                self.to_start()

    def draw(self, img, paddle_lane):
        draw_table(img)
        draw_paddle(img, table_point(self.cpu.lane, -0.08), 34 * scale_at(0), CPU_PADDLE)

        if self.shot is not None and self.state in ("rally", "point"):
            draw_ball(img, self.shot.lane(), self.shot.progress(), self.shot.height())
        elif self.state == "serve" and self.player_serving():
            draw_ball(img, paddle_lane, HIT_LINE, PADDLE_LIFT + 45)   # ball waiting on your paddle
        elif self.state == "serve":
            draw_ball(img, self.cpu.lane, 0.0, PADDLE_LIFT)

        px, py = table_point(paddle_lane, HIT_LINE)
        draw_paddle(img, (px, py - PADDLE_LIFT), 60, MY_PADDLE, 0.8)

        draw_scoreboard(img, self.me, self.them, self.player_serving(), self.streak, self.best_streak)
        if isinstance(self.paddle, MotorPaddle):
            draw_gyro_meter(img, self.paddle, 20, 210, 250)

        if self.state == "start":
            rounded_panel(img, 340, 230, 600, 180, WHITE, 0.92, 30)
            text(img, "Ping Pong", SCREEN_W // 2, 300, 2.0, BLUE, 4, center=True)
            text(img, f"First to {WIN_SCORE} wins", SCREEN_W // 2, 345, 0.9, center=True)
            text(img, "Swing your paddle to start!", SCREEN_W // 2, 390, 0.9, MY_PADDLE, 2, center=True)
        elif self.state == "serve" and self.player_serving():
            banner(img, "Your serve", BLUE, "Swing to serve")
        elif self.state == "point":
            banner(img, self.message[0], self.message[1], f"You {self.me}  -  {self.them} CPU")
        elif self.state == "game_over":
            banner(img, self.message[0], self.message[1],
                   f"Final score {self.me} - {self.them}   Best streak {self.best_streak}")


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera", type=int, default=CAMERA_INDEX)
    parser.add_argument("--no-motor", action="store_true", help="play without LEGO: SPACE = swing")
    parser.add_argument("--calibrate", action="store_true", help="redo the swing calibration")
    parser.add_argument("--make-tag", action="store_true", help="save a printable AprilTag and exit")
    args = parser.parse_args()
    if args.make_tag:
        make_tag()
        return

    if args.no_motor:
        paddle = KeyboardPaddle()
    else:
        try:
            paddle = MotorPaddle()
        except Exception as e:
            sys.exit(f"Couldn't connect to the Double Motor ({e}).\n"
                     "Turn it on and put it near the computer, or play with --no-motor.")
    calibration = None
    if not args.no_motor:
        paddle.threshold = None if args.calibrate else load_calibration()
        if paddle.threshold is None:
            calibration = Calibration(paddle)
        else:
            print(f"Swing threshold {paddle.threshold:.0f} (from {CALIBRATION_FILE.name}; --calibrate to redo)")

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        paddle.close()
        sys.exit(f"Could not open camera {args.camera}")
    pose = create_pose()
    background = make_background()
    game = Game(paddle)
    start = last = time.time()
    arm = ArmTracker()

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Camera frame failed; stopping.")
                break
            now = time.time()
            dt, last = now - last, now

            body = find_body(pose, frame, int((now - start) * 1000))
            arm.locked = calibration is None and game.state != "start"
            paddle_lane = arm.update(body)
            swinging = paddle.threshold is not None and now - paddle.swing_time < 0.15

            screen = background.copy()
            if calibration is not None:
                draw_table(screen)
                if calibration.update():
                    calibration = None
                    game.set_state("start")
                else:
                    calibration.draw(screen)
            else:
                game.update(paddle_lane, dt)
                game.draw(screen, paddle_lane)
            draw_pip(screen, frame, body, arm, swinging)

            cv2.imshow(WINDOW_NAME, screen)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord(" ") and isinstance(paddle, KeyboardPaddle):
                paddle.swing()
    finally:
        pose.close()
        cap.release()
        cv2.destroyAllWindows()
        paddle.close()


if __name__ == "__main__":
    main()
