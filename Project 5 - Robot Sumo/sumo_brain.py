"""Robot Sumo - the laptop "brain": a behavior tree with sensor fusion and shared control.

Run (from the ME193 folder):
    le-venv/bin/python "Project 5 - Robot Sumo/sumo_brain.py"              # the real match
    le-venv/bin/python "Project 5 - Robot Sumo/sumo_brain.py" --readings   # print color sensor readings (for tuning)
    le-venv/bin/python "Project 5 - Robot Sumo/sumo_brain.py" --imu        # print accelerometer readings (for tuning)
    le-venv/bin/python "Project 5 - Robot Sumo/sumo_brain.py" --no-lego    # test the tree without the LEGO parts
    (Windows: use "python" instead of "le-venv/bin/python". Add --speed 0.5 for a gentler run.)

Three layers of communication:
  1. Autonomous sensors:
     - LEGO color sensor on the BACK, pointing down: is the back of the robot on the blue tape?
     - Double Motor's IMU: a sudden backward jolt we didn't cause = pushed from the front;
       tilted far from how it sat at startup = tipped over.
  2. The UNO Q camera app (ArduinoApps "sumo-vision") + YOU: it finds the opponent and asks
     you before anything happens. Its questions show up in THIS terminal (with a beep);
     answer here, and the answer goes back to the app over MQTT (any network works):
         Enter  = approve ("Opponent?" -> charge, "We're OUT?" -> sad song)
         Delete = reject (or, with no question open, stop a charge)
         o      = we're out, right now
     (The app's web page at http://<board-ip>:7000 works too, if the laptop can reach the
     board.) The laptop never charges on its own; it waits for the app's messages on
     ME193/Rogers/MK:
         {"charge": true}   -> charge straight ahead for CHARGE_TIME seconds (each new
                               message restarts the timer, so repeating it keeps charging)
         {"charge": false}  -> stop charging now
         {"out": true}      -> you approved "we're out": stop and play the sad song
         {"out": false}     -> you rejected "we're out": false alarm, keep wrestling
     From the camera's raw numbers the laptop only uses tape_ahead (for "out", below).
  3. LEGO controller (teleop): an ABSOLUTE OVERRIDE. Touch either lever and you drive
     (tank style: left lever = left wheels, right lever = right wheels), ignoring everything
     else. Let go of both and the behavior tree takes back over.

Sensor fusion - are we out? The floor is the same inside and outside the ring, so only the
tape tells us. When we're shoved out backward, the BACK color sensor crosses the tape first
(sees blue, then floor again). If the FRONT camera THEN sees the tape in front of us, the
tape is in front AND behind us crossed: the whole robot is past it. The robot stops on its
own and sends {"out_detected": true}; the app asks you, and Approve plays the sad song.

The policy is a behavior tree, checked top to bottom 20 times a second:
    Selector
    |- the app said we're out?                 -> stop, play the sad song, quit
    |- back crossed tape + camera sees tape?   -> OUT: stop on its own, ask the app for the song
    |- a controller lever moved?               -> the driver drives (absolute override)
    |- tipped over (IMU)?                      -> stop until it's back on its wheels
    |- back on tape, rolling FORWARD at a line   -> we drove over it: back off (reverse). Needs the wheels rolling forward
    |  the camera had seen ahead?                  AND tape seen ahead recently (OVERSHOOT_BACKOFF)
    |- back on tape?                           -> full power forward off the edge, then turn
    |- in the middle of a maneuver?            -> keep doing it
    |- pushed from the front (IMU)?            -> full power forward, counteracting it
    |- the app said charge?                    -> charge straight ahead
    `- otherwise                               -> sit still

Edge reflexes (the back sensor touches the tape):
  - Normally we are being pushed out backward: full power forward, then turn back into the ring.
  - If the wheels had been rolling the robot FORWARD toward a line the camera had seen, we drove over it (for example
    charging a box out of the ring). Forward would take us further out, so we back off instead. The decision is made once,
    when the back first touches the tape, and a second contact within BACK_OFF_RECOVER seconds (the reverse can cross the
    line again) keeps the back-off going.
  - Either way the charge you approved is cancelled (the escape moves and turns the robot, so it is stale) and you
    approve a new one (EDGE_CANCELS_CHARGE).
The settings that depend on how this robot is built (IMU_FORWARD_SIGN, WHEEL_FORWARD_SIGNS) were measured with imu_check.py
and wheel_check.py: re-run them if the Double Motor is ever mounted differently.
"""

import argparse
import collections
import json
import math
import os
import queue
import struct
import subprocess
import sys
import tempfile
import threading
import time
import wave
from pathlib import Path

ME193 = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ME193))             # lelib.py
sys.path.insert(0, str(ME193 / "MQTT"))    # mqttlib.py

# ---- Hardware (TODO: fill in the serial numbers printed on your connection cards) ----
MOTOR_CARD = None        # None = first Double Motor found
COLOR_CARD = None
CONTROLLER_CARD = None
USE_CONTROLLER = True    # False = run without the controller
LEVER_DEADBAND = 0.05    # levers within this of center count as "let go"
MOTOR_BACKWARD = True    # the Double Motor faces the back of the robot (flips forward AND left/right)

# ---- MQTT (the UNO Q app) ----
MQTT_TOPIC = "ME193/Rogers/MK"
CHARGE_TIME = 1.5        # one {"charge": true} charges this long (s); another one restarts it

# ---- Color sensor (TODO: tune with --readings on the real ring) ----
TAPE_COLOR = "Blue"

# ---- Out (sensor fusion: back color sensor + front camera) ----
OUT_WINDOW = 3.0         # the back must have seen blue within this many seconds...
TAPE_AHEAD = 0.08        # ...and the camera must see at least this much tape ahead...
OUT_CONFIRM = 0.3        # ...for this long (s) in a row = we're out
OUT_IGNORE = 3.0         # after you reject "out", don't stop for it again for this long (s)
VISION_TIMEOUT = 0.5     # camera numbers older than this (s) are ignored

# ---- IMU (measured on this robot with imu_check.py; re-run it if the Double Motor is ever mounted differently) ----
IMU_AXIS = "accelerometerX"   # the Double Motor accelerometer axis along the driving direction
IMU_FORWARD_SIGN = -1    # +1 if shoving the robot FORWARD by hand makes that axis go up, else -1 (it goes DOWN here: the motor faces backward)
IMU_INTERVAL_MS = 20     # how often the Double Motor sends IMU readings
IMU_SMOOTH = 0.5         # 0 = raw, closer to 1 = smoother (but slower to react)
PUSH_ACCEL = 300         # a backward jolt bigger than this = pushed from the front
PUSH_HOLD = 0.8          # keep counteracting this long (s) after the jolt
BRAKE_QUIET = 0.3        # ignore jolts this long (s) after we slow down on purpose
TIP_ANGLE = 45           # tilted more than this many degrees from how it sat at startup = tipped over
TIP_SMOOTH = 0.8         # heavier smoothing so driving jolts don't look like tilting

# ---- Driving (Double Motor speeds are -100..100) ----
SPEED_SCALE = 0.7       # every speed below is multiplied by this (1.0 = full speed); also --speed
LOOP_HZ = 20
CHARGE_SPEED = 80
PUSH_SPEED = 100
RESEND_EVERY = 0.5       # resend the same speed this often (s) in case a Bluetooth command was lost
# Back on tape: full power forward, then turn right. (left, right, seconds) steps.
BACK_EDGE_MOVE = [(100, 100, 0.6), (50, -50, 0.4)]
# Back on tape while the robot was rolling FORWARD toward a line the camera had seen (it drove over the line, e.g. charging): "forward"
# would take it further out, so it backs off instead.
# ON: it passed both real-robot tests (charge a box over a taped line -> BACK OFF; pull the CHARGING robot backward over a line behind it ->
# forward escape). A first version that used only the wheel counters FAILED the second one: the wheel counters show how the wheels turn, not
# how the robot moves over the floor, and with the tyres slipping those differ. So it also needs the camera to have seen the line AHEAD
# (OVERSHOOT_NEEDS_LINE). Re-run both tests if you change speed a lot, and set this to False to get the original behavior back.
# The real fix for driving yourself out is a colour sensor at the FRONT.
OVERSHOOT_BACKOFF = True    # False = always use BACK_EDGE_MOVE (the original behavior)
OVERSHOOT_WINDOW = 1.0      # s: how far back to look at the wheels when the back first reaches the tape
OVERSHOOT_NOW = 0.25        # s: ... and the robot must not be rolling backward in this last bit
OVERSHOOT_TRAVEL = 90       # wheel degrees of net forward rolling in the window (about a quarter turn of the wheels)
# The two wheel counters can count in OPPOSITE directions when the robot drives straight (the motors are mirrored), so each wheel has its own
# sign: +1 if driving the robot FORWARD makes that wheel's position go UP, else -1. If both were wrong the readings would cancel and the
# robot would simply keep the old forward escape. wheel_check.py works the signs out and prints this line for you.
WHEEL_FORWARD_SIGNS = (1, -1)    # measured on this robot with wheel_check.py: driving forward makes wheel 0 count UP and wheel 1 count DOWN
BACK_OFF_MOVE = [(-100, -100, 0.4)]
# After ANY back-edge event the charge you approved is stale: the escape moved and turned the robot, so "charge straight ahead" now points
# somewhere else, perhaps back at the line (this happened on the real robot). So the charge is cancelled, like pressing Delete, and you approve
# a new one when you are ready. A "charge" message already on its way is ignored for EDGE_NO_CHARGE seconds.
EDGE_CANCELS_CHARGE = True  # False = the old behavior: the approved charge resumes after the escape
# The command takes a moment to reach the wheels, so the robot rolls on forward after a back-off is chosen and its back sensor can slide PAST
# the tape onto the floor outside. The reverse then carries the sensor back over the tape from the OUTSIDE, which looks like "pushed out" and
# would run the forward escape, driving the robot out again (seen on the real robot: the robot kept driving forward after backing off). So for
# this long after a back-off, a tape contact keeps the back-off going. (The cost: an opponent pushing us over a line within this time is
# also answered with a reverse, so keep it short.)
BACK_OFF_RECOVER = 1.0      # s
EDGE_NO_CHARGE = 2.0        # s
# The wheel counters follow the WHEELS, not the floor: a robot dragged backward over the line while its wheels spin forward still reads "rolling
# forward". What tells the two cases apart is the camera: a line we drove OVER was seen AHEAD just before; a line we are pushed back over was behind us.
OVERSHOOT_NEEDS_LINE = True # False = skip this camera check (then a robot pushed back while charging is wrongly told to back off)
TAPE_SEEN = 0.04            # the camera's tape_ahead above this counts as "I can see the line ahead"
LINE_AHEAD_WINDOW = 3.0     # s: the line must have been seen ahead within this long before the back reached the tape

SUCCESS, FAILURE, RUNNING = "SUCCESS", "FAILURE", "RUNNING"


# ---------------------------------------------------------------------------
# Behavior tree nodes
# ---------------------------------------------------------------------------

class Condition:
    def __init__(self, name, check):
        self.name, self.check = name, check

    def tick(self, bb):
        return SUCCESS if self.check(bb) else FAILURE


class Action:
    """Leaf that sets the motor speeds. It's RUNNING for as long as it's chosen."""
    def __init__(self, name, act):
        self.name, self.act = name, act

    def tick(self, bb):
        self.act(bb)
        bb.active = self.name
        return RUNNING


class Sequence:
    """Runs children in order; stops at the first one that doesn't succeed."""
    def __init__(self, *children):
        self.children = children

    def tick(self, bb):
        for child in self.children:
            status = child.tick(bb)
            if status != SUCCESS:
                return status
        return SUCCESS


class Selector:
    """Tries children in priority order; the first one that doesn't fail wins."""
    def __init__(self, *children):
        self.children = children

    def tick(self, bb):
        for child in self.children:
            status = child.tick(bb)
            if status != FAILURE:
                return status
        return FAILURE


# ---------------------------------------------------------------------------
# Blackboard: everything the tree knows this tick
# ---------------------------------------------------------------------------

class Blackboard:
    def __init__(self):
        # sensors
        self.floor = "floor"         # back color sensor: "floor" or "tape"
        self.pushed_front = False    # IMU
        self.tipped = False
        self.lever_left = 0.0        # controller, -1 .. +1
        self.lever_right = 0.0
        self.charge_until = 0.0      # the app said charge: charging until this time
        self.back_saw_tape_at = None # last time the back sensor saw blue
        self.edge_overshoot = False  # the back just reached the tape while the wheels were rolling forward toward a line we had seen
        self.backoff_until = 0.0     # until this time a tape contact keeps the back-off going (the reverse can cross the line again)
        self.tape_ahead_seen_at = None  # last time the camera saw tape ahead
        self.charge_block_until = 0.0   # after a back-off: ignore "charge" until this time
        self.tape_ahead = 0.0        # camera: how much tape is right in front
        self.out_since = None        # back crossed the tape + camera sees it, since when
        self.out_detected = False    # fusion says we're out: stopped, waiting for the app
        self.out_ignore_until = 0.0  # you rejected "out": ignore it until this time
        self.out = False             # the app said we're out (play the song)
        self.lost = False            # sad song played
        # outputs / memory
        self.drive = (0, 0)
        self.maneuver = []           # queued (left, right, until) steps
        self.active = ""


def clip(v):
    return max(-100, min(100, int(v)))


def angle_between(u, v):
    """Angle in degrees between two 3D vectors."""
    dot = sum(a * b for a, b in zip(u, v))
    norm = math.sqrt(sum(a * a for a in u)) * math.sqrt(sum(b * b for b in v))
    if norm == 0:
        return 0.0
    return math.degrees(math.acos(max(-1.0, min(1.0, dot / norm))))


# ---- Conditions ----

def is_out(bb):
    return bb.out

def out_detected(bb):
    return bb.out_detected

def tipped_over(bb):
    return bb.tipped

def driver_override(bb):
    return abs(bb.lever_left) > LEVER_DEADBAND or abs(bb.lever_right) > LEVER_DEADBAND

def maneuvering(bb):
    while bb.maneuver and time.monotonic() > bb.maneuver[0][2]:
        bb.maneuver.pop(0)
    return bool(bb.maneuver)

def back_on_tape(bb):
    return bb.floor == "tape"

def pushed_front(bb):
    return bb.pushed_front

def charge_approved(bb):
    return time.monotonic() < bb.charge_until


# ---- Actions ----

def lose(bb):
    bb.maneuver = []
    bb.drive = (0, 0)
    if not bb.lost:
        bb.lost = True
        print("\nOUT OF THE RING. Stopping.")
        play_sad_song()

def stop_out(bb):
    """Fusion says we're out: stop and wait for the app (Approve = sad song, Reject = resume)."""
    bb.maneuver = []
    bb.charge_until = 0.0
    bb.drive = (0, 0)

def stop_tipped(bb):
    """Stop until the robot is back on its wheels."""
    bb.maneuver = []
    bb.drive = (0, 0)

def driver_drives(bb):
    """Absolute override: the levers go straight to the wheels."""
    bb.maneuver = []
    bb.drive = (clip(bb.lever_left * 100), clip(bb.lever_right * 100))

def continue_maneuver(bb):
    left, right, _ = bb.maneuver[0]
    bb.drive = (left, right)

def start_maneuver(bb, steps):
    """Queue a maneuver of (left, right, seconds) steps."""
    t = time.monotonic()
    bb.maneuver = []
    for left, right, secs in steps:
        t += secs
        bb.maneuver.append((left, right, t))
    continue_maneuver(bb)

def leave_back_edge(bb):
    """The back usually only reaches the tape when we're being pushed: full power forward
    off the tape, then turn back into the ring."""
    start_maneuver(bb, BACK_EDGE_MOVE)

def back_overshoot(bb):
    return bb.floor == "tape" and bb.edge_overshoot

def back_off(bb):
    """The back reached the tape while we were rolling forward: we drove over the line, so going forward
    would take us further out. Reverse back into the ring."""
    start_maneuver(bb, BACK_OFF_MOVE)

def push_back(bb):
    """The IMU felt a push from the front: full power forward, counteracting it."""
    bb.maneuver = []
    bb.drive = (PUSH_SPEED, PUSH_SPEED)

def charge(bb):
    """The app said charge: straight ahead."""
    bb.drive = (CHARGE_SPEED, CHARGE_SPEED)

def wait(bb):
    """Nothing to react to: sit still. The driver searches with the controller if needed."""
    bb.drive = (0, 0)


TREE = Selector(
    Sequence(Condition("out", is_out), Action("OUT - sad song", lose)),
    Sequence(Condition("out detected", out_detected), Action("OUT? stopped, ask app", stop_out)),
    Sequence(Condition("lever moved", driver_override), Action("DRIVER OVERRIDE", driver_drives)),
    Sequence(Condition("tipped over", tipped_over), Action("TIPPED OVER - stop", stop_tipped)),
    Sequence(Condition("back on tape, rolling forward", back_overshoot), Action("back edge: BACK OFF", back_off)),
    Sequence(Condition("back on tape", back_on_tape), Action("back edge: forward", leave_back_edge)),
    Sequence(Condition("maneuvering", maneuvering), Action("edge maneuver", continue_maneuver)),
    Sequence(Condition("pushed from front", pushed_front), Action("IMU PUSH! pushing back", push_back)),
    Sequence(Condition("app said charge", charge_approved), Action("CHARGE (app)", charge)),
    Action("waiting (still)", wait),
)


# ---------------------------------------------------------------------------
# Keyboard
# ---------------------------------------------------------------------------

class Keyboard:
    """Reads single keys from this terminal without waiting for a whole line."""

    def __init__(self):
        self.keys = queue.Queue()
        self._old = None
        if not sys.stdin.isatty():
            print("(no terminal keyboard: keys won't work)")
            return
        if sys.platform == "win32":               # Windows has no termios: read keys with msvcrt instead
            threading.Thread(target=self._read_windows, daemon=True).start()
            return
        import termios
        import tty
        self._fd = sys.stdin.fileno()
        self._old = termios.tcgetattr(self._fd)
        tty.setcbreak(self._fd)               # keys arrive one at a time; Ctrl+C still works
        threading.Thread(target=self._read, daemon=True).start()

    def _read(self):
        while True:
            data = os.read(self._fd, 16)
            if b"\n" in data or b"\r" in data:
                self.keys.put("enter")
            elif b"\x7f" in data or b"\x08" in data or b"\x1b[3~" in data:   # Delete / fn+Delete
                self.keys.put("delete")
            elif b" " in data:
                self.keys.put("space")
            elif b"o" in data or b"O" in data:
                self.keys.put("o")

    def _read_windows(self):
        import msvcrt
        while True:
            ch = msvcrt.getwch()                  # waits for one key, no Enter needed
            if ch in ("\x00", "\xe0"):            # a special key arrives as two characters
                if msvcrt.getwch() == "S":        # ... and 'S' second means the Delete key
                    self.keys.put("delete")
            elif ch in ("\r", "\n"):
                self.keys.put("enter")
            elif ch in ("\x08", "\x7f"):          # Backspace works as Delete too
                self.keys.put("delete")
            elif ch == " ":
                self.keys.put("space")
            elif ch in ("o", "O"):
                self.keys.put("o")
            elif ch == "\x03":                    # Ctrl+C: getwch() swallows it, so pass it on to the main thread
                import _thread
                _thread.interrupt_main()
                return

    def pressed(self):
        out = []
        while not self.keys.empty():
            out.append(self.keys.get())
        return out

    def restore(self):
        if self._old is not None:
            import termios
            termios.tcsetattr(self._fd, termios.TCSADRAIN, self._old)


# ---------------------------------------------------------------------------
# Sensors and motors
# ---------------------------------------------------------------------------

class AppLink:
    """Charge and "out" commands from the UNO Q app over MQTT, plus the camera's tape_ahead."""

    def __init__(self):
        from mqttlib import MQTTClient
        self.charge_until = 0.0
        self.out = False
        self.out_rejected = False             # you rejected "out" (read and cleared by sense())
        self.ask = None                       # the app's open question: "opponent", "out" or None
        self.answered_at = 0.0                # when you last answered (the app needs a moment)
        self._tape_ahead, self._tape_at = 0.0, 0.0
        client = MQTTClient()
        client.connect()                      # some versions of mqttlib return the client from connect() and some return None: don't depend on it
        self.client = client
        self.client.subscribe(MQTT_TOPIC, self._on_message)
        print(f"MQTT: waiting for the app's charge messages on {MQTT_TOPIC}")

    def _on_message(self, topic, payload):
        try:
            msg = json.loads(payload)
        except ValueError:
            return
        if not isinstance(msg, dict):
            return
        if "tape_ahead" in msg:
            self._tape_ahead, self._tape_at = msg["tape_ahead"], time.monotonic()
            if time.monotonic() - self.answered_at > 0.5:   # skip the stale question
                self.ask = msg.get("ask")
        if msg.get("out") is True:
            self.out = True
            print(">> app: WE'RE OUT")
        elif msg.get("out") is False:
            self.out_rejected = True
            print(">> app: not out, keep going")
        if "charge" not in msg:
            return                            # camera numbers etc.: not a charge command
        if msg["charge"]:
            if time.monotonic() > self.charge_until:   # print only when a charge starts
                print(">> app: CHARGE")
            self.charge_until = time.monotonic() + CHARGE_TIME
        else:
            self.charge_until = 0.0
            print(">> app: stop charging")

    def tape_ahead(self):
        if time.monotonic() - self._tape_at > VISION_TIMEOUT:
            return 0.0                        # camera silent
        return self._tape_ahead

    def send(self, msg):
        self.client.publish(MQTT_TOPIC, json.dumps(msg))


QUESTIONS = {
    "opponent": "Camera sees an OPPONENT. Enter = charge, Delete = not the opponent",
    "out": "We're OUT? Enter = sad song, Delete = keep wrestling",
}


def answer_keys(app, keyboard, last_ask):
    """Show the app's question here and send your Enter / Delete back to it.
    Returns the question that is open now (to spot new ones)."""
    ask = app.ask
    if ask != last_ask and ask in QUESTIONS:
        print(f"\a?? {QUESTIONS[ask]}", flush=True)
    for key in keyboard.pressed():
        if key == "o":
            app.send({"answer": "out_now"})
        elif key in ("enter", "delete") and ask:
            app.send({"answer": f"{'approve' if key == 'enter' else 'reject'}_{ask}"})
            app.ask, app.answered_at = None, time.monotonic()   # don't double-send or re-ask
        elif key == "delete" and time.monotonic() < app.charge_until:
            app.send({"answer": "stop_charge"})
    return app.ask


def set_speed_scale(scale):
    global SPEED_SCALE
    SPEED_SCALE = max(0.0, min(1.0, scale))
    print(f"Speed: {SPEED_SCALE:.0%} of full")


def motor_speeds(left, right):
    """Robot (left, right) wheel speeds -> the Double Motor's own (left, right).
    If the motor faces backward, its left wheel is the robot's right wheel and
    its forward is the robot's backward. SPEED_SCALE slows everything down for testing."""
    left, right = round(left * SPEED_SCALE), round(right * SPEED_SCALE)
    if MOTOR_BACKWARD:
        return -right, -left
    return left, right


class LegoRobot:
    def __init__(self):
        import legoeducation as le
        from lelib import doubleMotor, colorSensor, controller
        print("Connecting to the Double Motor...")
        self.motor = doubleMotor()
        self.motor.connect(card_serial=MOTOR_CARD, device_notification_delay=IMU_INTERVAL_MS)
        print("Connecting to the color sensor...")
        self.color = colorSensor()
        self.color.connect(card_serial=COLOR_CARD)
        self.controller = None
        if USE_CONTROLLER:
            print("Connecting to the controller...")
            self.controller = controller()
            self.controller.connect(card_serial=CONTROLLER_CARD)
        else:
            print("No controller (USE_CONTROLLER = False).")
        print("All LEGO parts connected.")
        self._last = None
        self._sent_at = 0.0
        # IMU push detection runs in its own thread, faster than the tree
        self.accel = 0.0             # smoothed forward acceleration (minus the resting value)
        self.imu_ready = False
        self.tilt = 0.0              # degrees away from how the robot sat at startup
        self.pushed_at = None
        self.quiet_until = 0.0
        self._wheel_hist = collections.deque(maxlen=300)   # (time, left wheel position, right wheel position), about 6 s
        self._running = True
        threading.Thread(target=self._watch_imu, daemon=True).start()

    def _forward_accel(self):
        return IMU_FORWARD_SIGN * float(getattr(self.motor.imu_device, IMU_AXIS))

    def _gravity(self):
        imu = self.motor.imu_device
        return [float(imu.accelerometerX), float(imu.accelerometerY), float(imu.accelerometerZ)]

    def _watch_imu(self):
        """Learn the resting values (robot sitting still at startup), then flag any backward
        jolt bigger than PUSH_ACCEL that didn't come from us slowing down, and tipping over
        (gravity pointing more than TIP_ANGLE away from where it pointed at rest)."""
        dt = IMU_INTERVAL_MS / 1000
        rest, rest_g = [], []
        while len(rest) < 25:
            a, g = self._forward_accel(), self._gravity()
            if not math.isnan(a) and not any(map(math.isnan, g)):
                rest.append(a)
                rest_g.append(g)
            time.sleep(dt)
        baseline = sum(rest) / len(rest)
        up = [sum(axis) / len(rest_g) for axis in zip(*rest_g)]
        g_now = list(up)
        self.imu_ready = True
        while self._running:
            a = self._forward_accel() - baseline
            if not math.isnan(a):
                self.accel = IMU_SMOOTH * self.accel + (1 - IMU_SMOOTH) * a
                if self.accel < -PUSH_ACCEL and time.monotonic() > self.quiet_until:
                    self.pushed_at = time.monotonic()
            g = self._gravity()
            if not any(map(math.isnan, g)):
                g_now = [TIP_SMOOTH * s + (1 - TIP_SMOOTH) * v for s, v in zip(g_now, g)]
                self.tilt = angle_between(up, g_now)
            self._sample_wheels()
            time.sleep(dt)

    def _sample_wheels(self):
        try:
            left, right = float(self.motor.motor[0].position), float(self.motor.motor[1].position)   # fetched fresh: the library swaps the objects
        except (AttributeError, IndexError, TypeError, ValueError):
            return
        if not (math.isnan(left) or math.isnan(right)):
            self._wheel_hist.append((time.monotonic(), left, right))

    def wheel_travel(self, seconds):
        """How far each wheel's raw position counter changed in the last `seconds`: (wheel 0, wheel 1) in degrees.
        None if there are no wheel readings yet."""
        now = time.monotonic()
        recent = [h for h in list(self._wheel_hist) if now - h[0] <= seconds]
        if len(recent) < 2:
            return None
        return recent[-1][1] - recent[0][1], recent[-1][2] - recent[0][2]

    def forward_travel(self, seconds):
        """How far the wheels rolled the robot FORWARD (degrees, averaged over both wheels) in the last `seconds`.
        None if there are no wheel readings yet."""
        moved = self.wheel_travel(seconds)
        if moved is None:
            return None
        return (moved[0] * WHEEL_FORWARD_SIGNS[0] + moved[1] * WHEEL_FORWARD_SIGNS[1]) / 2.0

    def tipped_over(self):
        return self.tilt > TIP_ANGLE

    def pushed_from_front(self):
        return self.pushed_at is not None and time.monotonic() - self.pushed_at < PUSH_HOLD

    def floor_reading(self):
        return self.color.detect_color(), self.color.reflection()

    def levers(self):
        if self.controller is None:
            return 0.0, 0.0
        return self.controller.left_position() / 100, self.controller.right_position() / 100

    def drive(self, left, right):
        now = time.monotonic()
        changed = (left, right) != self._last
        if changed or now - self._sent_at > RESEND_EVERY:   # changes, plus a periodic resend
            before = sum(self._last) / 2 if self._last else 0
            if (left + right) / 2 < before - 10:  # slowing down makes a backward jolt too
                self.quiet_until = now + BRAKE_QUIET
            self.motor.movement_move_tank(*motor_speeds(left, right), blocking=False)
            self._last, self._sent_at = (left, right), now

    def stop(self):
        self._running = False
        self.motor.stop()


class NoLego:
    """Stand-in for --no-lego: always on the ring floor, levers centered."""
    def floor_reading(self):  return "No color", 50
    def levers(self):         return 0.0, 0.0
    def pushed_from_front(self): return False
    def tipped_over(self):    return False
    def forward_travel(self, seconds): return None
    def wheel_travel(self, seconds):   return None
    def drive(self, left, right): pass
    def stop(self):           pass


def rolling_forward(robot):
    """True if the wheels rolled the robot forward over the last OVERSHOOT_WINDOW seconds and are not rolling it backward now."""
    if not OVERSHOOT_BACKOFF:
        return False
    recent, latest = robot.forward_travel(OVERSHOOT_WINDOW), robot.forward_travel(OVERSHOOT_NOW)
    return recent is not None and recent >= OVERSHOOT_TRAVEL and (latest is None or latest >= 0)


def line_was_ahead(bb, now):
    """True if the camera saw tape ahead recently: the line we may have just driven over (not one behind us that we are pushed over)."""
    if not OVERSHOOT_NEEDS_LINE:
        return True
    return bb.tape_ahead_seen_at is not None and now - bb.tape_ahead_seen_at <= LINE_AHEAD_WINDOW


def sense(bb, robot, app):
    """Read every sensor (and the app's commands) into the blackboard."""
    now = time.monotonic()
    bb.tape_ahead = app.tape_ahead()
    if bb.tape_ahead > TAPE_SEEN:
        bb.tape_ahead_seen_at = now
    was_tape = bb.floor == "tape"
    color, _ = robot.floor_reading()
    bb.floor = "tape" if color == TAPE_COLOR else "floor"
    if bb.floor == "tape":
        bb.back_saw_tape_at = now
        if not was_tape:                       # the back JUST reached the tape: which way are the wheels rolling the robot?
            recovering = now < bb.backoff_until                    # we are in the middle of backing off
            bb.edge_overshoot = recovering or (rolling_forward(robot) and line_was_ahead(bb, now))
            charging = app.charge_until > now
            if bb.edge_overshoot:
                bb.backoff_until = now + BACK_OFF_RECOVER
                bb.out_ignore_until = max(bb.out_ignore_until, now + OUT_WINDOW)     # we are recovering: don't ask "out?" about it
                if recovering:
                    print("\nBack sensor on the tape again while backing off (the reverse crossed the line): keep backing off.")
                else:
                    print("\nBack reached the tape while rolling FORWARD toward a line we had seen ahead (we drove over it): backing off, charge cancelled.")
            elif EDGE_CANCELS_CHARGE and charging:
                print("\nBack reached the tape: charge cancelled (the escape moves and turns the robot). Approve a new one when you are ready.")
            if bb.edge_overshoot or EDGE_CANCELS_CHARGE:
                app.charge_until = 0.0                                              # cancel the charge, like pressing Delete,
                app.send({"answer": "stop_charge"})                                 # or the app would keep sending it and it would resume
                bb.charge_block_until = now + EDGE_NO_CHARGE                        # and ignore any "charge" already on its way
    if now < bb.charge_block_until:                                                 # just backed off: a late "charge" message is ignored
        app.charge_until = 0.0
    bb.charge_until = app.charge_until
    bb.out = app.out

    # Sensor fusion: the back crossed the tape (saw blue, now past it) AND the front camera
    # sees tape, for OUT_CONFIRM seconds in a row -> we're out. Stop and ask the app.
    if app.out_rejected:
        app.out_rejected = False
        bb.out_detected, bb.out_ignore_until = False, now + OUT_IGNORE
    crossed = (bb.back_saw_tape_at is not None and bb.floor != "tape"
               and now - bb.back_saw_tape_at < OUT_WINDOW)
    if crossed and bb.tape_ahead > TAPE_AHEAD:
        bb.out_since = bb.out_since or now
    else:
        bb.out_since = None
    if (not bb.out_detected and bb.out_since is not None and now - bb.out_since > OUT_CONFIRM
            and now > bb.out_ignore_until):
        bb.out_detected = True
        app.send({"out_detected": True})
        print("\nOUT? Back crossed the tape and the camera sees it ahead. Stopped.")

    bb.pushed_front = robot.pushed_from_front()
    bb.tipped = robot.tipped_over()
    bb.lever_left, bb.lever_right = robot.levers()


# ---------------------------------------------------------------------------
# Sad song (the LEGO parts can only beep, so it plays on the laptop)
# ---------------------------------------------------------------------------

def play_wav(path):
    """Play a .wav without waiting for it: winsound on Windows, afplay on macOS, paplay/aplay on Linux."""
    path = str(path)
    if sys.platform == "win32":
        import winsound
        winsound.PlaySound(path, winsound.SND_FILENAME | winsound.SND_ASYNC)
        return
    for player in ([["afplay"]] if sys.platform == "darwin" else [["paplay"], ["aplay", "-q"]]):
        try:
            subprocess.Popen(player + [path])
            return
        except FileNotFoundError:
            continue
    print("(no audio player found on this computer - imagine a sad trombone)")


def play_sad_song():
    """Sad trombone: wah wah wah waaaah."""
    rate = 22050
    notes = [(392, 0.45), (370, 0.45), (349, 0.45), (330, 1.4)]
    frames = bytearray()
    for i, (freq, secs) in enumerate(notes):
        for n in range(int(rate * secs)):
            t = n / rate
            wobble = 6 * math.sin(2 * math.pi * 5 * t) if i == len(notes) - 1 else 0
            fade = min(1.0, (secs - t) * 10)
            sample = 0.4 * fade * math.sin(2 * math.pi * (freq + wobble) * t)
            frames += struct.pack("<h", int(sample * 32767))
    path = Path(tempfile.gettempdir()) / "sumo_sad_song.wav"
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(bytes(frames))
    play_wav(path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def print_readings(robot):
    print("Color sensor readings (Ctrl+C to stop). Try: ring floor, blue tape, outside floor.")
    while True:
        color, reflection = robot.floor_reading()
        print(f"\r{color:10s} reflection={reflection}   ", end="", flush=True)
        time.sleep(0.2)


def print_imu(robot):
    print("Keep the robot still while it learns its resting value...")
    while not robot.imu_ready:
        time.sleep(0.05)
    print("Accelerometer readings (Ctrl+C to stop). The line below updates IN PLACE, so watch it while you move the robot.")
    print("  1. Shove the robot FORWARD (toward the camera) and let it stop. The FIRST jump, as you start pushing, is the one to read:")
    print("     the axis that jumps most is IMU_AXIS; if 'forward' goes DOWN at the start, set IMU_FORWARD_SIGN = -1.")
    print("     (The opposite jump when you stop it is normal.)")
    print(f"  2. Shove it BACKWARD, like an opponent hitting its front: 'forward' should go below -{PUSH_ACCEL} (PUSH_ACCEL) and print PUSH!")
    print("  'last 3 s' is the biggest swing down and up in the last 3 seconds, so you can read it after a quick shove.")
    recent = []                                   # (time, forward) for the last 3 seconds
    while True:
        imu = robot.motor.imu_device              # fetched EVERY time: the LEGO library swaps in a brand-new object for each reading
        now = time.monotonic()
        recent = [(t, a) for t, a in recent if now - t < 3.0] + [(now, robot.accel)]
        low, high = min(a for _, a in recent), max(a for _, a in recent)
        flag = "PUSH!" if robot.pushed_from_front() else "     "
        print(f"\rX={imu.accelerometerX:7.0f} Y={imu.accelerometerY:7.0f} Z={imu.accelerometerZ:7.0f}"
              f"   forward={robot.accel:+7.0f}  last 3 s: {low:+6.0f} .. {high:+6.0f}  tilt={robot.tilt:3.0f}deg  {flag}", end="", flush=True)
        time.sleep(0.05)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--readings", action="store_true", help="print color sensor readings")
    parser.add_argument("--imu", action="store_true", help="print accelerometer readings")
    parser.add_argument("--no-lego", action="store_true", help="run without the LEGO parts")
    parser.add_argument("--speed", type=float, default=SPEED_SCALE,
                        help=f"fraction of full speed, e.g. 0.3 (default {SPEED_SCALE})")
    args = parser.parse_args()
    set_speed_scale(args.speed)

    robot = NoLego() if args.no_lego else LegoRobot()
    keyboard = None
    try:
        if args.readings:
            print_readings(robot)
        if args.imu:
            print_imu(robot)

        app = AppLink()
        print("Place the robot inside the ring and hold it still...")
        while not getattr(robot, "imu_ready", True):   # IMU learning its resting value
            time.sleep(0.05)
        keyboard = Keyboard()
        print("GO!  Sitting still until the app says charge. Answer its questions here:")
        print("     Enter = yes, Delete = no / stop charge, o = we're out, Ctrl+C = quit")

        bb = Blackboard()
        last_active, last_ask = None, None
        while True:
            last_ask = answer_keys(app, keyboard, last_ask)
            sense(bb, robot, app)
            TREE.tick(bb)
            robot.drive(*bb.drive)
            if bb.active != last_active:    # print only when the decision changes
                print(f"[{bb.active:22s}] floor={bb.floor:5s} tape_ahead={bb.tape_ahead:.2f} "
                      f"levers={bb.lever_left:+.1f}/{bb.lever_right:+.1f}")
                last_active = bb.active
            if bb.lost:
                time.sleep(3)               # let the song finish
                break
            time.sleep(1 / LOOP_HZ)
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        robot.stop()
        if keyboard:
            keyboard.restore()


if __name__ == "__main__":
    main()
