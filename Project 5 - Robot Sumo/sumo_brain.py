"""Robot Sumo - the laptop "brain": a behavior tree with sensor fusion and shared control.

Run (from the ME193 folder):
    le-venv/bin/python "Project 5 - Project 5 - Robot Sumo/sumo_brain.py"              # the real match
    le-venv/bin/python "Project 5 - Project 5 - Robot Sumo/sumo_brain.py" --readings   # print color sensor readings (for tuning)
    le-venv/bin/python "Project 5 - Project 5 - Robot Sumo/sumo_brain.py" --imu        # print accelerometer readings (for tuning)
    le-venv/bin/python "Project 5 - Project 5 - Robot Sumo/sumo_brain.py" --no-lego    # test the tree with only the camera

Inputs:
  - LEGO color sensor on the BACK, pointing down (autonomous): is the back of the robot on
    the blue tape?
  - Logitech webcam on the FRONT, on the UNO Q (autonomous): where the opponent is and
    whether blue tape is coming up. It arrives over MQTT on ME193/Rogers/MK from the
    ArduinoApps "sumo-vision" app.
  - Double Motor's IMU (autonomous): a sudden backward jolt we didn't cause = we just got
    hit or pushed from the front.
  - LEGO controller (teleop): an ABSOLUTE OVERRIDE. Touch either lever and you drive
    (tank style: left lever = left wheels, right lever = right wheels), ignoring every
    sensor. Let go of both and the behavior tree takes back over.

Sensor fusion: the camera watches the front, the color sensor watches the back.
  - Front edge: only the camera sees it. A little tape ahead -> veer away. A lot -> back
    up and turn away from the side the tape is on (tape_x).
  - Back edge: the color sensor. The back usually only reaches the tape when we're being
    pushed, so the answer is always full power forward. With the opponent filling the
    camera, the camera steers that push straight into it. With nothing in front, drive
    forward off the edge and turn back into the ring.
  - Pushed from the front: the IMU feels the jolt even when the camera misses the
    opponent. The answer is full power forward for PUSH_HOLD seconds, steered at the
    opponent if the camera sees it. Braking makes the same jolt, so jolts right after
    we slow down on purpose are ignored.
  - Opponent in contact AND tape ahead -> the OPPONENT is the one at the edge, so in
    attack mode keep pushing instead of backing off.
  - Out: the floor is the same inside and outside the ring, so only the tape tells us.
    When we're shoved out backward, the back sensor crosses the tape first (sees blue,
    then floor again). If the front camera THEN sees the tape in front of us, the whole
    robot is past it: we're out. Without the camera there's no automatic "out".

The policy is a behavior tree, checked top to bottom 20 times a second:
    Selector
    |- a controller lever moved?               -> the driver drives (absolute override)
    |- out of the ring?                        -> stop, play the sad song
    |- back on tape + opponent in contact?     -> being pushed out: full power back at it
    |- back on tape?                           -> full power forward off the edge, then turn
    |- in the middle of a maneuver?            -> keep doing it
    |- pushed from the front (IMU)?            -> full power forward, counteracting it
    |- opponent in contact + attack mode?      -> full-power push (even at the edge)
    |- lots of tape ahead (camera)?            -> back up, turn away from the tape
    |- opponent seen?                          -> charge (camera aim blended with driver bias)
    |- a little tape ahead (camera)?           -> veer away from it
    `- otherwise                               -> search, spinning toward the driver's side
"""

import argparse
import json
import math
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
USE_CONTROLLER = False   # False = run without the controller (fully autonomous)
LEVER_DEADBAND = 0.05    # levers within this of center count as "let go"
MOTOR_BACKWARD = True    # the Double Motor faces the back of the robot (flips forward AND left/right)

# ---- MQTT ----
MQTT_TOPIC = "ME193/Rogers/MK"
VISION_TIMEOUT = 0.5     # camera data older than this (s) is ignored

# ---- Color sensor (TODO: tune with --readings on the real ring) ----
TAPE_COLOR = "Blue"
OUT_WINDOW = 3.0         # the back must have seen blue within this many seconds...
OUT_CONFIRM = 0.3        # ...and the camera must see tape ahead this long = we lost

# ---- Camera thresholds ----
CONTACT_SIZE = 0.25      # opponent covers this much of the ground area = touching us
TAPE_AHEAD = 0.08        # this much blue in the bottom strip = edge coming up (veer)
TAPE_CLOSE = 0.25        # this much = right at the edge (back up)

# ---- IMU push detection (TODO: tune with --imu) ----
IMU_AXIS = "accelerometerX"   # the Double Motor accelerometer axis along the driving direction
IMU_FORWARD_SIGN = 1     # +1 if shoving the robot FORWARD by hand makes that axis go up, else -1
IMU_INTERVAL_MS = 20     # how often the Double Motor sends IMU readings
IMU_SMOOTH = 0.5         # 0 = raw, closer to 1 = smoother (but slower to react)
PUSH_ACCEL = 300         # a backward jolt bigger than this = pushed from the front
PUSH_HOLD = 0.8          # keep counteracting this long (s) after the jolt
BRAKE_QUIET = 0.3        # ignore jolts this long (s) after we slow down on purpose

# ---- Driving (Double Motor speeds are -100..100) ----
SPEED_SCALE = 0.7       # every speed below is multiplied by this (1.0 = full speed); also --speed
LOOP_HZ = 20
CHARGE_SPEED = 80
PUSH_SPEED = 100
TURN_GAIN = 40           # how hard to steer toward the opponent
SEARCH_TURN = 35
VEER_SPEED, VEER_TURN = 30, 40
# Maneuvers: (left, right, seconds) steps. A step with left != right is a turn; its
# direction is chosen when the maneuver starts.
FRONT_EDGE_MOVE = [(-60, -60, 0.5), (50, -50, 0.4)]   # tape ahead: back up, then turn
BACK_EDGE_MOVE = [(100, 100, 0.6), (50, -50, 0.4)]    # back on tape: full power forward, then turn
DRIVER_SHARE = 0.3       # how much the driver's side bias counts vs. the camera's aim

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
        self.back_saw_tape_at = None # last time the back sensor saw blue
        self.out_since = None        # back crossed the tape + camera sees it, since when
        self.opp_seen = False        # camera
        self.opp_x = 0.0
        self.opp_size = 0.0
        self.tape_ahead = 0.0
        self.tape_x = 0.0
        self.pushed_front = False    # IMU
        self.lever_left = 0.0        # controller, -1 .. +1
        self.lever_right = 0.0
        # strategy: fixed for now (the controller is a full override instead)
        self.aggression = 1.0        # -1 (defend) .. +1 (attack)
        self.side_bias = 0.0         # -1 (favor left) .. +1 (favor right)
        # outputs / memory
        self.drive = (0, 0)
        self.maneuver = []           # queued (left, right, until) steps
        self.lost = False
        self.active = ""


def clip(v):
    return max(-100, min(100, int(v)))


# ---- Conditions ----

def is_out(bb):
    """The back crossed the blue tape AND the front camera now sees it (see sense())."""
    now = time.monotonic()
    past_tape = bb.out_since is not None and now - bb.out_since > OUT_CONFIRM
    return bb.lost or past_tape

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

def in_contact(bb):
    return bb.opp_seen and bb.opp_size > CONTACT_SIZE

def attack_mode(bb):
    return bb.aggression >= 0

def opponent_seen(bb):
    return bb.opp_seen

def tape_ahead(bb):
    return bb.tape_ahead > TAPE_AHEAD

def tape_close(bb):
    return bb.tape_ahead > TAPE_CLOSE


# ---- Actions ----

def lose(bb):
    bb.drive = (0, 0)
    if not bb.lost:
        bb.lost = True
        print("\nOUT OF THE RING. Stopping.")
        play_sad_song()

def driver_drives(bb):
    """Absolute override: the levers go straight to the wheels."""
    bb.maneuver = []
    bb.drive = (clip(bb.lever_left * 100), clip(bb.lever_right * 100))

def continue_maneuver(bb):
    left, right, _ = bb.maneuver[0]
    bb.drive = (left, right)

def push(bb):
    bb.drive = (PUSH_SPEED, PUSH_SPEED)

def start_maneuver(bb, steps, turn):
    """Queue a maneuver. turn = +1 makes its turning step go right, -1 left."""
    t = time.monotonic()
    bb.maneuver = []
    for left, right, secs in steps:
        if left != right:                 # the turning step
            left, right = turn * abs(left), -turn * abs(right)
        t += secs
        bb.maneuver.append((left, right, t))
    continue_maneuver(bb)

def driver_side(bb):
    return -1 if bb.side_bias < 0 else 1

def back_off_front_edge(bb):
    """Back up, then turn AWAY from the side the camera saw the tape on.
    (If the back reaches the tape while backing up, the back-edge branch takes over.)"""
    start_maneuver(bb, FRONT_EDGE_MOVE, -1 if bb.tape_x > 0 else 1)

def leave_back_edge(bb):
    """Drive forward off the tape behind us. The camera can't see that tape, so the
    turn goes toward the driver's side."""
    start_maneuver(bb, BACK_EDGE_MOVE, driver_side(bb))

def push_back(bb):
    """Being shoved backward (back on the tape, or the IMU felt a push from the front): full
    power forward, steered at the opponent by the camera so we push straight into it
    instead of sliding off to one side."""
    bb.maneuver = []
    aim = bb.opp_x
    bb.drive = (clip(PUSH_SPEED + TURN_GAIN * aim), clip(PUSH_SPEED - TURN_GAIN * aim))

def charge(bb):
    """Shared control: the camera aims, the driver nudges the aim and sets the speed."""
    aim = (1 - DRIVER_SHARE) * bb.opp_x + DRIVER_SHARE * bb.side_bias
    forward = CHARGE_SPEED * (0.5 + 0.5 * bb.aggression)   # defend = turn to face only
    bb.drive = (clip(forward + TURN_GAIN * aim), clip(forward - TURN_GAIN * aim))

def veer_from_tape(bb):
    away = -1 if bb.tape_x > 0 else 1
    bb.drive = (clip(VEER_SPEED + away * VEER_TURN), clip(VEER_SPEED - away * VEER_TURN))

def search(bb):
    direction = -1 if bb.side_bias < 0 else 1
    bb.drive = (direction * SEARCH_TURN, -direction * SEARCH_TURN)


TREE = Selector(
    Sequence(Condition("lever moved", driver_override), Action("DRIVER OVERRIDE", driver_drives)),
    Sequence(Condition("out", is_out), Action("OUT - sad song", lose)),
    Sequence(Condition("back on tape", back_on_tape), Condition("contact", in_contact),
             Action("PUSHED! pushing back", push_back)),
    Sequence(Condition("back on tape", back_on_tape), Action("back edge: forward", leave_back_edge)),
    Sequence(Condition("maneuvering", maneuvering), Action("edge maneuver", continue_maneuver)),
    Sequence(Condition("pushed from front", pushed_front), Action("IMU PUSH! pushing back", push_back)),
    Sequence(Condition("contact", in_contact), Condition("attack mode", attack_mode),
             Action("FULL PUSH", push)),
    Sequence(Condition("tape close", tape_close), Action("front edge: backing up", back_off_front_edge)),
    Sequence(Condition("opponent seen", opponent_seen), Action("charge", charge)),
    Sequence(Condition("tape ahead", tape_ahead), Action("veer from edge", veer_from_tape)),
    Action("search", search),
)


# ---------------------------------------------------------------------------
# Sensors and motors
# ---------------------------------------------------------------------------

class Vision:
    """Latest camera message from the UNO Q (sumo-vision app)."""

    def __init__(self):
        from mqttlib import MQTTClient
        self.latest, self.received = None, 0.0
        self.client = MQTTClient().connect()
        self.client.subscribe(MQTT_TOPIC, self._on_message)
        print(f"MQTT: listening for the camera on {MQTT_TOPIC}")

    def _on_message(self, topic, payload):
        try:
            self.latest, self.received = json.loads(payload), time.monotonic()
        except ValueError:
            pass                              # not a camera message

    def get(self):
        if self.latest is None or time.monotonic() - self.received > VISION_TIMEOUT:
            return None                       # camera silent -> act on the color sensor alone
        return self.latest


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
            print("No controller (USE_CONTROLLER = False): fully autonomous.")
        print("All LEGO parts connected.")
        self._last = None
        # IMU push detection runs in its own thread, faster than the tree
        self.accel = 0.0             # smoothed forward acceleration (minus the resting value)
        self.imu_ready = False
        self.pushed_at = None
        self.quiet_until = 0.0
        self._running = True
        threading.Thread(target=self._watch_imu, daemon=True).start()

    def _forward_accel(self):
        return IMU_FORWARD_SIGN * float(getattr(self.motor.imu_device, IMU_AXIS))

    def _watch_imu(self):
        """Learn the resting value (robot sitting still at startup), then flag any backward
        jolt bigger than PUSH_ACCEL that didn't come from us slowing down."""
        dt = IMU_INTERVAL_MS / 1000
        rest = []
        while len(rest) < 25:
            a = self._forward_accel()
            if not math.isnan(a):
                rest.append(a)
            time.sleep(dt)
        baseline = sum(rest) / len(rest)
        self.imu_ready = True
        while self._running:
            a = self._forward_accel() - baseline
            if not math.isnan(a):
                self.accel = IMU_SMOOTH * self.accel + (1 - IMU_SMOOTH) * a
                if self.accel < -PUSH_ACCEL and time.monotonic() > self.quiet_until:
                    self.pushed_at = time.monotonic()
            time.sleep(dt)

    def pushed_from_front(self):
        return self.pushed_at is not None and time.monotonic() - self.pushed_at < PUSH_HOLD

    def floor_reading(self):
        return self.color.detect_color(), self.color.reflection()

    def levers(self):
        if self.controller is None:
            return 0.0, 0.0
        return self.controller.left_position() / 100, self.controller.right_position() / 100

    def drive(self, left, right):
        if (left, right) != self._last:       # only send changes over Bluetooth
            before = sum(self._last) / 2 if self._last else 0
            if (left + right) / 2 < before - 10:  # slowing down makes a backward jolt too
                self.quiet_until = time.monotonic() + BRAKE_QUIET
            self.motor.movement_move_tank(*motor_speeds(left, right), blocking=False)
            self._last = (left, right)

    def stop(self):
        self._running = False
        self.motor.stop()


class NoLego:
    """Stand-in for --no-lego: always on the ring floor, levers centered."""
    def floor_reading(self):  return "No color", 50
    def levers(self):         return 0.0, 0.0
    def pushed_from_front(self): return False
    def drive(self, left, right): pass
    def stop(self):           pass


def sense(bb, robot, vision):
    """Read every sensor into the blackboard."""
    now = time.monotonic()
    color, _ = robot.floor_reading()
    bb.floor = "tape" if color == TAPE_COLOR else "floor"
    if bb.floor == "tape":
        bb.back_saw_tape_at = now

    cam = vision.get()
    bb.opp_seen = bool(cam and cam.get("opp_seen"))
    bb.opp_x = cam.get("opp_x", 0.0) if cam else 0.0
    bb.opp_size = cam.get("opp_size", 0.0) if cam else 0.0
    bb.tape_ahead = cam.get("tape_ahead", 0.0) if cam else 0.0
    bb.tape_x = cam.get("tape_x", 0.0) if cam else 0.0

    bb.pushed_front = robot.pushed_from_front()

    # Out = the back crossed the tape (saw blue, now past it) and the camera now sees the
    # tape in front of us, for OUT_CONFIRM seconds in a row.
    crossed = (bb.back_saw_tape_at is not None and bb.floor != "tape"
               and now - bb.back_saw_tape_at < OUT_WINDOW)
    if crossed and bb.tape_ahead > TAPE_AHEAD:
        bb.out_since = bb.out_since or now
    else:
        bb.out_since = None

    bb.lever_left, bb.lever_right = robot.levers()


# ---------------------------------------------------------------------------
# Sad song (the LEGO parts can only beep, so it plays on the laptop)
# ---------------------------------------------------------------------------

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
    try:
        subprocess.Popen(["afplay", str(path)])
    except FileNotFoundError:
        print("(no afplay on this computer - imagine a sad trombone)")


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
    print("Accelerometer readings (Ctrl+C to stop). Shove the robot FORWARD by hand:")
    print("  the axis that jumps most is IMU_AXIS; if it goes DOWN, set IMU_FORWARD_SIGN = -1.")
    print("  Then shove it BACKWARD: 'forward' should go negative past PUSH_ACCEL and print PUSH!")
    imu = robot.motor.imu_device
    while True:
        flag = "PUSH!" if robot.pushed_from_front() else "     "
        print(f"\rX={imu.accelerometerX:7.0f} Y={imu.accelerometerY:7.0f} Z={imu.accelerometerZ:7.0f}"
              f"   forward={robot.accel:+7.0f}  {flag}", end="", flush=True)
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
    try:
        if args.readings:
            print_readings(robot)
        if args.imu:
            print_imu(robot)

        vision = Vision()
        print("Place the robot inside the ring and hold it still...")
        while not getattr(robot, "imu_ready", True):   # IMU learning its resting value
            time.sleep(0.05)
        print("GO!")

        bb = Blackboard()
        last_active = None
        while True:
            sense(bb, robot, vision)
            TREE.tick(bb)
            robot.drive(*bb.drive)
            if bb.active != last_active:    # print only when the decision changes
                print(f"[{bb.active:18s}] floor={bb.floor:4s} opp={'Y' if bb.opp_seen else 'n'} "
                      f"x={bb.opp_x:+.2f} size={bb.opp_size:.2f} tape_ahead={bb.tape_ahead:.2f} "
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


if __name__ == "__main__":
    main()
