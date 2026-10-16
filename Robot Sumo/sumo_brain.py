"""Robot Sumo - the laptop "brain": a behavior tree with sensor fusion and shared control.

Run (from the ME193 folder):
    le-venv/bin/python "Robot Sumo/sumo_brain.py"              # the real match
    le-venv/bin/python "Robot Sumo/sumo_brain.py" --readings   # print color sensor readings (for tuning)
    le-venv/bin/python "Robot Sumo/sumo_brain.py" --no-lego    # test the tree with only the camera

Inputs:
  - LEGO color sensor (autonomous): is the front of the robot on the ring floor, on the
    blue tape, or off the ring entirely?
  - Logitech webcam on the UNO Q (autonomous): where the opponent is and whether blue tape
    is coming up. It arrives over MQTT on ME193/Rogers/MK from the ArduinoApps
    "sumo-vision" app.
  - LEGO controller (teleop): the driver's strategy.
        left lever  = aggression: up = attack, down = defend
        right lever = side bias:  up = favor turning right, down = favor left
        both levers fully down for 1 s = "we're out" (backup for the sad song)

Sensor fusion: the two autonomous sensors cover each other's blind spots.
  - The color sensor is certain about the edge, but only once the robot is already on it.
    The camera sees tape coming but is noisy. The camera warns early (veer away), and the
    color sensor gives the hard "back off" signal.
  - The color sensor says WHERE we are (on the edge). The camera says WHICH WAY the edge
    runs (tape_x), so the robot turns away from the edge and not along it.
  - On the tape alone means retreat. On the tape AND the opponent filling the camera
    means the opponent is past the tape too, so if the driver is on attack, keep pushing.

The policy is a behavior tree, checked top to bottom 20 times a second:
    Selector
    |- out of the ring?                  -> stop, play the sad song
    |- in the middle of a maneuver?      -> keep doing it
    |- on tape + opponent in contact + attack mode -> finish the push
    |- on tape?                          -> back up, turn away from the tape
    |- opponent seen?                    -> charge (camera aim blended with driver bias)
    |- tape ahead (camera)?              -> veer away from it
    `- otherwise                         -> search, spinning toward the driver's side
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
DRIVE_SIGN = 1           # set to -1 if "forward" drives the robot backward

# ---- MQTT ----
MQTT_TOPIC = "ME193/Rogers/MK"
VISION_TIMEOUT = 0.5     # camera data older than this (s) is ignored

# ---- Color sensor (TODO: tune with --readings on the real ring) ----
TAPE_COLOR = "Blue"
OFF_REFLECTION_DIFF = 15 # reflection this far from the ring floor = off the ring
OUT_TIME = 1.0           # off the ring this long (s) = we lost

# ---- Camera thresholds ----
CONTACT_SIZE = 0.25      # opponent covers this much of the ground area = touching us
TAPE_AHEAD = 0.08        # this much blue in the bottom strip = edge coming up

# ---- Driving (Double Motor speeds are -100..100) ----
LOOP_HZ = 20
CHARGE_SPEED = 80
PUSH_SPEED = 100
TURN_GAIN = 40           # how hard to steer toward the opponent
SEARCH_TURN = 35
VEER_SPEED, VEER_TURN = 30, 40
BACKUP = [(-60, -60, 0.5), (50, -50, 0.4)]   # (left, right, seconds): back up, then turn
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
        self.floor = "ring"          # color sensor: "ring", "tape" or "off"
        self.off_since = None
        self.opp_seen = False        # camera
        self.opp_x = 0.0
        self.opp_size = 0.0
        self.tape_ahead = 0.0
        self.tape_x = 0.0
        self.aggression = 0.0        # driver, -1 (defend) .. +1 (attack)
        self.side_bias = 0.0         # driver, -1 (left) .. +1 (right)
        self.give_up_since = None
        # outputs / memory
        self.drive = (0, 0)
        self.maneuver = []           # queued (left, right, until) steps
        self.lost = False
        self.active = ""


def clip(v):
    return max(-100, min(100, int(v)))


# ---- Conditions ----

def is_out(bb):
    now = time.monotonic()
    off_long = bb.off_since is not None and now - bb.off_since > OUT_TIME
    driver_says = bb.give_up_since is not None and now - bb.give_up_since > 1.0
    return bb.lost or off_long or driver_says

def maneuvering(bb):
    while bb.maneuver and time.monotonic() > bb.maneuver[0][2]:
        bb.maneuver.pop(0)
    return bool(bb.maneuver)

def on_tape(bb):
    return bb.floor == "tape"

def in_contact(bb):
    return bb.opp_seen and bb.opp_size > CONTACT_SIZE

def attack_mode(bb):
    return bb.aggression >= 0

def opponent_seen(bb):
    return bb.opp_seen

def tape_ahead(bb):
    return bb.tape_ahead > TAPE_AHEAD


# ---- Actions ----

def lose(bb):
    bb.drive = (0, 0)
    if not bb.lost:
        bb.lost = True
        print("\nOUT OF THE RING. Stopping.")
        play_sad_song()

def continue_maneuver(bb):
    left, right, _ = bb.maneuver[0]
    bb.drive = (left, right)

def push(bb):
    bb.drive = (PUSH_SPEED, PUSH_SPEED)

def back_off_edge(bb):
    """Back up, then turn AWAY from the side the camera saw the tape on."""
    away = -1 if bb.tape_x > 0 else 1     # tape on the right -> turn left
    t = time.monotonic()
    bb.maneuver = []
    for left, right, secs in BACKUP:
        if left != right:                 # the turning step
            left, right = away * abs(left), -away * abs(right)
        t += secs
        bb.maneuver.append((left, right, t))
    continue_maneuver(bb)

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
    Sequence(Condition("out", is_out), Action("OUT - sad song", lose)),
    Sequence(Condition("maneuvering", maneuvering), Action("backing off edge", continue_maneuver)),
    Sequence(Condition("on tape", on_tape), Condition("contact", in_contact),
             Condition("attack mode", attack_mode), Action("FINISH THE PUSH", push)),
    Sequence(Condition("on tape", on_tape), Action("edge! backing off", back_off_edge)),
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


class LegoRobot:
    def __init__(self):
        import legoeducation as le
        from lelib import doubleMotor, colorSensor, controller
        print("Connecting to the Double Motor...")
        self.motor = doubleMotor()
        self.motor.connect(card_serial=MOTOR_CARD)
        print("Connecting to the color sensor...")
        self.color = colorSensor()
        self.color.connect(card_serial=COLOR_CARD)
        print("Connecting to the controller...")
        self.controller = controller()
        self.controller.connect(card_serial=CONTROLLER_CARD)
        print("All LEGO parts connected.")
        self._last = None

    def floor_reading(self):
        return self.color.detect_color(), self.color.reflection()

    def levers(self):
        return self.controller.left_position() / 100, self.controller.right_position() / 100

    def drive(self, left, right):
        if (left, right) != self._last:       # only send changes over Bluetooth
            self.motor.movement_move_tank(DRIVE_SIGN * left, DRIVE_SIGN * right, blocking=False)
            self._last = (left, right)

    def stop(self):
        self.motor.stop()


class NoLego:
    """Stand-in for --no-lego: always on the ring floor, levers centered."""
    def floor_reading(self):  return "No color", 50
    def levers(self):         return 0.0, 0.0
    def drive(self, left, right): pass
    def stop(self):           pass


def classify_floor(color, reflection, ring_reflection):
    if color == TAPE_COLOR:
        return "tape"
    if abs(reflection - ring_reflection) > OFF_REFLECTION_DIFF:
        return "off"
    return "ring"


def sense(bb, robot, vision, ring_reflection):
    """Read every sensor into the blackboard."""
    now = time.monotonic()
    color, reflection = robot.floor_reading()
    bb.floor = classify_floor(color, reflection, ring_reflection)
    if bb.floor == "off":
        bb.off_since = bb.off_since or now
    else:
        bb.off_since = None

    cam = vision.get()
    bb.opp_seen = bool(cam and cam.get("opp_seen"))
    bb.opp_x = cam.get("opp_x", 0.0) if cam else 0.0
    bb.opp_size = cam.get("opp_size", 0.0) if cam else 0.0
    bb.tape_ahead = cam.get("tape_ahead", 0.0) if cam else 0.0
    bb.tape_x = cam.get("tape_x", 0.0) if cam else 0.0

    bb.aggression, bb.side_bias = robot.levers()
    if bb.aggression <= -0.9 and bb.side_bias <= -0.9:
        bb.give_up_since = bb.give_up_since or now
    else:
        bb.give_up_since = None


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


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--readings", action="store_true", help="print color sensor readings")
    parser.add_argument("--no-lego", action="store_true", help="run without the LEGO parts")
    args = parser.parse_args()

    robot = NoLego() if args.no_lego else LegoRobot()
    try:
        if args.readings:
            print_readings(robot)

        vision = Vision()
        print("Place the robot inside the ring. Learning the ring floor...")
        samples = []
        for _ in range(10):
            samples.append(robot.floor_reading()[1])
            time.sleep(0.05)
        ring_reflection = sum(samples) / len(samples)
        print(f"Ring floor reflection = {ring_reflection:.0f}. GO!")

        bb = Blackboard()
        last_active = None
        while True:
            sense(bb, robot, vision, ring_reflection)
            TREE.tick(bb)
            robot.drive(*bb.drive)
            if bb.active != last_active:    # print only when the decision changes
                print(f"[{bb.active:18s}] floor={bb.floor:4s} opp={'Y' if bb.opp_seen else 'n'} "
                      f"x={bb.opp_x:+.2f} size={bb.opp_size:.2f} tape_ahead={bb.tape_ahead:.2f} "
                      f"attack={bb.aggression:+.1f} side={bb.side_bias:+.1f}")
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
