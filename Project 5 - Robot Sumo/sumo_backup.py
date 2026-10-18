"""Robot Sumo - BACKUP brain: no UNO Q, no camera, no MQTT. Only the LEGO parts and this laptop.

Use it if the camera detection isn't working. It reuses the robot code in sumo_brain.py
(motors, color sensor, IMU, controller, sad song), so tune the settings there.

Run (from the ME193 folder, in a terminal so the keyboard works):
    le-venv/bin/python "Project 5 - Robot Sumo/sumo_backup.py"
    le-venv/bin/python "Project 5 - Robot Sumo/sumo_backup.py" --no-lego   # test without the LEGO parts

Inputs:
  - LEGO color sensor on the BACK (autonomous): is the back of the robot on the blue tape?
  - Double Motor's IMU (autonomous): pushed from the front, tipped over.
  - LEGO controller (teleop): absolute override, tank drive, same as sumo_brain.py.
  - This keyboard (teleop / shared control):
        Space  = you see the opponent in front: charge straight ahead for CHARGE_TIME
        Enter  = yes, we're out (answers the robot's question): sad song
        Delete = no, we're not out: keep wrestling
        o      = we're out, right now (no question needed)

Sensor fusion - are we out? Without the camera, the color sensor and the IMU decide
together: being shoved out backward means the IMU feels the push from the front AND the
back sensor goes across the tape (blue, then floor again). When the back leaves the tape
while we're being pushed, the robot stops on its own and asks you: Enter = sad song,
Delete = keep going. (Driving off the tape on purpose doesn't trigger it: no push.)

The policy is a behavior tree, checked top to bottom 20 times a second:
    Selector
    |- you said we're out (Enter / o)?          -> stop, play the sad song, quit
    |- back left the tape while pushed?         -> OUT? stop on its own, ask you
    |- a controller lever moved?                -> the driver drives (absolute override)
    |- tipped over (IMU)?                       -> stop until it's back on its wheels
    |- back on tape?                            -> full power forward off the edge, then turn
    |- in the middle of a maneuver?             -> keep doing it
    |- pushed from the front (IMU)?             -> full power forward, counteracting it
    |- you pressed Space?                       -> charge straight ahead
    `- otherwise                                -> sit still (the driver searches with the controller)
"""

import argparse
import time

import sumo_brain as brain
from sumo_brain import Condition, Action, Sequence, Selector, Keyboard

OUT_IGNORE = 3.0         # after Delete, don't ask about "out" again for this long (s)
PUSHED_WINDOW = 1.0      # "being pushed" = the IMU felt a push within this many seconds


# ---------------------------------------------------------------------------
# Blackboard and tree (actions and conditions come from sumo_brain.py)
# ---------------------------------------------------------------------------

class Blackboard(brain.Blackboard):
    def __init__(self):
        super().__init__()
        self.pushed_recently = False # IMU felt a push within PUSHED_WINDOW
        self.asking = False          # waiting for Enter / Delete about "out"


TREE = Selector(
    Sequence(Condition("out", brain.is_out), Action("OUT - sad song", brain.lose)),
    Sequence(Condition("out detected", brain.out_detected), Action("OUT? stopped, asking", brain.stop_out)),
    Sequence(Condition("lever moved", brain.driver_override), Action("DRIVER OVERRIDE", brain.driver_drives)),
    Sequence(Condition("tipped over", brain.tipped_over), Action("TIPPED OVER - stop", brain.stop_tipped)),
    Sequence(Condition("back on tape", brain.back_on_tape), Action("back edge: forward", brain.leave_back_edge)),
    Sequence(Condition("maneuvering", brain.maneuvering), Action("edge maneuver", brain.continue_maneuver)),
    Sequence(Condition("pushed from front", brain.pushed_front), Action("IMU PUSH! pushing back", brain.push_back)),
    Sequence(Condition("Space: charge", brain.charge_approved), Action("CHARGE (Space)", brain.charge)),
    Action("waiting (still)", brain.wait),
)


# ---------------------------------------------------------------------------
# Sensing
# ---------------------------------------------------------------------------

def sense(bb, robot, keyboard):
    """Read the LEGO sensors and the keyboard into the blackboard."""
    now = time.monotonic()
    was_on_tape = bb.floor == "tape"
    color, _ = robot.floor_reading()
    bb.floor = "tape" if color == brain.TAPE_COLOR else "floor"
    bb.pushed_front = robot.pushed_from_front()
    pushed_at = getattr(robot, "pushed_at", None)  # NoLego has no IMU
    bb.pushed_recently = pushed_at is not None and now - pushed_at < PUSHED_WINDOW
    bb.tipped = robot.tipped_over()
    bb.lever_left, bb.lever_right = robot.levers()

    for key in keyboard.pressed():
        if key == "o" or (key == "enter" and bb.asking):
            bb.out = True
            print(">> you: WE'RE OUT")
        elif key == "delete" and bb.asking:
            bb.asking, bb.out_detected, bb.out_ignore_until = False, False, now + OUT_IGNORE
            print(">> you: not out, keep going")
        elif key == "space":
            bb.charge_until = now + brain.CHARGE_TIME
            print(">> you: CHARGE")
        elif key == "delete" and now < bb.charge_until:
            bb.charge_until = 0.0
            print(">> you: stop charging")

    # Sensor fusion: the back just left the tape (blue -> floor) while the IMU says we're
    # being shoved from the front -> we were pushed backward over the line. Stop and ask.
    left_tape = was_on_tape and bb.floor != "tape"
    if (left_tape and bb.pushed_recently and not bb.out_detected
            and now > bb.out_ignore_until):
        bb.out_detected = bb.asking = True
        print("\a\nOUT? Pushed backward across the tape. Stopped.  Enter = sad song, Delete = keep going")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--no-lego", action="store_true", help="run without the LEGO parts")
    parser.add_argument("--speed", type=float, default=brain.SPEED_SCALE,
                        help=f"fraction of full speed, e.g. 0.3 (default {brain.SPEED_SCALE})")
    args = parser.parse_args()
    brain.set_speed_scale(args.speed)

    robot = brain.NoLego() if args.no_lego else brain.LegoRobot()
    keyboard = None
    try:
        print("BACKUP mode: no camera. Place the robot inside the ring and hold it still...")
        while not getattr(robot, "imu_ready", True):   # IMU learning its resting value
            time.sleep(0.05)
        keyboard = Keyboard()
        print("GO!  Space = charge, Enter = yes out, Delete = no / stop, o = out now, Ctrl+C = quit")

        bb = Blackboard()
        last_active = None
        while True:
            sense(bb, robot, keyboard)
            TREE.tick(bb)
            robot.drive(*bb.drive)
            if bb.active != last_active:    # print only when the decision changes
                print(f"[{bb.active:24s}] floor={bb.floor:5s} pushed={'Y' if bb.pushed_recently else 'n'} "
                      f"levers={bb.lever_left:+.1f}/{bb.lever_right:+.1f}")
                last_active = bb.active
            if bb.lost:
                time.sleep(3)               # let the song finish
                break
            time.sleep(1 / brain.LOOP_HZ)
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        robot.stop()
        if keyboard:
            keyboard.restore()


if __name__ == "__main__":
    main()
