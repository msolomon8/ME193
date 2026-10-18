"""Robot Sumo - test ONLY the LEGO Double Motor (no camera, color sensor or controller).

Run (from the ME193 folder), with the robot's wheels off the ground or lots of space:
    le-venv/bin/python "Project 5 - Robot Sumo/motor_test.py"              # at SPEED_SCALE from sumo_brain.py
    le-venv/bin/python "Project 5 - Robot Sumo/motor_test.py" --speed 0.3  # or pick a speed

It drives a short sequence of the same moves the behavior tree uses and says what the
robot SHOULD be doing at each step. If it's wrong:
  - drives backward when it says "forward"  -> flip MOTOR_BACKWARD in sumo_brain.py
  - turns right when it says "turn left"     -> swap the left/right speeds (tell Claude)
The card serial and MOTOR_BACKWARD come from sumo_brain.py, so fixes there carry over.
"""

import argparse
import sys
import time
from pathlib import Path

ME193 = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ME193))             # lelib.py
sys.path.insert(0, str(Path(__file__).resolve().parent))

from lelib import doubleMotor
from sumo_brain import (MOTOR_CARD, CHARGE_SPEED, PUSH_SPEED, SEARCH_TURN, FRONT_EDGE_MOVE,
                        SPEED_SCALE, motor_speeds, set_speed_scale)

# (what it should look like, left speed, right speed, seconds)
MOVES = [
    ("forward (charge)",           CHARGE_SPEED, CHARGE_SPEED, 1.0),
    ("backward (back off edge)",   *FRONT_EDGE_MOVE[0]),
    ("spin in place to the RIGHT (search)", SEARCH_TURN, -SEARCH_TURN, 1.0),
    ("spin in place to the LEFT",  -SEARCH_TURN, SEARCH_TURN, 1.0),
    ("curve forward to the RIGHT", 70, 30, 1.0),
    ("full power forward (push)",  PUSH_SPEED, PUSH_SPEED, 0.7),
]


def drive(motor, left, right):
    motor.movement_move_tank(*motor_speeds(left, right), blocking=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--speed", type=float, default=SPEED_SCALE, help="fraction of full speed")
    set_speed_scale(parser.parse_args().speed)

    print("Connecting to the Double Motor...")
    motor = doubleMotor()
    motor.connect(card_serial=MOTOR_CARD)
    print("Connected.\n")

    try:
        for name, left, right, secs in MOVES:
            input(f"Press Enter to: {name}  (left={left}, right={right}, {secs} s) ")
            drive(motor, left, right)
            time.sleep(secs)
            motor.stop()
        print("\nDone. Did every move match its description?")
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        motor.stop()


if __name__ == "__main__":
    main()
