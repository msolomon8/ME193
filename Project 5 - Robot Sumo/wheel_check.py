"""Robot Sumo - check that the brain knows which way the wheels are rolling the robot.

The "back off" reflex (the back sensor reaches the tape while the robot was rolling FORWARD) decides from the wheel position counters.
The two counters may count in OPPOSITE directions when the robot drives straight (the motors are mirrored), so each wheel has its own
sign in sumo_brain.py (WHEEL_FORWARD_SIGNS). This check makes the robot drive itself, forward for one second and then backward for
one second, shows what each wheel's counter did, works out the right signs, and tells you the exact line to change if they are wrong.
(The motors hold the wheels when nothing is driving, so the robot cannot be rolled by hand.)

Run (from the ME193 folder). THE ROBOT WILL MOVE, so put it on a box with the wheels off the ground (best), or on the floor with
50 cm clear in front and behind, and keep a hand near the power button:
    python "Project 5 - Robot Sumo/wheel_check.py"            # gentle (40% of full speed)
    python "Project 5 - Robot Sumo/wheel_check.py" --speed 0.6
It uses the brain's own drive() and wheel readings, so a PASS means the reflex will see the same thing.
"""

import math
import sys
import time
from pathlib import Path

ME193 = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ME193))                       # lelib.py
sys.path.insert(0, str(Path(__file__).resolve().parent))

MIN_WHEEL = 30             # wheel degrees: a counter that changed by less than this did not really turn
TRIES = 3


def drive_and_measure(robot, left, right, seconds, sleep=time.sleep, now=time.monotonic):
    """Drive for `seconds`. Returns {"moved": (wheel 0, wheel 1 counter change) or None, "speed": (peak speed reading of each wheel) or None}.
    Always stops."""
    top = [0.0, 0.0]
    seen_speed = False
    try:
        start = now()
        while now() - start < seconds:
            robot.drive(left, right)
            try:
                for i in (0, 1):
                    value = abs(float(robot.motor.motor[i].speed))
                    if not math.isnan(value):
                        top[i], seen_speed = max(top[i], value), True
            except (AttributeError, IndexError, TypeError, ValueError):
                pass
            sleep(0.05)
        moved = robot.wheel_travel(seconds + 0.1)
    finally:
        robot.drive(0, 0)
    sleep(0.6)                                       # let it come to rest before the next measurement
    return {"moved": moved, "speed": tuple(top) if seen_speed else None}


def describe(moved):
    return f"wheel 0 {moved[0]:+.0f}, wheel 1 {moved[1]:+.0f} degrees"


def direction(sign):
    return "UP" if sign > 0 else "DOWN"


def run_check(robot, signs, speed, left=80, right=80, wait=input, out=print, sleep=time.sleep, now=time.monotonic, tick=1.0,
              seconds=1.0, overshoot_travel=90):
    """Returns ({"forward": (status, moved), "backward": (status, moved)}, lines to change)."""
    out(f"Checking: WHEEL_FORWARD_SIGNS = {tuple(signs)}, OVERSHOOT_TRAVEL = {overshoot_travel} wheel degrees; driving at {speed:.0%} of full speed\n")
    steps = [("forward", "The robot will drive FORWARD (toward the side the camera faces) for one second.", (left, right)),
             ("backward", "The robot will drive BACKWARD for one second.", (-left, -right))]
    results, readings, needed = {}, {}, None
    for number, (name, instruction, (l, r)) in enumerate(steps, 1):
        status = "SKIPPED"
        for attempt in range(TRIES):
            out(f"[{number}/{len(steps)}] DRIVE {name.upper()}" + ("" if attempt == 0 else f"   (try {attempt + 1})"))
            out("      " + instruction + " Wheels off the ground, or 50 cm clear, hand near the power button.")
            wait("      Press Enter when you are ready ... ")
            for word in ("3", "2", "1"):
                out(f"      {word} ...")
                sleep(tick)
            out("      GO!")
            got = drive_and_measure(robot, l, r, seconds, sleep=sleep, now=now)
            out("      stopped.")
            moved = got["moved"]
            speeds = got["speed"]
            if moved is None:
                out("      FAIL  no wheel readings arrived: is the Double Motor connected and reporting?")
                status = "FAIL"
            elif abs(moved[0]) < MIN_WHEEL and abs(moved[1]) < MIN_WHEEL:
                extra = f" The wheels' speed readings reached {speeds[0]:.0f} and {speeds[1]:.0f}." if speeds else ""
                out(f"      FAIL  neither wheel counter changed ({describe(moved)}) although the robot drove.{extra} Either the counters are not "
                    "updating while the robot drives, or something is blocking the wheels. Send this whole output.")
                status = "FAIL"
            elif abs(moved[0]) < MIN_WHEEL or abs(moved[1]) < MIN_WHEEL:
                out(f"      FAIL  only one wheel counter changed ({describe(moved)}). Is the other wheel turning, or blocked?")
                status = "FAIL"
            else:
                readings[name] = moved
                sign_of = lambda v: 1 if v > 0 else -1
                if name == "forward":
                    needed = (sign_of(moved[0]), sign_of(moved[1]))
                    if tuple(signs) == needed:
                        out(f"      PASS  driving forward: {describe(moved)}. The brain's signs {tuple(signs)} read this as FORWARD.")
                        status = "PASS"
                    else:
                        out(f"      FAIL  driving forward: {describe(moved)}. Wheel 0 counts {direction(needed[0])} and wheel 1 counts {direction(needed[1])} "
                            f"when the robot goes forward, but the brain's signs are {tuple(signs)}.")
                        status = "FAIL"
                else:
                    if needed is None:                       # the forward step was skipped: work the signs out from this one
                        needed = (-sign_of(moved[0]), -sign_of(moved[1]))
                    reversed_ok = moved[0] * needed[0] < 0 and moved[1] * needed[1] < 0
                    if reversed_ok and tuple(signs) == needed:
                        out(f"      PASS  driving backward: {describe(moved)}. Both counters reversed, as the brain expects.")
                        status = "PASS"
                    elif reversed_ok:
                        out(f"      FAIL  driving backward: {describe(moved)}. Both counters reversed, but the brain's signs {tuple(signs)} do not match.")
                        status = "FAIL"
                    else:
                        out(f"      FAIL  driving backward: {describe(moved)}. The counters did not both reverse compared with driving forward.")
                        status = "FAIL"
                break
            answer = wait("      Press Enter to try this step again (or type s and Enter to skip it) ... ")
            if answer.strip().lower().startswith("s"):
                status = "SKIPPED"
                break
        results[name] = (status, readings.get(name))
        out("")
    out("SUMMARY")
    for name, (status, moved) in results.items():
        out(f"  drive {name:<9}{status}" + (f"   ({describe(moved)})" if moved is not None else ""))
    fixes = []
    if needed is not None and tuple(signs) != needed and all(abs(v) >= MIN_WHEEL for m in readings.values() for v in m):
        fixes.append(f"WHEEL_FORWARD_SIGNS = {needed}")
    if fixes:
        out("\nCHANGE THIS LINE in sumo_brain.py, then run this check again:")
        for line in fixes:
            out("    " + line)
    elif all(s == "PASS" for s, _ in results.values()):
        out("\nALL PASSED: the brain reads forward as forward.")
        sizes = [abs(sum(m[i] * signs[i] for i in (0, 1))) / 2 for m in readings.values()]
        if sizes:
            avg = sum(sizes) / len(sizes)
            out(f"One second of driving turned the wheels about {avg:.0f} degrees at this speed, so OVERSHOOT_TRAVEL = {overshoot_travel} is about "
                f"{overshoot_travel / avg:.0%} of that: a short burst of forward driving counts, a nudge does not.")
    else:
        out("\nNot everything could be judged (see the lines above). Run the check again when you can.")
    return results, fixes


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--speed", type=float, default=0.4, help="fraction of full speed to drive at (default 0.4)")
    args = parser.parse_args()
    from sumo_brain import LegoRobot, WHEEL_FORWARD_SIGNS, OVERSHOOT_TRAVEL, CHARGE_SPEED, set_speed_scale
    set_speed_scale(args.speed)
    robot = LegoRobot()
    try:
        print("Keep the robot still for a moment...")
        while not robot.imu_ready:
            time.sleep(0.05)
        print("Ready.\n")
        run_check(robot, WHEEL_FORWARD_SIGNS, args.speed, left=CHARGE_SPEED, right=CHARGE_SPEED, overshoot_travel=OVERSHOOT_TRAVEL)
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        robot.drive(0, 0)
        robot.stop()


if __name__ == "__main__":
    main()