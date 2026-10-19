"""Robot Sumo - find out WHY the IMU does not react to quick jolts.

imu_check.py found that the accelerometer readings did not change at all during knocks and shoves. This probe measures what the
Double Motor actually sends, using one fact about the LEGO library: every reading that arrives REPLACES motor.imu_device with a
brand-new object, so counting new objects counts readings, whether or not the numbers inside changed.

Run (from the ME193 folder). Nothing drives:
    python "Project 5 - Robot Sumo/imu_probe.py"                 # connected the way the behavior tree connects (motor + color sensor + controller)
    python "Project 5 - Robot Sumo/imu_probe.py" --motor-only    # ONLY the Double Motor: do the other Bluetooth devices slow it down?

Four short recordings (still, shake, push the wheels, tilt). It prints one line each and then what it thinks the numbers mean.
"""

import math
import statistics
import sys
import time
from pathlib import Path

ME193 = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ME193))                       # lelib.py
sys.path.insert(0, str(Path(__file__).resolve().parent))

FAST = 25.0         # readings per second: at least this many counts as "arriving on time" (50 are asked for)
SLOW = 8.0          # fewer than this many per second is "arriving rarely"
SWING = 200.0       # an accelerometer swing this big (about 0.2 g) counts as "reacting" to a hard shake


def _clean(values):
    return [v for v in values if not (isinstance(v, float) and math.isnan(v))]


def watch(motor, seconds, sleep=time.sleep, now=time.monotonic, interval=0.002):
    """Count the readings that arrive (new objects), and keep what was inside them."""
    start = now()
    last_imu, last_wheels = motor.imu_device, [motor.motor[0], motor.motor[1]]    # held on to, so a new object can never reuse an old identity
    imu_times, imu_values = [], []
    wheel_times, wheel_values = [[], []], [[], []]
    while now() - start < seconds:
        t = now() - start
        imu = motor.imu_device
        if imu is not last_imu:
            last_imu = imu
            imu_times.append(t)
            imu_values.append((imu.accelerometerX, imu.accelerometerY, imu.accelerometerZ,
                               imu.gyroscopeX, imu.gyroscopeY, imu.gyroscopeZ))
        for i in (0, 1):
            wheel = motor.motor[i]
            if wheel is not last_wheels[i]:
                last_wheels[i] = wheel
                wheel_times[i].append(t)
                wheel_values[i].append((wheel.position, wheel.speed))
        sleep(interval)
    return {"seconds": seconds, "imu_times": imu_times, "imu_values": imu_values, "wheel_times": wheel_times, "wheel_values": wheel_values}


def summarize(rec):
    s = rec["seconds"]
    times, values = rec["imu_times"], [v for v in rec["imu_values"] if not any(isinstance(x, float) and math.isnan(x) for x in v)]
    gaps = [b - a for a, b in zip(times, times[1:])]
    changes = sum(1 for a, b in zip(values, values[1:]) if a[:3] != b[:3])
    swing = [max(v[i] for v in values) - min(v[i] for v in values) if values else 0.0 for i in range(6)]
    wheels = []
    for i in (0, 1):
        pos = _clean([p for p, _ in rec["wheel_values"][i]])
        spd = _clean([p for _, p in rec["wheel_values"][i]])
        wheels.append({"rate": len(rec["wheel_times"][i]) / s, "moved": (max(pos) - min(pos)) if pos else 0.0,
                       "speed_swing": (max(spd) - min(spd)) if spd else 0.0})
    return {"rate": len(times) / s, "gap_ms": statistics.median(gaps) * 1000 if gaps else None, "changes_per_s": changes / s,
            "accel_swing": swing[:3], "gyro_swing": swing[3:], "wheels": wheels}


def diagnose(result):
    """Plain-words reading of the four recordings. result: {phase: summarize(...)}"""
    still, shake, push, tilt = result["still"], result["shake"], result["push the wheels"], result["tilt"]
    lines = []
    rate, swing = shake["rate"], max(shake["accel_swing"])
    gap = f" (typically {shake['gap_ms']:.0f} ms apart)" if shake["gap_ms"] else ""
    if rate >= FAST and swing >= SWING:
        lines.append(f"THE IMU LOOKS FAST AND REACTIVE: {rate:.0f} readings a second{gap}, and a hard shake moved the accelerometer by {swing:.0f}. "
                     "The IMU is fine. (An earlier imu_check.py read one stale snapshot and reported frozen readings; that is fixed: run imu_check.py again.)")
    elif rate >= FAST:
        lines.append(f"THE READINGS ARRIVE ON TIME BUT BARELY REACT: {rate:.0f} readings a second{gap}, yet a hard shake moved the accelerometer by only "
                     f"{swing:.0f}. The sensor is smoothing quick jolts away (or ignoring small changes), so a push cannot be seen in the accelerometer.")
    elif rate < SLOW and still["rate"] < SLOW:
        lines.append(f"THE READINGS ARRIVE RARELY: only {rate:.1f} a second while shaking and {still['rate']:.1f} a second when still, although "
                     "the code asks for 50. A quick jolt lasts about a tenth of a second, so it falls between readings.")
    else:
        lines.append(f"MIXED PICTURE: {rate:.1f} readings a second while shaking, {still['rate']:.1f} when still, accelerometer swing {swing:.0f}.")
    if still["rate"] < SLOW <= rate:
        lines.append(f"Also: when the robot is still almost no readings arrive ({still['rate']:.1f} a second), but {rate:.0f} a second while it moves: the "
                     "hardware only reports while something changes. That is why a still robot shows 'no change at all'.")
    if tilt["rate"] >= SLOW and max(tilt["accel_swing"]) >= SWING and rate < FAST:
        lines.append(f"Slow tilting does get through ({tilt['rate']:.0f} a second, moving the accelerometer by {max(tilt['accel_swing']):.0f}): the stream works, "
                     "it just cannot follow fast changes. That matches the working tilt check.")
    wheel = max(push["wheels"], key=lambda w: w["moved"])
    if wheel["moved"] >= 90 and max(w["rate"] for w in push["wheels"]) >= SLOW:
        lines.append(f"THE WHEELS REACT TO BEING PUSHED: a wheel turned by {wheel['moved']:.0f} degrees and its reports arrived "
                     f"{max(w['rate'] for w in push['wheels']):.0f} times a second. Detecting 'pushed from the front' from the wheels, instead of the "
                     "accelerometer, looks possible, as a backup if you ever need one.")
    else:
        lines.append(f"The wheel reports did not clearly follow the push (moved {wheel['moved']:.0f} degrees; "
                     f"{max(w['rate'] for w in push['wheels']):.1f} reports a second), so the wheels are not an easy replacement either.")
    return lines


def run_probe(motor, wait=input, out=print, sleep=time.sleep, now=time.monotonic, tick=1.0, seconds=(3.0, 5.0, 5.0, 5.0)):
    steps = [
        ("still", "Put the robot down flat and DON'T touch it.", seconds[0]),
        ("shake", "Shake the robot HARD and without stopping: back and forth and side to side, like shaking a box. Start when you see GO!", seconds[1]),
        ("push the wheels", "Roll the robot forward about 20 cm by hand, then back again, a few times (the wheels should turn). Start when you see GO!", seconds[2]),
        ("tilt", "Tilt the robot slowly back and forth between flat and about 45 degrees. Start when you see GO!", seconds[3]),
    ]
    result = {}
    out(f"{'':<17}{'IMU readings':>14}{'(gap)':>9}{'values changed':>16}{'accelerometer swing X / Y / Z':>34}{'wheel reports':>15}{'wheel moved':>13}")
    for number, (name, instruction, secs) in enumerate(steps, 1):
        out(f"\n[{number}/{len(steps)}] {name.upper()}\n      {instruction}")
        wait("      Press Enter when you are ready ... ")
        for word in ("3", "2", "1"):
            out(f"      {word} ...")
            sleep(tick)
        out(f"      GO!  (recording {secs:.0f} s)")
        rec = watch(motor, secs, sleep=sleep, now=now)
        out("      stop.")
        r = summarize(rec)
        result[name] = r
        gap = f"{r['gap_ms']:.0f} ms" if r["gap_ms"] else "-"
        a = r["accel_swing"]
        out(f"      {name:<15}{r['rate']:>9.1f} /s{gap:>10}{r['changes_per_s']:>12.1f} /s{a[0]:>12.0f} /{a[1]:>6.0f} /{a[2]:>6.0f}"
            f"{max(w['rate'] for w in r['wheels']):>11.1f} /s{max(w['moved'] for w in r['wheels']):>10.0f} deg")
    out("\nWHAT THIS MEANS")
    for line in diagnose(result):
        out("  - " + line)
    return result


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--motor-only", action="store_true",
                        help="connect ONLY the Double Motor (no color sensor, no controller) to see whether sharing Bluetooth with them slows the IMU")
    args = parser.parse_args()
    from sumo_brain import MOTOR_CARD, IMU_INTERVAL_MS
    robot = None
    try:
        if args.motor_only:
            from lelib import doubleMotor
            print("Connecting to the Double Motor ONLY...")
            motor = doubleMotor()
            motor.connect(card_serial=MOTOR_CARD, device_notification_delay=IMU_INTERVAL_MS)
            print("Connected.\n")
        else:
            from sumo_brain import LegoRobot
            robot = LegoRobot()
            motor = robot.motor
            print()
        run_probe(motor)
        print(f"\nConnected as: {'Double Motor only' if args.motor_only else 'motor + color sensor + controller (like the behavior tree)'}. "
              "Send me this whole output.")
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        if robot is not None:
            robot.stop()
        elif args.motor_only:
            try:
                motor.stop()
            except Exception:
                pass


if __name__ == "__main__":
    main()
