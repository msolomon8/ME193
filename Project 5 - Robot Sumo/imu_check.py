"""Robot Sumo - a guided check of the IMU (the accelerometer inside the LEGO Double Motor).

Run (from the ME193 folder), with the robot on the floor or a table and room to move it by hand. Nothing drives:
    python "Project 5 - Robot Sumo/imu_check.py"

Six short steps. For each one it tells you what to do, you press Enter, it counts 3 2 1 GO and records for a few
seconds while you do it, then it prints what it measured and PASS or FAIL:
    1. keep still      are readings arriving, and is the robot level
    2. tap             does the sensor react to a quick knock at all (if not, shoves cannot be judged)
    3. shove FORWARD   which axis is the driving direction, and does it go UP     (IMU_AXIS, IMU_FORWARD_SIGN)
    4. shove BACKWARD  does the behavior tree flag PUSH!                          (PUSH_ACCEL)
    5. the same, with the wheels HELD, as in a match (the brain commands speed 0 while waiting, and the motors resist)
    6. tilt            does the 'tipped over' check see it                        (TIP_ANGLE)
To run only the held-wheel shove (plus the still step it needs):  python imu_check.py --held-only
It reads the same numbers the behavior tree uses, and tells you the exact line of sumo_brain.py to change if one is wrong.
If a shove was too gentle or too sideways to judge, it says so and lets you try that step again.
"""

import math
import sys
import time
from pathlib import Path

ME193 = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ME193))                       # lelib.py
sys.path.insert(0, str(Path(__file__).resolve().parent))

AXES = ("accelerometerX", "accelerometerY", "accelerometerZ")
LETTERS = "XYZ"
SHOVE_FRACTION = 0.12     # a shove counts when an axis moves by this fraction of gravity
TAP_WEAK = 0.02           # a tap that moves the readings by less than this fraction of gravity was not noticed at all
CLEAR_RATIO = 1.5         # the axis that moves most must beat the next one by this factor
LEVEL_DEGREES = 10        # "flat" means within this many degrees of how it sat at the start
TRIES = 3


# --------------------------------------------------------------------------------------------- recording
def record(robot, seconds, sleep=time.sleep, now=time.monotonic, interval=0.01, hold=False):
    """Sample the raw accelerometer, and what the behavior tree computes from it, for `seconds`.
    hold=True keeps commanding speed 0 to the wheels, exactly as the behavior tree does while it waits."""
    samples, start = [], now()
    while now() - start < seconds:
        if hold:
            robot.drive(0, 0)
        imu = robot.motor.imu_device        # fetched EVERY time: the LEGO library swaps in a brand-new object for each reading
        raw = [float(getattr(imu, axis)) for axis in AXES]
        if not any(math.isnan(v) for v in raw):
            samples.append({"t": now() - start, "raw": raw, "accel": float(robot.accel),
                            "tilt": float(robot.tilt), "pushed": bool(robot.pushed_from_front())})
        sleep(interval)
    return samples


# --------------------------------------------------------------------------------------------- analysis
def analyze_rest(samples):
    if len(samples) < 20:
        return {"n": len(samples)}
    n = len(samples)
    mean = [sum(s["raw"][i] for s in samples) / n for i in range(3)]
    wiggle = max(max(s["raw"][i] for s in samples) - min(s["raw"][i] for s in samples) for i in range(3))
    changes = sum(1 for a, b in zip(samples, samples[1:]) if a["raw"] != b["raw"])
    return {"n": n, "mean": mean, "wiggle": wiggle, "gravity": math.sqrt(sum(m * m for m in mean)), "changes": changes,
            "up": max(range(3), key=lambda i: abs(mean[i])), "tilt": max(s["tilt"] for s in samples)}


def analyze_shove(samples, rest):
    """Which horizontal axis moved most in a shove, and which way did it move FIRST (starting to push comes before stopping)."""
    threshold = SHOVE_FRACTION * rest["gravity"]
    dev = {i: [s["raw"][i] - rest["mean"][i] for s in samples] for i in range(3)}
    peaks = {i: max((abs(d) for d in dev[i]), default=0.0) for i in range(3)}
    candidates = [i for i in range(3) if i != rest["up"]]      # the direction of travel is horizontal: leave out the gravity axis
    best = max(candidates, key=lambda i: peaks[i])
    second = max((peaks[i] for i in candidates if i != best), default=0.0)
    first = next((1 if d > 0 else -1 for d in dev[best] if abs(d) > threshold), 0)
    return {"peaks": peaks, "axis": best, "second": second, "first_sign": first, "threshold": threshold,
            "shoved": peaks[best] >= threshold, "clear": peaks[best] >= CLEAR_RATIO * second}


# --------------------------------------------------------------------------------------------- verdicts
# Each returns (checks, retry, fixes): checks = [(PASS|FAIL|NOTE, text)], retry = "the step could not be judged, try again",
# fixes = lines to change in sumo_brain.py.
def judge_rest(samples, state, cfg):
    a = analyze_rest(samples)
    if "mean" not in a:
        return [("FAIL", f"only {a['n']} readings arrived: is the Double Motor connected and switched on?")], True, []
    checks = [("PASS", f"readings are arriving ({a['n']} of them); gravity reads {a['gravity']:.0f} (that is 1 g in these units), mostly on {LETTERS[a['up']]}")]
    still = a["wiggle"] < 0.1 * a["gravity"]
    checks.append(("PASS" if still else "FAIL", f"the robot held still (the readings wiggled by {a['wiggle']:.0f})" if still
                   else f"the robot was moving (the readings wiggled by {a['wiggle']:.0f}): put it down and don't touch it"))
    level = a["tilt"] < LEVEL_DEGREES
    checks.append(("PASS" if level else "FAIL", f"level (tilt {a['tilt']:.0f} degrees)" if level
                   else f"not level (tilt {a['tilt']:.0f} degrees): put it flat on the floor or table"))
    if a["changes"] == 0:
        checks.append(("NOTE", f"the readings did not change AT ALL in {samples[-1]['t']:.0f} s. A normal accelerometer shows a little noise, so this one probably "
                               "smooths its readings heavily or only reports big changes. The tap step is there to find out whether it still sees quick jolts."))
    if still and level:
        state["rest"] = a
    return checks, not (still and level), []


def judge_tap(samples, state, cfg):
    """Does the sensor react to a quick knock at all? Looks at every axis, so it does not depend on which way is forward."""
    rest = state["rest"]
    changes = sum(1 for a, b in zip(samples, samples[1:]) if a["raw"] != b["raw"])
    peaks = [max((abs(s["raw"][i] - rest["mean"][i]) for s in samples), default=0.0) for i in range(3)]
    peak, axis = max(peaks), max(range(3), key=lambda i: peaks[i])
    seen = f"(biggest change per axis: X {peaks[0]:.0f}, Y {peaks[1]:.0f}, Z {peaks[2]:.0f}; the readings changed {changes} times)"
    if peak >= SHOVE_FRACTION * rest["gravity"]:
        state["responsive"] = True
        return [("PASS", f"the sensor saw the tap: axis {LETTERS[axis]} jumped by {peak:.0f} {seen}")], False, []
    state["responsive"] = False
    if peak >= TAP_WEAK * rest["gravity"]:
        return [("FAIL", f"only a weak reaction to the tap {seen}. Knock harder (a firm rap with your knuckles on the top). If it stays this weak, "
                         "the sensor is smoothing quick jolts away.")], True, []
    return [("FAIL", f"the sensor did not react to the tap at all {seen}. Knock firmly, a few times. If it still reads nothing, this sensor is not "
                     "reporting quick changes.")], True, []


def judge_forward(samples, state, cfg):
    rest = state["rest"]
    a = analyze_shove(samples, rest)
    best, peak = LETTERS[a["axis"]], a["peaks"][a["axis"]]
    if not a["shoved"]:
        return [("FAIL", f"no shove seen: the biggest change was X {a['peaks'][0]:.0f}, Y {a['peaks'][1]:.0f}, Z {a['peaks'][2]:.0f} (a shove must change a "
                         f"reading by at least {a['threshold']:.0f}). Push firmer and faster.")], True, []
    if not a["clear"]:
        return [("FAIL", f"two axes moved about the same ({best} by {peak:.0f}, the next by {a['second']:.0f}): push STRAIGHT forward, "
                         "not sideways, and try again.")], True, []
    checks, fixes = [("PASS", f"a shove was seen: axis {best} moved most (by {peak:.0f}; the next axis only {a['second']:.0f})")], []
    same_axis = AXES[a["axis"]] == cfg["axis"]
    if same_axis:
        checks.append(("PASS", f"{best} is the axis set as IMU_AXIS ({cfg['axis']})"))
    else:
        checks.append(("FAIL", f"the axis that jumped most is {best}, but IMU_AXIS is {cfg['axis']}"))
        fixes.append(f'IMU_AXIS = "{AXES[a["axis"]]}"')
    went = "UP" if a["first_sign"] > 0 else "DOWN"
    if same_axis and a["first_sign"] == cfg["sign"]:
        checks.append(("PASS", f"the first jump was {went}, which matches IMU_FORWARD_SIGN = {cfg['sign']}"))
    else:
        wants = "UP" if cfg["sign"] > 0 else "DOWN"
        checks.append(("FAIL", f"the first jump on {best} was {went}, but IMU_FORWARD_SIGN = {cfg['sign']} expects {wants}"
                       if same_axis else f"the first jump on {best} was {went}, so IMU_FORWARD_SIGN must be {a['first_sign']} for that axis"))
        fixes.append(f"IMU_FORWARD_SIGN = {a['first_sign']}")
    state["forward_fixes"] = fixes
    return checks, False, fixes


def judge_backward_held(samples, state, cfg):
    return judge_backward(samples, state, cfg, held=True)


def judge_backward(samples, state, cfg, held=False):
    with_wheels = "with the wheels HELD, " if held else ""
    if state.get("forward_fixes"):
        return [("NOTE", "not judged: the axis or sign from the forward shove is wrong, and the PUSH! check depends on both. "
                         "Change the lines from the forward shove, then run this check again.")], False, []
    rest, axis = state["rest"], AXES.index(cfg["axis"])
    threshold = SHOVE_FRACTION * rest["gravity"]
    dev = [s["raw"][axis] - rest["mean"][axis] for s in samples]
    peak = max((abs(d) for d in dev), default=0.0)
    if peak < threshold:
        return [("FAIL", f"no shove seen on {LETTERS[axis]} (the biggest change was only {peak:.0f}; it must be at least {threshold:.0f}). "
                         "Shove firmer and faster" + (": a sharp slap, not a slow push (held wheels resist a slow push, so it makes no jolt)." if held else "."))], True, []
    first = next((1 if d > 0 else -1 for d in dev if abs(d) > threshold), 0)
    if first * cfg["sign"] > 0:
        return [("FAIL", "the first jump was in the FORWARD direction: you probably pushed forward this time. Shove BACKWARD and try again.")], True, []
    low = min(s["accel"] for s in samples)
    flagged = any(s["pushed"] for s in samples)
    limit = cfg["push_accel"]
    if flagged and low < -limit:
        return [("PASS", f"{with_wheels}the behavior tree flagged PUSH! (its biggest backward jolt was {low:.0f}; the limit is -{limit})")], False, []
    if low < -0.25 * limit:
        suggest = max(50, int(round(0.7 * abs(low) / 10.0)) * 10)
        hint = (" Held wheels make the robot hard to move, so a slow push gives no jolt: use a sharp slap, like a ram." if held else "")
        return [("FAIL", f"{with_wheels}the behavior tree did NOT flag a push: its biggest backward jolt was {low:.0f}, and it needs to go below -{limit}. "
                         f"Shove harder, or lower the limit to about {suggest} (lower catches lighter hits, but also false alarms from bumps).{hint}")], False, \
               [f"PUSH_ACCEL = {suggest}"]
    return [("FAIL", f"{with_wheels}barely any jolt reached the behavior tree (its biggest backward jolt was only {low:.0f}). Shove firmer and faster"
                     + (": a sharp slap, not a slow push (the held wheels resist a slow push, so it makes no jolt)." if held else "."))], True, []


def judge_tilt(samples, state, cfg):
    peak = max(s["tilt"] for s in samples)
    end_time = samples[-1]["t"]
    tail = [s["tilt"] for s in samples if s["t"] >= end_time - 0.5] or [samples[-1]["tilt"]]
    end = sum(tail) / len(tail)
    limit = cfg["tip_angle"]
    checks, retry = [], False
    if peak > limit:
        checks.append(("PASS", f"tilt reached {peak:.0f} degrees, past TIP_ANGLE = {limit}, so the 'tipped over' stop would trigger"))
    else:
        checks.append(("FAIL", f"tilt only reached {peak:.0f} degrees; it must pass TIP_ANGLE = {limit}. Tilt it over further (about 60 degrees) "
                               "and hold it there for about 2 seconds."))
        retry = True
    if end < LEVEL_DEGREES:
        checks.append(("PASS", f"it came back to level (tilt {end:.0f} degrees at the end)"))
    else:
        checks.append(("FAIL", f"it did not come back to level (tilt {end:.0f} degrees at the end): put it flat again, and if it WAS flat, run the check again"))
        retry = True
    return checks, retry, []


# --------------------------------------------------------------------------------------------- the walkthrough
def merge_fixes(fixes):
    """One line per setting. If two steps suggest different PUSH_ACCEL values, the lower one catches both kinds of shove."""
    merged = {}
    for line in fixes:
        name, _, value = line.partition(" = ")
        if name not in merged:
            merged[name] = value
        elif name == "PUSH_ACCEL":
            merged[name] = str(min(int(merged[name]), int(value)))
    return [f"{name} = {value}" for name, value in merged.items()]


def run_check(robot, cfg, wait=input, out=print, sleep=time.sleep, now=time.monotonic, tick=1.0, seconds=(4.0, 4.0, 4.0, 4.0, 4.0, 6.0), only=None):
    """Returns {step name: ("PASS"|"FAIL"|"NOT JUDGED"|"SKIPPED", fixes)}. only: a set of step names to run (the still step always runs)."""
    out(f"Checking these settings from sumo_brain.py: IMU_AXIS = {cfg['axis']!r}, IMU_FORWARD_SIGN = {cfg['sign']}, "
        f"PUSH_ACCEL = {cfg['push_accel']}, TIP_ANGLE = {cfg['tip_angle']}\n")
    state, results = {}, {}
    steps = [
        ("keep still", "Put the robot down flat and DON'T touch it.", seconds[0], judge_rest),
        ("tap", "Knock on the top of the robot firmly with your knuckles, or slap it with your palm, 2 or 3 times (it should rattle, not slide). "
                "Start when you see GO!", seconds[1], judge_tap),
        ("shove forward", "Shove the robot FORWARD (toward the side the camera faces), about 15 cm: one quick, firm push, then let it stop. "
                          "Start pushing when you see GO!", seconds[2], judge_forward),
        ("shove backward", "Shove the robot BACKWARD, as if an opponent hit its front: one quick, firm push about 15 cm, then let it stop. "
                           "Start pushing when you see GO!", seconds[3], judge_backward),
        ("held shove", "The wheels will now be HELD, as when the robot waits in a match: they resist, that is normal. Shove the robot BACKWARD "
                       "sharply, like an opponent ramming it: a quick slap, not a slow push. Start when you see GO!", seconds[4], judge_backward_held),
        ("tilt", "Tilt the robot over about 60 degrees (lift one side, or stand it on its side). Hold it there for about 2 seconds, "
                 "then put it back flat.", seconds[5], judge_tilt),
    ]
    for number, (name, instruction, secs, judge) in enumerate(steps, 1):
        if only and name not in only and judge is not judge_rest:
            continue
        if number > 1 and "rest" not in state:
            results[name] = ("SKIPPED", [])
            continue
        if judge in (judge_forward, judge_backward, judge_backward_held) and state.get("responsive") is False:
            out(f"[{number}/{len(steps)}] {name.upper()}\n      Skipping this one: the sensor did not react to a firm tap, so a shove cannot be judged.\n")
            results[name] = ("NOT JUDGED", [])
            continue
        if judge in (judge_backward, judge_backward_held) and state.get("forward_fixes"):
            out(f"[{number}/{len(steps)}] {name.upper()}\n      Skipping this one: the PUSH! check depends on the axis and sign from the forward shove, "
                "which need changing first. Change them, then run this check again.\n")
            results[name] = ("NOT JUDGED", [])
            continue
        status, fixes = "SKIPPED", []
        for attempt in range(TRIES):
            out(f"[{number}/{len(steps)}] {name.upper()}" + ("" if attempt == 0 else f"   (try {attempt + 1})"))
            out("      " + instruction)
            wait("      Press Enter when you are ready, then do it when you see GO ... ")
            for word in ("3", "2", "1"):
                out(f"      {word} ...")
                sleep(tick)
            out(f"      GO!  (recording {secs:.0f} s)")
            samples = record(robot, secs, sleep=sleep, now=now, hold=judge is judge_backward_held)
            out("      stop.")
            checks, retry, fixes = judge(samples, state, cfg)
            for kind, text in checks:
                out(f"      {kind:<4}  {text}")
            kinds = {k for k, _ in checks}
            status = "NOT JUDGED" if kinds == {"NOTE"} else "PASS" if kinds <= {"PASS", "NOTE"} else "FAIL"     # a note alone never fails a step
            if not retry:
                break
            answer = wait("      Press Enter to try this step again (or type s and Enter to skip it) ... ")
            if answer.strip().lower().startswith("s"):
                status = "SKIPPED"
                break
        results[name] = (status, fixes)
        out("")
    if "rest" not in state:
        out("Could not get the resting readings, so the rest of the check was skipped. Is the Double Motor connected and the robot lying still?")
    out("SUMMARY")
    for name, (status, _) in results.items():
        out(f"  {name:<15}{status}")
    fixes = merge_fixes([f for _, fx in results.values() for f in fx])
    if state.get("responsive") is False:
        out("\nWHAT THIS MEANS: the accelerometer in this robot does not react to quick jolts. The behavior tree's 'pushed from the front' reflex "
            "(PUSH_ACCEL) depends on exactly that, so it may never fire on this hardware. Show this output to your teammate before relying on it.")
    if fixes:
        out("\nCHANGE THESE LINES in sumo_brain.py, then run this check again:")
        for line in fixes:
            out("    " + line)
    elif all(s == "PASS" for s, _ in results.values()):
        out("\nALL PASSED: the IMU settings in sumo_brain.py match this robot.")
    else:
        out("\nNot everything could be judged (see the lines above). Run the check again when you can.")
    return results


def main():
    from sumo_brain import LegoRobot, IMU_AXIS, IMU_FORWARD_SIGN, PUSH_ACCEL, TIP_ANGLE
    cfg = {"axis": IMU_AXIS, "sign": IMU_FORWARD_SIGN, "push_accel": PUSH_ACCEL, "tip_angle": TIP_ANGLE}
    only = {"held shove"} if "--held-only" in sys.argv else None
    robot = LegoRobot()
    try:
        print("Keep the robot still while it learns its resting value...")
        while not robot.imu_ready:
            time.sleep(0.05)
        print("Ready.\n")
        run_check(robot, cfg, only=only)
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        robot.stop()


if __name__ == "__main__":
    main()