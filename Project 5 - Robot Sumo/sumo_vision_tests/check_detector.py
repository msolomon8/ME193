"""Does a sumo-vision main.py pass the stress test?  Run this after ANY change to the detection code.

    python check_detector.py "Project 5 - Robot Sumo/sumo-vision/python/main.py"

It runs the real detection functions from that file on controlled synthetic pictures, on a grey floor AND an orange wood floor
(see synthetic_eval.py for the full table and for what each scene is), and prints PASS or FAIL for the things the robot must get right. The pass marks are my rules of
thumb. A pass means it survives these stress tests, NOT that your real camera is fine: use vision_check.py --guided for that.
"""

import sys

from synthetic_eval import FLOORS, Detector, evaluate

# (what must be true, key in the results, "max" = at most / "min" = at least, limit)
CRITERIA = [
    ("an empty ring is not reported as an opponent (floor that changes with distance, exposure and white balance drifting)",
     "empty ring: reports an opponent that isn't there", "max", 0.05),
    ("an empty ring is not reported as tape", "empty ring: reports tape that isn't there", "max", 0.05),
    ("a dark opponent is found", "opponent found: dark", "min", 0.90),
    ("a red opponent is found", "opponent found: red", "min", 0.90),
    ("a white opponent is found (white LEGO parts, the white track)", "opponent found: white", "min", 0.90),
    ("a teal LEGO opponent is found", "opponent found: LEGO teal", "min", 0.90),
    ("normal blue tape is seen with the starting colour range", "tape seen: normal blue (starting range)", "min", 0.90),
    ("a click on the plain floor is refused (a wooden floor must never become the tape colour)", "a click on the plain floor is refused", "min", 1.00),
    ("a ring corner in view does not make an empty ring report an opponent", "empty ring with a ring corner in view: reports an opponent", "max", 0.02),
    ("relearning the floor with a big ring corner in view does not spoil it (no opponent reported once the tape is gone)",
     "floor relearned with a big ring corner in view, then no tape: reports an opponent", "max", 0.02),
]
OPTIONAL = {"a click on the plain floor is refused",                  # only detectors that have click-to-learn can be asked this
            "floor relearned with a big ring corner in view, then no tape: reports an opponent"}      # and only ones that learn the floor
APRON_KEY = "  same, with SELF_MASK_BOTTOM = 0.10: false tape"
FLOOR_COLOURED_KEY = "opponent found: floor-coloured"


def judge(results):
    """[(passed, text)] for every criterion, plus the notes at the end."""
    lines = []
    for text, key, kind, limit in CRITERIA:
        if key not in results and key in OPTIONAL:
            continue
        value = results[key]
        ok = value <= limit if kind == "max" else value >= limit
        lines.append((ok, f"{text}: {value:.0%} (need {'at most' if kind == 'max' else 'at least'} {limit:.0%})"))
    if APRON_KEY in results:
        value = results[APRON_KEY]
        lines.append((value <= 0.05, f"the robot's own apron can be masked out (SELF_MASK_BOTTOM = 0.10 leaves {value:.0%} false tape)"))
    else:
        lines.append((False, "the robot's own apron can be masked out (this main.py has no SELF_MASK_BOTTOM, so a blue apron in view reads as tape)"))
    return lines


def main():
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    detector_path, everything_ok = sys.argv[1], True
    for floor_name, floor in FLOORS.items():
        results = evaluate(Detector(detector_path), floor=floor)
        lines = judge(results)
        print(f"\n=== on a {floor_name} ===")
        for ok, text in lines:
            print(f"{'PASS' if ok else 'FAIL'}  {text}")
        print(f"For information: a robot the colour of the floor is found {results[FLOOR_COLOURED_KEY]:.0%} of the time. "
              "Colour alone cannot see it; that is the case a trained detector would be for.")
        everything_ok &= all(p for p, _ in lines)
    print("\n" + ("ALL PASSED." if everything_ok else "SOME FAILED: see the lines marked FAIL."))
    sys.exit(0 if everything_ok else 1)


if __name__ == "__main__":
    main()
