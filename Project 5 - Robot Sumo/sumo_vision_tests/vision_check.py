"""Is the camera app telling the behavior tree the truth, and on time?

Listens to what the UnoQ's sumo-vision app publishes (by default ME193/Rogers/MK on test.mosquitto.org) and compares it with
the numbers sumo_brain.py acts on. The topic carries the camera's numbers AND the commands (charge, out, answer): this tool
tells them apart. Two modes:

  python vision_check.py --guided --brain "Project 5 - Robot Sumo/sumo_brain.py"
        five short scenarios (empty ring, tape, opponent, opponent left, opponent right), each with pass or fail and a hint
        about what to change.

  python vision_check.py --watch 15
        just listen for 15 seconds: messages per second, gaps, what else is on the topic.

Start the sumo-vision app first, with the robot sitting in an EMPTY ring (it learns the floor colour when it starts).
"""

import argparse
import collections
import json
import threading
import time

from common import connect, load_thresholds, make_client, percentile

DEFAULT_BROKER = "test.mosquitto.org"
DEFAULT_TOPIC = "ME193/Rogers/MK"
COMMAND_KEYS = ("charge", "out", "out_detected", "answer")

HINT = {
    "phantom_tape": ("tape_ahead was above the threshold in a scene with nothing to see. Look at the web page and check, in this order: "
                     "(1) is real ring tape visible BELOW the orange line (the robot is at the ring edge or a corner)? Then nothing is wrong: move "
                     "the robot away from the tape and rerun this scenario. (2) Is the only blue at the bottom of the picture the robot's OWN "
                     "apron? Then raise SELF_MASK_BOTTOM in the vision app until it passes, or tilt the camera up (a camera knocked out of "
                     "place during a match can show the apron: tape_ahead feeds the 'we're OUT?' check). (3) Is blue painted where there is none? "
                     "Click on the real tape so the app learns its true colour."),
    "phantom_opp": ("The camera reports an 'opponent' (or asks 'Opponent?') with nothing there. Look at the web page: the red areas are what it "
                    "thinks is not floor. Common causes: (1) the floor was learned somewhere else: a glossy floor looks different from "
                    "another position or angle, so press Relearn floor with the robot where it will fight (blue tape in view is fine, it is "
                    "left out of the floor model); (2) a shadow or a bright reflection along the floor; (3) something real in view (the white "
                    "track, your hand). If it is the bottom of the picture, raise SELF_MASK_BOTTOM."),
    "tape_miss": ("tape_ahead stayed too low. Look at the web page while the tape is laid out and check, in this order: (1) is the tape painted blue? "
                  "If not, CLICK ON THE TAPE in the picture so the app learns its real colour (blue tape can look purple, cyan or pale on a webcam). "
                  "(2) Is the blue BELOW the orange line? tape_ahead only counts that strip, so tape near the white horizon line reads 0.00 however "
                  "well it is detected: move the tape (or the robot) until it lands inside the strip. (3) Does it fill enough of the strip? "
                  "It must cover about 8% of the strip's area (the threshold in sumo_brain.py): a short piece or a thin line can fall short, so "
                  "use a long strip across the picture. If the tape fills the strip and still reads low, lower TAPE_AHEAD in sumo_brain.py."),
    "opp_miss": ("The camera isn't reporting the opponent. Press Relearn floor with an empty ring in view; if the object is the same colour as "
                 "the floor, colour alone can't see it (that is the case for a trained detector)."),
    "no_prompt": ("The camera sees the opponent but the app never asked 'Opponent?'. It asks only after OPP_CONFIRM frames in a row, and not for "
                  "IGNORE_TIME seconds after you pressed Reject. If you rejected a moment ago, wait a few seconds and run this scenario again."),
    "flipped": "Left and right are swapped. Set FLIP_X = True in the vision app.",
}


class Listener:
    """Records every camera message (arrival time on this computer's clock), keeps the commands apart, and counts strange ones."""

    def __init__(self, broker, port, topic):
        self.records = []            # camera messages: (time, dict) for JSON objects that have "tape_ahead"
        self.commands = []           # charge / out / answer messages: (time, dict)
        self.bad = 0                 # not JSON at all
        self.not_dict = 0            # JSON, but not an object (a bare number, a list)
        self.unknown = 0             # a JSON object that is neither a camera message nor one of our commands
        self.lock = threading.Lock()
        self.client = make_client("vision-check")
        self.client.on_message = self._on_message
        connect(self.client, broker, port)
        self.client.subscribe(topic)

    def _on_message(self, client, userdata, msg):
        t = time.perf_counter()
        try:
            obj = json.loads(msg.payload)
        except ValueError:
            self.bad += 1
            return
        if not isinstance(obj, dict):
            self.not_dict += 1
            return
        with self.lock:
            if "tape_ahead" in obj:
                self.records.append((t, obj))
            elif any(k in obj for k in COMMAND_KEYS):
                self.commands.append((t, obj))
            else:
                self.unknown += 1

    def between(self, t0, t1):
        with self.lock:
            return [(t, m) for t, m in self.records if t0 <= t <= t1]

    def commands_between(self, t0, t1):
        with self.lock:
            return [(t, m) for t, m in self.commands if t0 <= t <= t1]

    def close(self):
        self.client.loop_stop()
        self.client.disconnect()


# ----------------------------------------------------------------- the numbers
def watch_stats(records, t_start, t_end, timeout, bad=0, not_dict=0, unknown=0):
    """Rate, gaps and the time the behavior tree would have been blind (no camera message newer than `timeout`)."""
    times = [t for t, _ in records]
    duration = max(1e-9, t_end - t_start)
    edges = [t_start] + times + [t_end]
    gaps = [b - a for a, b in zip(edges, edges[1:])]
    tape = sorted(float(m.get("tape_ahead", 0.0)) for _, m in records)
    seen = [m for _, m in records if m.get("opp_seen")]
    return {"n": len(records), "rate": len(records) / duration, "duration": duration,
            "max_gap": max(gaps), "blind_s": sum(max(0.0, g - timeout) for g in gaps),
            "bad": bad, "not_dict": not_dict, "unknown": unknown,
            "tape_med": percentile(tape, 50) if tape else float("nan"), "tape_p95": percentile(tape, 95) if tape else float("nan"),
            "tape_max": tape[-1] if tape else float("nan"),
            "opp_frac": len(seen) / len(records) if records else 0.0,
            "ask_frac": sum(1 for _, m in records if m.get("ask")) / len(records) if records else 0.0}


def scenario_values(records):
    """What one scenario's messages looked like."""
    n = len(records)
    tape = sorted(float(m.get("tape_ahead", 0.0)) for _, m in records)
    seen = [m for _, m in records if m.get("opp_seen")]
    xs = sorted(float(m.get("opp_x", 0.0)) for m in seen)
    sizes = sorted(float(m.get("opp_size", 0.0)) for m in seen)
    asked = [m for _, m in records if m.get("ask") == "opponent"]
    return {"n": n, "tape_med": percentile(tape, 50) if tape else 0.0, "tape_p95": percentile(tape, 95) if tape else 0.0,
            "opp_frac": len(seen) / n if n else 0.0, "x_med": percentile(xs, 50) if xs else 0.0,
            "size_med": percentile(sizes, 50) if sizes else 0.0, "size_p95": percentile(sizes, 95) if sizes else 0.0,
            "ask_frac": len(asked) / n if n else 0.0}


SCENARIOS = [
    ("empty", "Robot inside the ring with NOTHING in front of it: no opponent, and no ring tape below the orange line on the web page "
              "(so not at the ring edge). This is also the check for the robot seeing its own apron."),
    ("tape", "Lay blue tape flat on the floor in front of the camera so that, on the web page, it shows up BELOW the orange line and stretches "
              "across most of the picture (use a long strip, or the real ring edge). Tape up near the white horizon line does not count: "
              "tape_ahead is the share of the area below the orange line that is blue."),
    ("opponent", "Put the other robot (or a box its size) in front of you, in the middle of the picture."),
    ("left", "Put the opponent in front but over to the LEFT of the picture."),
    ("right", "Put the opponent in front but over to the RIGHT of the picture."),
]


def judge(name, v, th):
    """(list of (passed, text), list of hint keys) for one scenario."""
    if v["n"] < 5:
        return [(False, f"only {v['n']} messages arrived: is the vision app running?")], []
    checks, hints = [], []
    if name == "empty":
        ok = v["tape_p95"] < th["TAPE_AHEAD"]
        checks.append((ok, f"tape_ahead stays under {th['TAPE_AHEAD']:.2f} (95% of messages at or below {v['tape_p95']:.2f})"))
        hints += [] if ok else ["phantom_tape"]
        ok = v["opp_frac"] < 0.05 and v["ask_frac"] < 0.05
        checks.append((ok, f"no opponent reported and no 'Opponent?' question raised (reported in {v['opp_frac']:.0%}, asked in {v['ask_frac']:.0%}"
                       + (f"; size up to {v.get('size_p95', 0.0):.2f}" if v["opp_frac"] > 0 and v.get("size_p95", 0.0) > 0 else "") + ")"))
        hints += [] if ok else ["phantom_opp"]
    elif name == "tape":
        ok = v["tape_med"] >= th["TAPE_AHEAD"]
        checks.append((ok, f"tape_ahead reaches {th['TAPE_AHEAD']:.2f} or more (typical {v['tape_med']:.2f}; 95% of messages at or below {v['tape_p95']:.2f}); that is the value the 'out' check compares with"))
        hints += [] if ok else ["tape_miss"]
    elif name == "opponent":
        ok = v["opp_frac"] > 0.8
        checks.append((ok, f"opponent seen in most messages ({v['opp_frac']:.0%})"))
        hints += [] if ok else ["opp_miss"]
        if ok:
            ok2 = v["ask_frac"] > 0.2
            checks.append((ok2, f"the app raised the 'Opponent?' question (open in {v['ask_frac']:.0%} of messages)"))
            hints += [] if ok2 else ["no_prompt"]
    else:                                                   # left or right
        sign = -1 if name == "left" else 1
        ok = v["opp_frac"] > 0.8
        checks.append((ok, f"opponent seen in most messages ({v['opp_frac']:.0%})"))
        hints += [] if ok else ["opp_miss"]
        if ok:
            ok2 = sign * v["x_med"] > 0.2
            checks.append((ok2, f"opp_x is on the {name} (typical {v['x_med']:+.2f}; left is negative)"))
            hints += [] if ok2 else (["flipped"] if sign * v["x_med"] < -0.2 else [])
    return checks, hints


def size_advice(values):
    """If the false 'opponent' in the empty scene was much smaller than the real ones, say which MIN_OPP_AREA to try."""
    empty = values.get("empty")
    if not empty or empty["opp_frac"] < 0.05 or empty["size_p95"] <= 0:
        return None
    real = [values[n]["size_med"] for n in ("opponent", "left", "right") if n in values and values[n]["opp_frac"] > 0.8]
    if not real:
        return None
    false_size, smallest = empty["size_p95"], min(real)
    if false_size * 1.6 > smallest:
        return (f"The false 'opponent' in the empty scene was about as big as the real ones (size {false_size:.2f} against {smallest:.2f}), so a "
                "size limit cannot separate them. Try pressing Relearn floor with the robot where it will fight.")
    suggestion = max(round((false_size * smallest) ** 0.5, 2), round(false_size + 0.01, 2))
    return (f"SIZE: the false 'opponent' was small (size up to {false_size:.2f}) and the real opponent read at least {smallest:.2f}. Setting "
            f"MIN_OPP_AREA = {suggestion:.2f} in sumo-vision/python/main.py (the app's default is 0.02) would ignore the small one with room on both "
            "sides. The price: anything smaller than that is ignored too. After changing it, move the box away from the robot until its outline "
            "turns yellow on the web page (yellow = too small) and check that this is farther than any opponent will be.")


def run_guided(listener, th, seconds=5.0, settle=1.0, wait=input, out=print):
    """Walk through the scenarios. `wait` is called to let you set up (tests pass a no-op). Returns {name: (checks, hints)}."""
    results, values = {}, {}
    out(f"\nThresholds from the behavior tree ({th['_source']}):\n"
        f"  tape_ahead above {th['TAPE_AHEAD']} counts as tape for the 'out' check (needs {th['OUT_CONFIRM']} s in a row, "
        f"within {th['OUT_WINDOW']} s of the back sensor seeing blue);  camera numbers older than {th['VISION_TIMEOUT']} s are ignored\n")
    for i, (name, instruction) in enumerate(SCENARIOS, 1):
        out(f"[{i}/{len(SCENARIOS)}] {instruction}")
        wait("      Press Enter when it's set up, then keep still ... ")
        out(f"      Recording for about {seconds + settle:.0f} s: keep still ...")      # without this line it looks like nothing happened
        time.sleep(settle)
        t0 = time.perf_counter()
        time.sleep(seconds)
        v = scenario_values(listener.between(t0, time.perf_counter()))
        checks, hints = judge(name, v, th)
        results[name] = (checks, hints)
        values[name] = v
        for ok, text in checks:
            out(f"      {'PASS' if ok else 'FAIL'}  {text}")
    failed = [h for _, hints in results.values() for h in hints]
    out("")
    if not failed and all(ok for checks, _ in results.values() for ok, _ in checks):
        out("ALL SCENARIOS PASSED: the camera reports what the behavior tree expects.")
    else:
        out("SOMETHING TO FIX:")
        for h in dict.fromkeys(failed):
            out("  - " + HINT[h])
        advice = size_advice(values)
        if advice:
            out("  - " + advice)
    return results


def run_watch(listener, th, seconds, out=print):
    t0 = time.perf_counter()
    time.sleep(seconds)
    t1 = time.perf_counter()
    s = watch_stats(listener.between(t0, t1), t0, t1, th["VISION_TIMEOUT"], listener.bad, listener.not_dict, listener.unknown)
    kinds = collections.Counter(k for _, m in listener.commands_between(t0, t1) for k in COMMAND_KEYS if k in m)
    out(f"Listened for {s['duration']:.0f} s.")
    out(f"  camera messages: {s['n']} ({s['rate']:.1f} per second; the app aims for about 10)")
    out(f"  longest silence: {s['max_gap'] * 1000:.0f} ms   (the tree ignores camera data older than {th['VISION_TIMEOUT'] * 1000:.0f} ms)")
    out(f"  time the tree would have been blind: {s['blind_s']:.1f} s ({100 * s['blind_s'] / s['duration']:.0f}%)")
    out(f"  commands also on the topic: " + (", ".join(f"{k} x{n}" for k, n in sorted(kinds.items())) or "none"))
    out(f"  not JSON: {s['bad']}   JSON but not an object (e.g. a bare number): {s['not_dict']}   objects that aren't ours: {s['unknown']}")
    out(f"  tape_ahead: median {s['tape_med']:.2f}, 95% at or below {s['tape_p95']:.2f}, highest {s['tape_max']:.2f}"
        f"   (counts as tape above {th['TAPE_AHEAD']})")
    out(f"  opponent reported in {s['opp_frac']:.0%} of messages; the app was asking a question in {s['ask_frac']:.0%}")
    verdict = []
    if s["n"] == 0:
        verdict.append("NO CAMERA MESSAGES. Is the sumo-vision app running, and is this the same topic and broker it uses?")
    else:
        if s["rate"] < 6:
            verdict.append("SLOW: under 6 camera messages a second means the app is struggling (CPU?) or the network is dropping them.")
        if s["blind_s"] / s["duration"] > 0.02:
            verdict.append("BLIND TOO OFTEN: the tree ignores stale camera data, so during these gaps the 'out' check cannot use the camera.")
        if s["bad"] or s["not_dict"] or s["unknown"]:
            verdict.append("SOMETHING ELSE IS PUBLISHING ON THIS TOPIC (another project, or someone with the same initials). "
                           "Give the camera app a topic of its own in BOTH programs, for example ME193/Rogers/MK/sumo.")
        # Watch mode can't tell whether the ring in front of the robot was empty, so these are conditional warnings.
        if s["tape_med"] >= th["TAPE_AHEAD"]:
            verdict.append(f"CHECK: tape_ahead sat at or above the threshold ({th['TAPE_AHEAD']}) for most of this time. If the ring in front "
                           "of the robot was EMPTY, the camera is seeing its own blue apron: run --guided to confirm, then raise SELF_MASK_BOTTOM.")
        if s["opp_frac"] > 0.5:
            verdict.append("CHECK: an opponent was reported in most messages. If nothing was in front of the robot, the camera is seeing "
                           "something that isn't floor (the white track? a shadow?): run --guided.")
    notes = []                                  # things worth knowing that are not failures
    if s["n"] and s["max_gap"] > th["VISION_TIMEOUT"]:
        notes.append(f"Note: the longest silence ({s['max_gap'] * 1000:.0f} ms) was longer than the tree's {th['VISION_TIMEOUT'] * 1000:.0f} ms limit, so for "
                     f"about {s['blind_s']:.1f} s the tree would have ignored the camera. Fine if it is rare. If it keeps happening, suspect the "
                     "network to the public broker or a busy board: run --watch again with the web page closed.")
    if s["n"] and 6 <= s["rate"] < 8:
        notes.append(f"Note: {s['rate']:.1f} messages a second is under the ~10 the app aims for. If the app's loop sleeps a fixed 0.1 s after its work, "
                     f"the work itself is taking about {1000 / s['rate'] - 100:.0f} ms; sleeping only the remainder of the 0.1 s brings it back to 10.")
    healthy = "LOOKS HEALTHY: steady messages and nothing strange on the topic."
    out("\n" + "\n".join(verdict or [healthy]) + ("\n" + "\n".join(notes) if notes else ""))
    return s


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--broker", default=DEFAULT_BROKER)
    p.add_argument("--port", type=int, default=1883)
    p.add_argument("--topic", default=DEFAULT_TOPIC)
    p.add_argument("--brain", help="path to sumo_brain.py, to read its thresholds")
    p.add_argument("--guided", action="store_true", help="the five-scenario check (the default)")
    p.add_argument("--watch", type=float, metavar="SECONDS", help="just listen for this long")
    p.add_argument("--seconds", type=float, default=5.0, help="how long each guided scenario is recorded")
    a = p.parse_args()
    th, source = load_thresholds(a.brain)
    th["_source"] = source
    listener = Listener(a.broker, a.port, a.topic)
    print(f"Connected to {a.broker}, listening on {a.topic}")
    try:
        if a.watch:
            run_watch(listener, th, a.watch)
        else:
            run_guided(listener, th, a.seconds)
    except KeyboardInterrupt:
        print("\nstopped")
    finally:
        listener.close()


if __name__ == "__main__":
    main()
