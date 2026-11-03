"""Record a session with a known step count (runs on the laptop).

The UNO Q step-counter app streams samples on ME193/Rogers/MiaSteps/raw. Count your
steps out loud between the two sounds (both feet count). Afterwards this asks how many
steps you took and, if you walked a measured course, how far.

    python record.py walking              # 45 s between the sounds
    python record.py walking --seconds 10 # a short walk
    python record.py walking --course 5   # walk a measured 5 m, press Enter at the end
    python record.py standing             # zero-step sessions teach the net what is NOT a step

Each session saves data/<activity>_<time>.csv (seq, ax, ay, az, gx, gy, gz, dist) plus a
.json with the counted steps, distance and where the counted part starts and ends. The
recording keeps 2 s before the first sound and after the second, so every counted moment
sits in the middle of some 2 s network window.

Samples can reach the laptop seconds late and in bursts (Wi-Fi). So the sounds are placed
by the board's own sample numbers (seq, 50 per second), not by when samples arrive: the
least-delayed message tells how board time lines up with laptop time.
"""
import argparse, csv, json, subprocess, sys, threading, time
from datetime import datetime
from pathlib import Path
import paho.mqtt.client as mqtt

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import steps_data as sd

BROKER, PORT, TOPIC = "broker.emqx.io", 1883, "ME193/Rogers/MiaSteps"
PAD = 2.0   # seconds kept before the start sound and after the end sound


def sound(name):
    subprocess.Popen(["afplay", f"/System/Library/Sounds/{name}.aiff"])


def ask_number(prompt, allow_blank=False):
    while True:
        text = input(prompt).strip()
        if allow_blank and text == "":
            return None
        try:
            v = float(text)
            if v >= 0:
                return v
        except ValueError:
            pass
        print("  Please type a number" + (" (or press Enter to skip)" if allow_blank else ""))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("activity", help="walking, running, standing, biking, stairs, fidget, ... (just a label)")
    ap.add_argument("--seconds", type=float, default=45, help="time between the two sounds")
    ap.add_argument("--course", type=float, help="metres of a measured course: walk it, press Enter at the end")
    ap.add_argument("--countdown", type=float, default=5)
    ap.add_argument("--note", default="", help="anything worth remembering, e.g. 'slow', 'hallway 20 m'")
    args = ap.parse_args()

    rows, lock = [], threading.Lock()
    recording = threading.Event()
    last_msg = [0.0]
    offset = [float("inf")]        # laptop time minus board time; the smallest seen is the true one

    def on_connect(client, userdata, flags, rc, props):
        client.subscribe(TOPIC + "/raw")

    def on_message(client, userdata, msg):
        now = last_msg[0] = time.time()
        d = json.loads(msg.payload)
        with lock:
            offset[0] = min(offset[0], now - (d["seq"] + len(d["s"]) - 1) / sd.RATE)
            if not recording.is_set():
                return
            for i, s in enumerate(d["s"]):
                rows.append([d["seq"] + i] + s)

    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
    client.on_connect, client.on_message = on_connect, on_message
    client.connect(BROKER, PORT, keepalive=30)
    client.loop_start()

    print("Waiting for the wearable...", end="", flush=True)
    t0 = time.time()
    while time.time() - last_msg[0] > 1:
        if time.time() - t0 > 15:
            sys.exit("\nNo data from the board in 15 s. Is the step-counter app running and online?")
        time.sleep(0.2)
    print(" streaming.")

    def last_seq():
        with lock:
            return rows[-1][0] if rows else -1

    def board_seq(t):
        """The board's sample number at laptop time t."""
        with lock:
            return int(round((t - offset[0]) * sd.RATE))

    recording.set()                               # save from the countdown on; trimmed below
    for k in range(int(args.countdown), 0, -1):
        print(f"Start {args.activity} at the sound, counting steps out loud... {k}", flush=True)
        time.sleep(1)
    sound("Glass")
    t_start = time.time()
    if args.course:
        input(f"Walk the {args.course:g} m, then press Enter at the end ")
    else:
        end_t = t_start + args.seconds
        while time.time() < end_t:
            print(f"\rCounting: {end_t - time.time():4.0f} s left", end="", flush=True)
            time.sleep(0.25)
    sound("Hero")
    t_end = time.time()
    seconds = round(t_end - t_start, 1)
    print("\nStop and stand still...", flush=True)
    time.sleep(PAD)
    # wait for late samples: everything up to 2 s after the end sound
    want, t_wait = board_seq(t_end) + int(PAD * sd.RATE), time.time()
    while last_seq() < want and time.time() - t_wait < 20:
        time.sleep(0.2)
    recording.clear()
    client.loop_stop()

    s0, s1 = board_seq(t_start), board_seq(t_end)
    with lock:
        lo, hi = s0 - int(PAD * sd.RATE), s1 + int(PAD * sd.RATE)
        rows[:] = [r for r in rows if lo <= r[0] <= hi]
        start = sum(1 for r in rows if r[0] < s0)
        end = sum(1 for r in rows if r[0] < s1)
    if end - start < 0.8 * seconds * sd.RATE:
        print(f"Warning: only {(end - start) / sd.RATE:.1f} s of the {seconds:.1f} s walk arrived. Record it again.")

    steps = ask_number("How many steps did you count between the sounds? (both feet, 0 if none) ")
    if args.course:
        dist = args.course
    else:
        dist = ask_number("Distance covered between the sounds in metres (Enter if not measured, 0 if in place): ",
                          allow_blank=True)

    stem = f"{args.activity}_{datetime.now():%m%d-%H%M%S}"
    out = HERE / "data" / f"{stem}.csv"
    out.parent.mkdir(exist_ok=True)
    with lock, open(out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["seq"] + sd.CHANNELS)
        w.writerows(rows)
    meta = {"activity": args.activity, "steps": int(round(steps)), "distance_m": dist,
            "start": start, "end": end, "seconds": seconds, "note": args.note,
            "recorded": datetime.now().isoformat(timespec="seconds")}
    out.with_suffix(".json").write_text(json.dumps(meta, indent=2))
    expected = (seconds + 2 * PAD) * sd.RATE
    no_echo = sum(1 for r in rows if r[-1] == 0) / max(len(rows), 1)
    print(f"Saved {len(rows)} samples ({len(rows) / expected:.0%} of expected) to {out.relative_to(HERE)}")
    if no_echo > 0.9:
        print(f"Note: the ultrasonic sensor saw nothing {no_echo:.0%} of the time. Check its wiring and aim.")


if __name__ == "__main__":
    main()
