"""Record labeled IMU data from the Gait Identifier wearable (runs on the laptop).

The UNO Q app streams raw samples on ME193/Rogers/MiaGait/raw. This saves them to
data/<person>_<activity>_<time>.csv while you do the activity.

    python record.py mia walking            # 60 s after a 5 s countdown
    python record.py mia biking --seconds 90

Activities: standing, walking, running, biking. A sound plays when recording starts and
stops, so you can leave the laptop. Record every activity for several people, at
different speeds (slow walk, fast walk, jog, sprint; real bike and air-bike on your back).
"""
import argparse, csv, subprocess, sys, threading, time
from datetime import datetime
from pathlib import Path
import json
import paho.mqtt.client as mqtt

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "gait-identifier" / "python"))
import gait_features as gf

BROKER, PORT, TOPIC = "broker.hivemq.com", 1883, "ME193/Rogers/MiaGait"


def sound(name):
    subprocess.Popen(["afplay", f"/System/Library/Sounds/{name}.aiff"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("person")
    ap.add_argument("activity", choices=gf.CLASSES)
    ap.add_argument("--seconds", type=float, default=60)
    ap.add_argument("--countdown", type=float, default=5)
    args = ap.parse_args()

    rows, lock = [], threading.Lock()
    recording = threading.Event()
    last_msg = [0.0]

    def on_connect(client, userdata, flags, rc, props):
        client.subscribe(TOPIC + "/raw")

    def on_message(client, userdata, msg):
        last_msg[0] = time.time()
        if not recording.is_set():
            return
        d = json.loads(msg.payload)
        with lock:
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
            sys.exit("\nNo data from the board in 15 s. Is the gait-identifier app running and online?")
        time.sleep(0.2)
    print(" streaming.")

    for k in range(int(args.countdown), 0, -1):
        print(f"Start {args.activity} in {k}...", flush=True)
        time.sleep(1)
    sound("Glass")
    recording.set()
    end = time.time() + args.seconds
    while time.time() < end:
        with lock:
            n = len(rows)
        print(f"\rRecording {args.activity}: {end - time.time():4.0f} s left, {n} samples", end="", flush=True)
        time.sleep(0.5)
    recording.clear()
    sound("Hero")
    client.loop_stop()

    out = HERE / "data" / f"{args.person}_{args.activity}_{datetime.now():%m%d-%H%M%S}.csv"
    out.parent.mkdir(exist_ok=True)
    with lock, open(out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["seq"] + gf.AXES)
        w.writerows(rows)
    expected = args.seconds * gf.RATE
    print(f"\nSaved {len(rows)} samples ({len(rows) / expected:.0%} of expected) to {out.relative_to(HERE)}")


if __name__ == "__main__":
    main()
