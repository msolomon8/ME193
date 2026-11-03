"""Count steps live with the trained nets (runs on the laptop).

Listens to the wearable on ME193/Rogers/MiaSteps/raw. Every 0.5 s the last 2 s of samples
go through all 5 nets. Their average answer (steps in the middle 0.5 s) is added to the
count, and the result goes out on ME193/Rogers/MiaSteps/count for live.html and the LED
matrix.

    python count_live.py            # type r + Enter (or press Reset on the page) to start again

The +- is a 95 % range. It combines how far off the nets were on recordings they never saw
(train.py's cross-validation, which grows with the count) and how much the 5 nets disagree
right now.
"""
import json, sys, threading, time
from collections import deque
from pathlib import Path
import numpy as np
import paho.mqtt.client as mqtt
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import steps_data as sd
import step_net as sn

BROKER, PORT, TOPIC = "broker.emqx.io", 1883, "ME193/Rogers/MiaSteps"
torch.set_num_threads(1)

model_file = HERE / "models" / "stepnet.pt"
if not model_file.exists():
    sys.exit("No models/stepnet.pt yet. Record sessions (record.py), then run train.py.")
nets, info = sn.load(model_file)
sd_rel = info.get("nn_sd_pct", 5.0) / 100
dist_rel = (info.get("distance_sd_pct") or info.get("distance_mean_abs_error_pct") or 15.0) / 100
step_len = info.get("step_length", {"a": 0.225, "b": 0.105, "height_m": 1.68})
print(f"{len(nets)} nets loaded. Cross-validated error: {info.get('nn_mean_abs_error_pct', float('nan')):.1f} % "
      f"average, sd {sd_rel * 100:.1f} %. Distance sd {dist_rel * 100:.0f} %.")

lock = threading.Lock()
buf = deque(maxlen=sd.WINDOW)
state = {"since": 0, "next_seq": None, "steps": 0.0, "per_net": np.zeros(len(nets)), "dist": 0.0, "hops": deque(maxlen=4)}


def reset():
    with lock:
        state.update(steps=0.0, per_net=np.zeros(len(nets)), dist=0.0)
        state["hops"].clear()
    print("\nCount reset.")


def hop():
    """Count the middle 0.5 s of the newest 2 s window."""
    X = np.array(buf, dtype=np.float32).T[None]
    per_net = sn.window_counts(nets, X)[:, 0]
    k = float(per_net.mean())
    state["steps"] += k
    state["per_net"] += per_net                       # each net keeps its own running total
    state["hops"].append(k)
    cad = sum(state["hops"]) / (len(state["hops"]) * sd.HOP / sd.RATE)      # steps per second, last 2 s
    state["dist"] += k * step_len["height_m"] * (step_len["a"] + step_len["b"] * cad)
    ens_sd = float(state["per_net"].std())            # how much the nets' totals disagree
    pm = 1.96 * np.hypot(sd_rel * state["steps"], ens_sd)
    msg = {"steps": int(round(state["steps"])), "exact": round(state["steps"], 2), "pm": round(float(pm), 1),
           "spm": round(cad * 60), "dist_m": round(state["dist"], 1), "dist_pm": round(dist_rel * 1.96 * state["dist"], 1),
           "hop": round(k, 2), "nets": [round(float(v), 2) for v in per_net]}
    client.publish(TOPIC + "/count", json.dumps(msg))
    print(f"\r{msg['steps']:5d} ± {msg['pm']:4.1f} steps   {msg['spm']:3d} steps/min   "
          f"{msg['dist_m']:6.1f} ± {msg['dist_pm']:4.1f} m   ", end="", flush=True)


def on_connect(client, userdata, flags, rc, props):
    client.subscribe([(TOPIC + "/raw", 0), (TOPIC + "/reset", 0)])
    print(f"Listening on {TOPIC}/raw")


def on_message(client, userdata, msg):
    if msg.topic.endswith("/reset"):
        reset()
        return
    d = json.loads(msg.payload)
    with lock:
        if state["next_seq"] is not None and d["seq"] - state["next_seq"] > 5:
            buf.clear()                                       # a long gap: start a fresh window
        state["next_seq"] = d["seq"] + len(d["s"])
        for s in sd.scale(d["s"]):
            buf.append(s)
            state["since"] += 1
            if len(buf) == sd.WINDOW and state["since"] >= sd.HOP:
                state["since"] = 0
                hop()


client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
client.on_connect, client.on_message = on_connect, on_message
client.connect_async(BROKER, PORT, keepalive=30)
client.loop_start()
print("Type r + Enter to reset the count, Ctrl+C to quit.")
try:
    for line in sys.stdin:
        if line.strip().lower() == "r":
            reset()
except KeyboardInterrupt:
    pass
