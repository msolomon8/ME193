"""Live dashboard for the Gait Identifier (runs on the laptop, for demos).

Shows the last 10 s of what the wearable feels and, side by side, what model 1
(one peak) and model 2 (all IMU) think you are doing right now. Both decisions are
made on the UNO Q; this window only displays them.

    python live.py
"""
import json, sys, threading, time
from collections import deque
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import paho.mqtt.client as mqtt

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "gait-identifier" / "python"))
import gait_features as gf

BROKER, PORT, TOPIC = "broker.hivemq.com", 1883, "ME193/Rogers/MiaGait"
COLORS = dict(zip(gf.CLASSES, ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]))
ICON = {"standing": "STANDING", "walking": "WALKING", "running": "RUNNING", "biking": "BIKING", None: "no model yet"}
SECONDS = 10

lock = threading.Lock()
amag = deque([1.0] * SECONDS * gf.RATE, maxlen=SECONDS * gf.RATE)
gmag = deque([0.0] * SECONDS * gf.RATE, maxlen=SECONDS * gf.RATE)
state = {"pred": None, "last": 0.0}


def on_connect(client, userdata, flags, rc, props):
    client.subscribe([(TOPIC + "/raw", 0), (TOPIC + "/pred", 0)])


def on_message(client, userdata, msg):
    d = json.loads(msg.payload)
    with lock:
        state["last"] = time.time()
        if msg.topic.endswith("/raw"):
            for ax, ay, az, gx, gy, gz in d["s"]:
                amag.append((ax * ax + ay * ay + az * az) ** 0.5)
                gmag.append((gx * gx + gy * gy + gz * gz) ** 0.5)
        else:
            state["pred"] = d


client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
client.on_connect, client.on_message = on_connect, on_message
client.connect_async(BROKER, PORT, keepalive=30)
client.loop_start()

fig = plt.figure(figsize=(11, 6.5), facecolor="white")
gs = fig.add_gridspec(3, 2, height_ratios=[1, 1, 1.4], hspace=0.45)
ax_a = fig.add_subplot(gs[0, :])
ax_g = fig.add_subplot(gs[1, :], sharex=ax_a)
t = np.arange(-SECONDS * gf.RATE, 0) / gf.RATE
line_a, = ax_a.plot(t, list(amag), color="#2a78d6", lw=1.5)
line_g, = ax_g.plot(t, list(gmag), color="#eb6834", lw=1.5)
ax_a.set_ylabel("acceleration (g)")
ax_g.set_ylabel("rotation (deg/s)")
ax_g.set_xlabel("seconds ago")
for ax in (ax_a, ax_g):
    ax.grid(color="#e4e3dd", lw=0.6)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
panels = []
for col, title in enumerate(["MODEL 1 · one peak acceleration", "MODEL 2 · all IMU outputs"]):
    ax = fig.add_subplot(gs[2, col])
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title, loc="left", fontsize=10, color="#6b6a64")
    big = ax.text(0.5, 0.55, "", ha="center", va="center", fontsize=34, weight="bold", transform=ax.transAxes)
    small = ax.text(0.5, 0.15, "", ha="center", va="center", fontsize=10, color="#6b6a64", transform=ax.transAxes)
    panels.append((ax, big, small))
status = fig.text(0.01, 0.98, "", va="top", fontsize=9, color="#6b6a64")


def update(_):
    with lock:
        a, g, p, last = list(amag), list(gmag), state["pred"], state["last"]
    line_a.set_ydata(a)
    line_g.set_ydata(g)
    ax_a.set_ylim(0, max(2.0, max(a) * 1.1))
    ax_g.set_ylim(0, max(100.0, max(g) * 1.1))
    online = time.time() - last < 2
    status.set_text("wearable online" if online else "waiting for the wearable...")
    for i, (ax, big, small) in enumerate(panels):
        label = p and p.get(f"m{i + 1}")
        big.set_text(ICON.get(label, "") if online else "")
        ax.set_facecolor(COLORS[label] + "33" if (online and label in COLORS) else "#f4f3ef")
        for s in ax.spines.values():
            s.set_color(COLORS.get(label, "#e4e3dd"))
        if p and online and label:
            small.set_text(f"peak {p['peak_name'][-1]} = {p['peak']:.2f} g" if i == 0 else
                           f"cadence {p['cadence']:.1f} Hz · rotation {p['gyro']:.0f} deg/s")
        else:
            small.set_text("")
    return []


anim = FuncAnimation(fig, update, interval=100, cache_frame_data=False)
plt.show()
