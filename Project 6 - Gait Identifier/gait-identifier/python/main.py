# Gait Identifier - UNO Q Linux side.
#
# Every 0.1 s: collect new IMU samples from the sketch (50 Hz MPU-6050) and stream them
# to the laptop over MQTT (ME193/Rogers/MiaGait/raw) so record.py can save labeled data.
# Every 0.5 s: take the last 2 s of samples, compute features (gait_features.py) and run
# both models:
#   model 1 - a decision tree on ONE feature, the peak acceleration along one axis
#   model 2 - a decision tree on all IMU features (both accel and gyro)
# Model 2's answer goes to the LED matrix as a stick figure that stands, walks, runs or
# bikes at your measured stride rate. Both answers go to ME193/Rogers/MiaGait/pred for
# live.html on the laptop.
#
# The models come from model.json next to this file, or from the laptop: train.py --send
# publishes a new model on ME193/Rogers/MiaGait/model (retained), and this app saves and
# uses it straight away, no redeploy needed.
from arduino.app_utils import *
import paho.mqtt.client as mqtt
import json, time, uuid
from collections import Counter, deque
from pathlib import Path
import gait_features as gf

BROKER = "broker.hivemq.com"
PORT = 1883
TOPIC = "ME193/Rogers/MiaGait"
CLIENT_ID = "msuno-gait-" + uuid.uuid4().hex[:6]
SMOOTH = 3              # majority vote over the last 3 decisions (1.5 s) so the figure doesn't flicker
DEFAULT_HZ = {"walking": 0.9, "running": 1.45, "biking": 1.2}   # stride rate when the signal isn't clearly periodic
FRAMES_PER_STRIDE = 4   # the sketch's animations

try:
    MODEL_FILE = Path(__file__).resolve().with_name("model.json")
except NameError:       # no __file__ when pasted into a console
    MODEL_FILE = Path("model.json")

model = None
window = deque(maxlen=gf.WINDOW)
since_decision = 0
next_seq = None
votes1, votes2 = deque(maxlen=SMOOTH), deque(maxlen=SMOOTH)
shown = None
warned_noimu = False


def load_model(text, source):
    global model
    try:
        m = json.loads(text)
        m["model1"]["tree"]["left"], m["model2"]["tree"]["left"]   # make sure it has what we need
        model = m
        votes1.clear()
        votes2.clear()
        print(f"Model loaded from {source}: model 1 uses {m['model1']['feature']}, "
              f"model 2 uses {len(m['model2']['features'])} features")
        return True
    except Exception as e:
        print(f"Bad model from {source}: {e}")
        return False


if MODEL_FILE.exists():
    load_model(MODEL_FILE.read_text(), MODEL_FILE.name)
else:
    print("No model.json yet: streaming data only (LED shows ? ?). Record data, then run train.py --send")


def on_connect(client, userdata, flags, reason_code, properties):
    print(f"MQTT connected ({reason_code}), publishing on {TOPIC}/raw and {TOPIC}/pred")
    client.subscribe(TOPIC + "/model", qos=1)


def on_message(client, userdata, msg):
    text = msg.payload.decode()
    if load_model(text, "the laptop"):
        try:
            MODEL_FILE.write_text(text)
        except OSError as e:
            print(f"Could not save model.json: {e}")


client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id=CLIENT_ID)
client.on_connect = on_connect
client.on_message = on_message
client.connect_async(BROKER, PORT, keepalive=30)   # keeps retrying if Wi-Fi isn't up yet
client.loop_start()


def show(activity, stride_hz=1.0):
    """activity: index into CLASSES, 4 = no model yet, 5 = no IMU. Resent only when it changes
    or the pace changes by more than 15 %."""
    global shown
    ms = int(1000 / (max(stride_hz, 0.2) * FRAMES_PER_STRIDE))
    if shown is None or shown[0] != activity or abs(ms - shown[1]) > 0.15 * shown[1]:
        Bridge.call("show", activity, ms)
        shown = (activity, ms)


def decide():
    f = gf.features(list(window))
    if model is None:
        show(4)
        client.publish(TOPIC + "/pred", json.dumps({"m1": None, "m2": None}))
        return
    votes1.append(gf.predict(model["model1"]["tree"], f))
    votes2.append(gf.predict(model["model2"]["tree"], f))
    m1 = Counter(votes1).most_common(1)[0][0]
    m2 = Counter(votes2).most_common(1)[0][0]
    hz = f["cadence_hz"] if f["cadence_hz"] > 0.3 and f["periodicity"] > 0.3 else DEFAULT_HZ.get(m2, 1.0)
    show(gf.CLASSES.index(m2), hz)
    k = model["model1"]["feature"]
    client.publish(TOPIC + "/pred", json.dumps({
        "m1": m1, "m2": m2, "raw1": votes1[-1], "raw2": votes2[-1],
        "peak": round(f[k], 3), "peak_name": k,
        "cadence": round(f["cadence_hz"], 2), "periodicity": round(f["periodicity"], 2),
        "gyro": round(f["mean_gmag"], 1)}))


def loop():
    global since_decision, next_seq, warned_noimu
    time.sleep(0.1)
    reply = Bridge.call("imu")
    if reply == "NOIMU":
        if not warned_noimu:
            print("No MPU-6050 found on Wire1 (Qwiic) or Wire (A4/A5) at 0x68/0x69. Check the wiring and restart the app.")
            warned_noimu = True
            show(5)
        return
    parts = reply.split(";")
    seq = int(parts[0])
    samples = [gf.to_units([int(v) for v in p.split(",")]) for p in parts[1:]]
    if not samples:
        return
    if next_seq is not None and seq != next_seq:
        print(f"Lost {seq - next_seq} samples")
    next_seq = seq + len(samples)
    client.publish(TOPIC + "/raw", json.dumps({"seq": seq, "s": [[round(v, 3) for v in s] for s in samples]}))
    for s in samples:
        window.append(s)
        since_decision += 1
        if len(window) == gf.WINDOW and since_decision >= gf.HOP:
            since_decision = 0
            decide()


App.run(user_loop=loop)
