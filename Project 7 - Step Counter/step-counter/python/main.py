# Step Counter - UNO Q Linux side.
#
# Every 0.1 s: collect new samples from the sketch (50 Hz: MPU-6050 accel + gyro and the
# HC-SR04 distance) and stream them to the laptop over MQTT (ME193/Rogers/MiaSteps/raw).
# The laptop records them (record.py) or counts steps with the neural net (count_live.py),
# and sends the count back on ME193/Rogers/MiaSteps/count, which goes on the LED matrix.
from arduino.app_utils import *
import paho.mqtt.client as mqtt
import json, time, uuid

BROKER = "broker.emqx.io"
PORT = 1883
TOPIC = "ME193/Rogers/MiaSteps"
CLIENT_ID = "msuno-steps-" + uuid.uuid4().hex[:6]
ACCEL_SCALE = 2048.0   # counts per g     (MPU-6050 at +-16 g)
GYRO_SCALE = 16.4      # counts per deg/s (MPU-6050 at +-2000 deg/s)

next_seq = None
warned_noimu = False
pending_count = None   # newest count from the laptop, shown from the main loop
last_echo = time.time()
warned_echo = False


def to_units(raw):
    """Raw counts -> ax, ay, az (g), gx, gy, gz (deg/s), distance (m, 0 = no echo)."""
    return [round(v / ACCEL_SCALE, 3) for v in raw[:3]] + [round(v / GYRO_SCALE, 3) for v in raw[3:6]] + [raw[6] / 1000]


def on_connect(client, userdata, flags, reason_code, properties):
    print(f"MQTT connected ({reason_code}), streaming on {TOPIC}/raw")
    client.subscribe(TOPIC + "/count")


def on_message(client, userdata, msg):
    global pending_count
    try:
        pending_count = int(json.loads(msg.payload)["steps"])
    except Exception as e:
        print(f"Bad count message: {e}")


client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id=CLIENT_ID)
client.on_connect = on_connect
client.on_message = on_message
client.connect_async(BROKER, PORT, keepalive=30)   # keeps retrying if Wi-Fi isn't up yet
client.loop_start()


def loop():
    global next_seq, warned_noimu, pending_count, last_echo, warned_echo
    time.sleep(0.1)
    if pending_count is not None:
        Bridge.call("count", pending_count)
        pending_count = None
    reply = Bridge.call("imu")
    if reply == "NOIMU":
        if not warned_noimu:
            print("No MPU-6050 found on Wire1 (Qwiic) or Wire (A4/A5) at 0x68/0x69. Check the wiring and restart the app.")
            warned_noimu = True
        return
    parts = reply.split(";")
    seq = int(parts[0])
    samples = [to_units([int(v) for v in p.split(",")]) for p in parts[1:]]
    if not samples:
        return
    if next_seq is not None and seq != next_seq:
        print(f"Lost {seq - next_seq} samples")
    next_seq = seq + len(samples)
    if any(s[6] > 0 for s in samples):
        last_echo = time.time()
        warned_echo = False
    elif time.time() - last_echo > 10 and not warned_echo:
        print("No ultrasonic echo for 10 s: check the HC-SR04 wiring (Trig D2, Echo D3 through a divider) or what it points at")
        warned_echo = True
    client.publish(TOPIC + "/raw", json.dumps({"seq": seq, "s": samples}))


App.run(user_loop=loop)
