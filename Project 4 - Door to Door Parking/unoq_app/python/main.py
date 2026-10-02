# Project 4 - UNO Q Linux side (runs in Arduino App Lab)
# Receives the minifig box from minifig_mqtt.py over MQTT, shows a dot on the LED matrix,
# and drives forward/backward until the minifig is in the middle of the camera image.
#
# Expected message (topic ME193/minifig/tashamia), pixels:
#   {"x": 412, "y": 260, "width": 85, "height": 140, "img_w": 1280, "img_h": 720}
# img_w / img_h are optional; FRAME_W / FRAME_H below are used if they're missing.
from arduino.app_utils import *
import paho.mqtt.client as mqtt
import json, time, threading

# ---- MQTT ----
BROKER = "test.mosquitto.org"
PORT = 1883
TOPIC = f"ME193/minifig/tashamia"    # must match minifig_mqtt.py
CLIENT_ID = "ME193-minifig-car1"     # unique per car

# Fallback camera size if the message doesn't include img_w / img_h
FRAME_W = 1280
FRAME_H = 720

# ---- Display ----
FLIP_X = False
FLIP_Y = True

# ---- Driving ----
TARGET_X = 0.5      # where to stop (0.5 = middle of the screen)
DEADBAND = 0.04     # "close enough" -> stop
RESUME = 0.08       # once stopped, only move again if it drifts this far (prevents twitching)
DIRECTION = 1       # set to -1 if the car drives AWAY from the center
MIN_SPEED = 110     # slowest PWM that actually moves the car
MAX_SPEED = 170     # keep low at first - camera lag causes overshoot
GAIN = 600          # speed per unit of error
TIMEOUT = 0.5       # no message for this long -> stop

latest = None       # (x, y, time_received), x/y normalized 0-1
lock = threading.Lock()


def on_connect(client, userdata, flags, reason_code, properties):
    print(f"MQTT connected ({reason_code}), subscribing to {TOPIC}")
    client.subscribe(TOPIC)


def on_message(client, userdata, msg):
    global latest
    try:
        d = json.loads(msg.payload)
        w = float(d.get("img_w", FRAME_W))
        h = float(d.get("img_h", FRAME_H))
        x, y = float(d["x"]), float(d["y"])
        if x > 1 or y > 1:           # pixels -> normalize
            x, y = x / w, y / h
        with lock:
            latest = (x, y, time.time())
    except Exception as e:
        print("Bad message:", msg.payload, e)


client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id=CLIENT_ID)
client.on_connect = on_connect
client.on_message = on_message
client.connect(BROKER, PORT, keepalive=30)
client.loop_start()

last_cell = None
stopped = True
last_mode = None


def loop():
    global last_cell, stopped, last_mode
    with lock:
        cur = latest

    if cur is None or time.time() - cur[2] > TIMEOUT:
        speed, cell, mode, x = 0, (-1, -1), "NO MINIFIG -> STOP", None
        stopped = True
    else:
        x, y, _ = cur
        x = min(max(x, 0.0), 1.0)
        y = min(max(y, 0.0), 1.0)
        dx = 1 - x if FLIP_X else x
        dy = 1 - y if FLIP_Y else y
        cell = (round(dx * 12), round(dy * 7))

        error = x - TARGET_X
        limit = RESUME if stopped else DEADBAND
        if abs(error) <= limit:
            speed, mode, stopped = 0, "CENTERED -> STOP", True
        else:
            stopped = False
            mag = min(MAX_SPEED, max(MIN_SPEED, int(abs(error) * GAIN)))
            speed = DIRECTION * (mag if error < 0 else -mag)
            mode = "FORWARD" if speed > 0 else "BACKWARD"

    Bridge.call("drive", speed, speed)     # resent every loop so the sketch watchdog stays fed

    if cell != last_cell:
        Bridge.call("dot", cell[0], cell[1])
        last_cell = cell

    if mode != last_mode:                  # print only when the behavior changes
        print(mode if x is None else f"{mode}  (x={x:.2f}, speed={speed})")
        last_mode = mode

    time.sleep(0.05)                        # ~20 updates per second


App.run(user_loop=loop)
