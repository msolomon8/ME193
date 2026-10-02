"""Publish FAKE minifig boxes (same format as minifig_mqtt.py) to test the UNO Q without a camera.

    python test_publish.py      (Ctrl+C to stop)
"""
import json, math, sys, time
from pathlib import Path

ME193 = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ME193 / "MQTT"))
from mqttlib import MQTTClient

TOPIC = "ME193/minifig"
W, H = 1280, 720          # pretend camera size

with MQTTClient() as client:
    print(f"Publishing fake boxes to '{TOPIC}'. Ctrl+C to stop.")
    t0 = time.time()
    try:
        while True:
            t = time.time() - t0
            x = (t % 4) / 4 * W                      # sweep left -> right every 4 s
            y = (0.5 + 0.4 * math.sin(t * 1.5)) * H  # bob up and down
            msg = json.dumps({"x": round(x), "y": round(y), "width": 85, "height": 140,
                              "img_w": W, "img_h": H})
            client.publish(TOPIC, msg)
            print(msg)
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("Stopped. The dot should disappear on the UNO Q within ~0.5 s.")
