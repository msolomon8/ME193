"""Print everything the ping-pong game posts over MQTT:
    ME193/Rogers/Mia       your record streak as a float (the assignment's score)
    ME193/Rogers/Mia/game  readable updates: score, streak, YOU WIN! / YOU LOSE!

Run it in a second terminal while you play:
    le-venv/bin/python "Midterm - Ping Pong/watch_score.py"

Press Ctrl+C to stop.
"""

import sys
import time
from pathlib import Path

# mqttlib.py lives in ME193/MQTT/
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "MQTT"))
from mqttlib import MQTTClient

TOPIC = "ME193/Rogers/Mia"             # must match MQTT_TOPIC in ping_pong.py
GAME_TOPIC = "ME193/Rogers/Mia/game"   # must match MQTT_GAME_TOPIC


def on_score(topic, payload):
    print(f"{time.strftime('%H:%M:%S')}  [{topic}]  record streak = {payload}", flush=True)


def on_game(topic, payload):
    print(f"{time.strftime('%H:%M:%S')}  [{topic}]  {payload}", flush=True)


with MQTTClient() as client:
    # Wait for the broker to accept the connection first; paho drops a subscribe
    # sent before that.
    for _ in range(50):
        if client._client.is_connected():
            break
        time.sleep(0.2)
    else:
        sys.exit("Couldn't connect to the MQTT broker (test.mosquitto.org). Try again.")
    client.subscribe(TOPIC, on_score)
    client.subscribe(GAME_TOPIC, on_game)
    print(f"Listening on '{TOPIC}' and '{GAME_TOPIC}'. Play a game; updates print here. "
          "Ctrl+C to stop.", flush=True)
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass
