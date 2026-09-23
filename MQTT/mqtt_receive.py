import time

from mqttlib import MQTTClient

TOPIC = "ME193"


def on_message(topic, payload):
    print(f"[{topic}] {payload}")


with MQTTClient() as client:
    client.subscribe(TOPIC, on_message)
    print(f"Listening on '{TOPIC}'. Press Ctrl+C to stop.")

    while True:
        time.sleep(1)
