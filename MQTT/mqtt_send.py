"""Send the "start" command over MQTT on the ME193/Rogers topic.

Usage:
    ../le-venv/bin/python mqtt_send.py

Connects to the broker, publishes "start" once, waits until the broker has
it, then exits.
"""

from mqttlib import MQTTClient

TOPIC = "ME193/Rogers"
MESSAGE = "start"

with MQTTClient() as client:
    client.publish(TOPIC, MESSAGE).wait_for_publish(timeout=10)
    print(f"Sent '{MESSAGE}' on '{TOPIC}'")
