import paho.mqtt.client as mqtt

DEFAULT_BROKER = "test.mosquitto.org"
DEFAULT_PORT = 1883


class MQTTClient:
    """Thin wrapper around paho-mqtt for easy pub/sub against a public broker."""

    def __init__(self, broker=DEFAULT_BROKER, port=DEFAULT_PORT, keepalive=60):
        self.broker = broker
        self.port = port
        self.keepalive = keepalive
        self._callbacks = {}
        self._client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
        self._client.on_message = self._on_message

    def _on_message(self, client, userdata, msg):
        callback = self._callbacks.get(msg.topic)
        if callback:
            callback(msg.topic, msg.payload.decode("utf-8", errors="replace"))

    def connect(self):
        self._client.connect(self.broker, self.port, self.keepalive)
        self._client.loop_start()
        return self

    def disconnect(self):
        self._client.loop_stop()
        self._client.disconnect()

    def subscribe(self, topic, callback):
        self._callbacks[topic] = callback
        self._client.subscribe(topic)

    def publish(self, topic, payload):
        self._client.publish(topic, payload)

    def __enter__(self):
        return self.connect()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.disconnect()
        return False
