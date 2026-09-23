from mqttlib import MQTTClient

TOPIC = "ME193"

with MQTTClient() as client:
    print(f"Connected. Type a message and press Enter to send it on '{TOPIC}'.")
    print("Type 'quit' to stop.")

    while True:
        message = input("2")
        if message == "quit":
            break
        client.publish(TOPIC, message)
