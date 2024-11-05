import asyncio
import os
import ssl
import logging
import paho.mqtt.client as mqtt
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)

# MQTT Broker details from environment variables
broker = os.getenv('MQTT_BROKER', 'mqtt.example.com')
port = int(os.getenv('MQTT_PORT', '8883'))
topic = os.getenv('MQTT_TOPIC', 'home/device/control')
response_topic = os.getenv('MQTT_RESPONSE_TOPIC', 'home/device/response')

# Define the message to send to the IoT device
message = "TURN_ON"

# Callback function when the client receives a CONNACK response from the server
async def on_connect(client, userdata, flags, rc):
    if rc == 0:
        logging.info("Connected successfully")
        await client.subscribe(response_topic)
    else:
        logging.error(f"Connection failed with code {rc}")

# Callback function when a PUBLISH message is received from the server
async def on_message(client, userdata, msg):
    logging.info(f"Received message '{msg.payload.decode()}' on topic '{msg.topic}'")

# Create an asynchronous MQTT client instance
client = mqtt.Client(asyncio=True)

# Set up TLS/SSL with certificate-based mutual authentication
client.tls_set(
    ca_certs="/path/to/ca.crt",
    certfile="/path/to/client.crt",
    keyfile="/path/to/client.key",
    tls_version=ssl.PROTOCOL_TLS
)

# Assign the callback functions
client.on_connect = on_connect
client.on_message = on_message

async def main():
    try:
        # Connect to the MQTT broker
        await client.connect(broker, port, 60)
        # Publish the message to the specified topic
        await client.publish(topic, message)
        logging.info(f"Message '{message}' sent to topic '{topic}'")
        # Start the loop to process network traffic and dispatch callbacks
        await asyncio.get_event_loop().run_forever()
    except Exception as e:
        logging.error(f"An error occurred: {e}")
    finally:
        # Disconnect from the broker
        await client.disconnect()

if __name__ == "__main__":
    asyncio.run(main())
