## IoT Device Control Using MQTT and Python

This project demonstrates how to control IoT devices using the MQTT protocol with Python. It uses the `paho-mqtt` library to publish messages to an MQTT broker, which are then used to control an IoT device.

## Compatibility
**IoT Devices Compatible with MQTT:**
1. **Smart Lights:** Devices like Philips Hue or any other smart lighting system that can connect to an MQTT broker can be controlled to turn on/off, change color, or adjust brightness.
2. **Smart Thermostats:** Devices like the Nest Thermostat or Ecobee can use MQTT for setting temperature, scheduling, or mode changes.
3. **Home Security Systems:** Sensors and alarms that support MQTT can be armed/disarmed and monitored through the script.
4. **Smart Plugs:** Devices like TP-Link HS100 can be turned on/off remotely using MQTT commands.
5. **Industrial Sensors:** Many industrial IoT devices like temperature sensors, pressure sensors, or flow meters that support MQTT can be monitored and controlled.
6. **Agricultural Sensors:** IoT devices used in farming to monitor soil moisture, weather conditions, or control irrigation systems.

## Example IoT-Based Projects:
1. **Home Automation System:**
- Project Description: Create a centralized system to control various smart home devices such as lights, thermostats, and security cameras.
- MQTT Usage: Use MQTT to send commands to different devices and receive their statuses.
2. **Remote Monitoring System:**
- Project Description: Set up a system to monitor environmental parameters like temperature, humidity, and air quality in real-time.
- MQTT Usage: Sensors publish their data to specific topics, and the central system subscribes to these topics to collect and analyze data.
3. **Smart Farming Solution:**
- Project Description: Implement a solution for monitoring and controlling agricultural environments, optimizing water usage, and managing crop health.
- MQTT Usage: Use MQTT to communicate with soil moisture sensors, weather stations, and irrigation controllers.
4. **Industrial Automation:**
- Project Description: Develop a system for monitoring and controlling industrial machines, ensuring efficient operations and safety.
- MQTT Usage: Machines and sensors communicate their status and receive control commands over MQTT.
5. **Health Monitoring System:**
- Project Description: A system for remote health monitoring of patients, tracking vital signs and providing alerts.
- MQTT Usage: Wearable devices send patient data to a central server via MQTT, which then processes and responds to critical alerts.

## Features

- **Secure Communication**: Utilizes TLS/SSL for secure communication with the MQTT broker.
- **Dynamic Topic Management**: Supports dynamic subscription and publishing to topics.
- **Enhanced Error Handling**: Robust error handling for connection issues and message delivery failures.
- **Logging**: Detailed logging for monitoring and troubleshooting.

## Prerequisites

Before you begin, ensure you have the following installed:
- Python 3.6 or higher
- pip (Python package installer)

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/iot-device-control.git
   cd Iot_control
   ```
2. Install the required Python packages:
```bash
pip install paho-mqtt python-dotenv
```
## Configuration
1. Update the .env file with your MQTT broker details:
```bash
MQTT_BROKER=mqtt.example.com
MQTT_PORT=8883
MQTT_TOPIC=home/device/control
MQTT_RESPONSE_TOPIC=home/device/response
```
2. Ensure your MQTT broker is configured to support TLS/SSL and is accessible from your network.
## Usage
To run the script, use the following command:
```bash
python iot_control.py
```
