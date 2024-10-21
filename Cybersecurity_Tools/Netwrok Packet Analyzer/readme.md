# Network Packet Analyzer

## Description
The **Network Packet Analyzer** is a Python-based tool that captures and analyzes network packets in real-time using the Scapy library. It logs detailed information about TCP and UDP packets, including source and destination IP addresses and ports. This tool is useful for network monitoring, troubleshooting, and educational purposes.

## Features
- Captures network packets in real-time.
- Logs packet details (protocol, source IP, source port, destination IP, destination port) to a text file.
- Supports TCP and UDP protocols.
- Simple command-line interface for starting and stopping packet capture.

## Prerequisites
- Python 3.x
- Scapy library

### Installation
You can install the required Scapy library using pip:

```bash
pip install scapy
```

## Running the Packet Analyzer
1. **Clone the repository** (if applicable) or save the script as `packet_analyzer.py`.

2. **Open a terminal** and navigate to the directory where the script is saved.

3. **Run the script** using the following command:

   ```bash
   python packet_analyzer.py
   ```

4. **Start capturing packets**. The program will display captured packet details in the console and log them to `packet_log.txt`.

5. **Stop the packet capture** by pressing `Ctrl + C`. The program will exit gracefully.

## Log File
All captured packet details will be logged in the `packet_log.txt` file located in the same directory as the script. You can review this file to analyze the captured network traffic later.

## Usage Example
Once the script is running, you might see output similar to the following:

```
Protocol: TCP, Source IP: 192.168.1.5, Source Port: 54321, Destination IP: 93.184.216.34, Destination Port: 80
```

This indicates a TCP packet sent from the source IP `192.168.1.5` on port `54321` to the destination IP `93.184.216.34` on port `80`.
