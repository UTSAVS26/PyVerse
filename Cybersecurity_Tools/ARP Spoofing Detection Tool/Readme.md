### ARP Spoofing Detection Tool: README

#### Overview:
This tool detects ARP spoofing attempts on a local network by monitoring the ARP table for suspicious changes in IP-MAC mappings. When such changes are detected, the user is alerted via a tkinter-based GUI, and a log of the incident is saved in a file named `arp_spoofing_log.txt`. It works by continuously scanning the ARP table every 10 seconds.

#### Features:
- **ARP Spoofing Detection**: Scans your network’s ARP table for IP-MAC mismatches and logs suspicious activities.
- **Real-time Monitoring**: Alerts the user when changes in the ARP table are detected.
- **Logging**: Logs ARP spoofing attempts along with timestamps into `arp_spoofing_log.txt`.
- **User-Friendly GUI**: Uses tkinter for simple control (Start/Stop) and alerts.

---

### Prerequisites:
#### For Windows:
1. **Install Python**: Download and install Python from [python.org](https://www.python.org/downloads/).
2. **Install Scapy**: Run the following command to install Scapy:
   ```bash
   pip install scapy
   ```
3. **Install Npcap**: Scapy requires Npcap to capture network packets on Windows. Download and install from [Npcap](https://nmap.org/npcap/).

#### For macOS/Linux:
1. **Install Python**: Ensure Python is installed (usually comes pre-installed on macOS/Linux). If not, install via [python.org](https://www.python.org/downloads/).
2. **Install Scapy**: Run the following command:
   ```bash
   pip install scapy
   ```
3. **Run with Sudo**: Scapy needs root permissions on macOS/Linux to capture network traffic, so the tool must be run using `sudo`.

---

### How to Run the Tool:
#### Windows:
1. **Run the Program**:
   - Open the command prompt, navigate to the folder containing the script, and execute:
     ```bash
     python arp_spoofing_detection.py
     ```
2. **Start Monitoring**: Click the "Start Monitoring" button on the GUI to begin.
3. **Stop Monitoring**: Click the "Stop Monitoring" button to stop.
4. **Check Logs**: After spoofing detection, check the `arp_spoofing_log.txt` file for any recorded incidents.

#### macOS/Linux:
1. **Run the Program with sudo**:
   - Open a terminal, navigate to the folder, and run:
     ```bash
     sudo python3 arp_spoofing_detection.py
     ```
2. **Start and Stop Monitoring**: Use the GUI buttons to start and stop monitoring.
3. **Logs**: View the `arp_spoofing_log.txt` file for any alerts and details.

---

### How to Simulate ARP Spoofing:
To ensure the tool is detecting ARP spoofing correctly, you can simulate an ARP spoofing attack using tools like **Bettercap** or **Ettercap**. These tools allow you to manipulate ARP requests and simulate man-in-the-middle (MITM) attacks on a network.

- **Bettercap**:
   ```bash
   sudo bettercap -T <target-ip> -X --proxy
   ```

- **Ettercap**:
   ```bash
   sudo ettercap -T -M arp:remote /<target-ip>/ /<gateway-ip>/
   ```

---

### Expected Logs:
When ARP spoofing is detected, the following content is logged into the `arp_spoofing_log.txt` file:

```
2024-10-17 12:34:56 - WARNING - IP: 192.168.1.10 has changed MAC address from 00:11:22:33:44:55 to 66:77:88:99:AA:BB
```

This indicates that the IP `192.168.1.10` was detected with a different MAC address than expected, which could indicate ARP poisoning or spoofing.

---

### Important Commands:
- **Windows**:
   - Install Python dependencies:
     ```bash
     pip install scapy
     ```
   - Run the script:
     ```bash
     python arp_spoofing_detection.py
     ```
   - To simulate ARP Spoofing, use tools like **Cain & Abel** or **Ettercap** on Windows.

- **macOS/Linux**:
   - Install Scapy:
     ```bash
     sudo pip3 install scapy
     ```
   - Run the script with root privileges:
     ```bash
     sudo python3 arp_spoofing_detection.py
     ```
   - Simulate ARP spoofing using **Bettercap** or **Ettercap**.

---

### Note:
The tool relies on network conditions and permissions. Ensure you are running the tool with the necessary permissions and on a network where you can capture packets to avoid false negatives.