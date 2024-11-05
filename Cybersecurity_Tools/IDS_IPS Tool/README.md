## **Intrusion Detection Systems (IDS) and Intrusion Prevention Systems (IPS) tool**

### 🎯 **Goal**

The goal of the IDS/IPS network packet sniffer code is to provide a simple but effective tool for monitoring network traffic in real-time, detecting potentially malicious or suspicious activity, and automatically responding by blocking the offending IP addresses.

### 🧾 **Description**

This project is a simple Intrusion Detection System (IDS) and Intrusion Prevention System (IPS) implemented in Python using the Scapy library. The tool monitors network traffic in real-time and checks for suspicious activity based on predefined keywords within packet payloads. If a certain number of suspicious packets are detected from the same IP address, the tool can automatically block the offending IP address using system-level firewall rules.

Key Features:

**Real-Time Packet Monitoring:** Continuously monitors incoming network traffic for both IPv4 and IPv6 packets.
- **Keyword Detection:** Scans TCP/UDP packet payloads for predefined suspicious keywords like "attack", "exploit", etc.
- **IP Blocking Mechanism:** Automatically blocks an IP address after it sends a threshold number of suspicious packets.
- **Cross-Platform Support:** Uses platform-specific firewall tools:
  - **Linux:** Blocks IPs using `iptables`.
  - **macOS:** Blocks IPs using `pfctl`.
  - **Windows:** Blocks IPs using the built-in firewall via `netsh`.
- **Whitelisting Support:** Allows trusted IP addresses to be whitelisted and protected from blocking.
- **Logging:** Records all detected alerts and blocked IPs to a log file (`ids_ips_alerts.log`) for auditing purposes.

### 📚 **Libraries Needed**

To run this project, you'll need the following libraries installed:

- **Scapy:** For packet sniffing and network protocol manipulation.
- **subprocess (built-in):** For running system commands to block IPs.
- **collections (built-in):** For managing a counter of suspicious activities by IP.

### 📢 **Conclusion**

This project serves as a basic demonstration of a real-time IDS/IPS tool that can detect suspicious network activities based on packet content and apply system-level firewall rules to block malicious traffic. While it's a simplified example, it can be extended with more advanced packet inspection techniques and integrated into larger network security systems.



