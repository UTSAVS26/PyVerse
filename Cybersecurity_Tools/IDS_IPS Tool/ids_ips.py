import os
import platform
import logging
from scapy.all import sniff, IP, TCP, UDP, IPv6
from collections import defaultdict
import subprocess

# Set up logging to record alerts to a file
logging.basicConfig(filename='ids_ips_alerts.log', level=logging.INFO, format='%(asctime)s - %(message)s')

# Suspicious keywords in packet payloads
SUSPICIOUS_KEYWORDS = ["attack", "malicious", "exploit", "payload"]

# List of blocked IPs and a whitelist of safe IPs
BLOCKED_IPS = set()
WHITELIST_IPS = {"192.168.1.1", "8.8.8.8"}  # Trusted IP addresses

# Threshold for blocking: block IP after 'n' suspicious packets
THRESHOLD = 3
ip_suspicion_count = defaultdict(int)  # Store how many suspicious packets are seen from each IP

# Function to block IP addresses (supports Linux, macOS, Windows)
def block_ip(ip_address):
    if ip_address in WHITELIST_IPS:
        logging.info(f"Skipping whitelist IP: {ip_address}")
        return
    
    if ip_address not in BLOCKED_IPS:
        print(f"[IPS] Blocking IP address: {ip_address}")
        system_name = platform.system()
        try:
            if system_name == "Linux":
                # Block with iptables (Linux)
                subprocess.run(["sudo", "iptables", "-A", "INPUT", "-s", ip_address, "-j", "DROP"], check=True)
            elif system_name == "Darwin":
                # Block with pf (macOS)
                subprocess.run(["sudo", "pfctl", "-t", "blocklist", "-T", "add", ip_address], check=True)
            elif system_name == "Windows":
                # Block with Windows Firewall
                subprocess.run(["netsh", "advfirewall", "firewall", "add", "rule", 
                                f"name=Block IP {ip_address}", "dir=in", "action=block", 
                                f"remoteip={ip_address}"], check=True)
            else:
                print(f"Unsupported OS: {system_name}. Cannot block IP.")
                return
        except subprocess.CalledProcessError as e:
            print(f"Error while blocking IP {ip_address}: {e}")
        else:
            BLOCKED_IPS.add(ip_address)
            logging.info(f"Blocked IP: {ip_address}")

# Function to process each packet for IDS/IPS
def packet_callback(packet):
    # Detect IP or IPv6 packets
    if IP in packet or IPv6 in packet:
        ip_src = packet[IP].src if IP in packet else packet[IPv6].src
        ip_dst = packet[IP].dst if IP in packet else packet[IPv6].dst

        # Check for TCP or UDP layer to get payload
        if TCP in packet or UDP in packet:
            payload = str(packet[TCP].payload) if TCP in packet else str(packet[UDP].payload)
            
            # Check for suspicious keywords in the payload
            for keyword in SUSPICIOUS_KEYWORDS:
                if keyword in payload:
                    alert_message = f"[ALERT] Suspicious packet detected from {ip_src} to {ip_dst}: {payload}"
                    print(alert_message)
                    logging.info(alert_message)
                    
                    # Increment the count for the suspicious activity from the source IP
                    ip_suspicion_count[ip_src] += 1
                    
                    # Block IP if suspicion count exceeds the threshold
                    if ip_suspicion_count[ip_src] >= THRESHOLD:
                        block_ip(ip_src)
                    break

# Start sniffing network packets
print("Starting IDS/IPS... Press Ctrl+C to stop.")
sniff(prn=packet_callback, store=0)  # `store=0` means do not keep packets in memory
