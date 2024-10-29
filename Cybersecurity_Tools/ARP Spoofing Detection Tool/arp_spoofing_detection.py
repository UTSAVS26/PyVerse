import scapy.all as scapy
import tkinter as tk
import time
import logging
from threading import Thread

# Initialize logging
logging.basicConfig(filename="arp_spoofing_log.txt", level=logging.INFO, format="%(asctime)s - %(message)s")

monitoring = False

# Function to scan network and get ARP table
def get_arp_table():
    arp_request = scapy.ARP(pdst="192.168.1.1/24")  # Replace with your network range
    broadcast = scapy.Ether(dst="ff:ff:ff:ff:ff:ff")
    arp_request_broadcast = broadcast / arp_request
    answered_list = scapy.srp(arp_request_broadcast, timeout=1, verbose=False)[0]

    arp_table = {}
    for element in answered_list:
        arp_table[element[1].psrc] = element[1].hwsrc  # Mapping IP -> MAC

    return arp_table

# Function to detect ARP spoofing
def detect_spoofing(arp_table, previous_arp_table):
    for ip in arp_table:
        if ip in previous_arp_table:
            if arp_table[ip] != previous_arp_table[ip]:
                logging.warning(f"{time.ctime()} - IP: {ip} has changed MAC address from {previous_arp_table[ip]} to {arp_table[ip]}")
                alert_user(ip, f"IP: {ip} changed MAC from {previous_arp_table[ip]} to {arp_table[ip]}")
        else:
            logging.info(f"{time.ctime()} - New device detected: IP: {ip} with MAC: {arp_table[ip]}")
            alert_user(ip, f"New device detected: IP {ip} with MAC {arp_table[ip]}")

# Function to display alerts in GUI
def alert_user(ip, message):
    alert_label.config(text=f"ARP Spoofing Alert: {message}")
    alert_label.after(5000, lambda: alert_label.config(text=""))  # Clear message after 5 seconds

# Main function to start ARP monitoring in a separate thread
def start_monitoring():
    global monitoring
    monitoring = True
    previous_arp_table = {}

    alert_label.config(text="Monitoring started...")
    
    while monitoring:
        arp_table = get_arp_table()
        detect_spoofing(arp_table, previous_arp_table)
        previous_arp_table = arp_table
        time.sleep(10)  # Monitor every 10 seconds

# Function to stop monitoring
def stop_monitoring():
    global monitoring
    monitoring = False
    alert_label.config(text="Monitoring stopped.")

# Function to start monitoring in a separate thread
def run_monitoring():
    monitoring_thread = Thread(target=start_monitoring)
    monitoring_thread.start()

# Setup tkinter GUI
root = tk.Tk()
root.title("ARP Spoofing Detection Tool")

start_button = tk.Button(root, text="Start Monitoring", command=run_monitoring)
start_button.pack(pady=10)

stop_button = tk.Button(root, text="Stop Monitoring", command=stop_monitoring)
stop_button.pack(pady=10)

alert_label = tk.Label(root, text="")
alert_label.pack(pady=20)

root.mainloop()
