from scapy.all import sniff, IP, TCP, UDP
import logging

# Setup logging to a file
logging.basicConfig(filename='packet_log.txt', level=logging.INFO, format='%(asctime)s - %(message)s')

# Function to process captured packets
def packet_callback(packet):
    if IP in packet:
        ip_src = packet[IP].src
        ip_dst = packet[IP].dst
        proto = "TCP" if TCP in packet else "UDP" if UDP in packet else "Other"
        port_src = packet[TCP].sport if TCP in packet else packet[UDP].sport if UDP in packet else None
        port_dst = packet[TCP].dport if TCP in packet else packet[UDP].dport if UDP in packet else None
        
        # Log packet details
        log_message = f"Protocol: {proto}, Source IP: {ip_src}, Source Port: {port_src}, Destination IP: {ip_dst}, Destination Port: {port_dst}"
        print(log_message)
        logging.info(log_message)

# Function to start packet capturing
def start_sniffing():
    print("Starting the packet analyzer...")
    sniff(prn=packet_callback, store=0)

# Function to stop packet capturing gracefully
def stop_sniffing():
    print("Stopping the packet analyzer...")
    # Assuming a mechanism to stop the sniffing; could use a threading event or similar approach.
    # Here we'll just exit for simplicity.
    exit()

# Main function
def main():
    print("Welcome to the Network Packet Analyzer!")
    try:
        start_sniffing()
    except KeyboardInterrupt:
        stop_sniffing()

if __name__ == "__main__":
    main()
