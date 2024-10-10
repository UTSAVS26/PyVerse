from scapy.all import sniff, IP, TCP, UDP, Raw

def analyze_packet(packet):
    # Check for the presence of the IP layer
    if IP in packet:
        ip_layer = packet[IP]  # Extract the IP layer
        src_ip = ip_layer.src  # Source IP address
        dst_ip = ip_layer.dst  # Destination IP address
        protocol = ip_layer.proto  # Protocol number

        # Display packet information
        print(f"--- Packet Captured ---")
        print(f"Source IP: {src_ip}")
        print(f"Destination IP: {dst_ip}")
        print(f"Protocol: {protocol}")

        # Check if the packet is TCP
        if TCP in packet:
            print("Protocol: TCP")
            tcp_payload = packet[TCP].payload
            print(f"TCP Payload: {tcp_payload}")
        
        # Check if the packet is UDP
        elif UDP in packet:
            print("Protocol: UDP")
            udp_payload = packet[UDP].payload
            print(f"UDP Payload: {udp_payload}")

        # Check for raw payload data
        if Raw in packet:
            raw_payload = packet[Raw].load
            print(f"Raw Payload: {raw_payload}")

        print("--------------------------\n")

def main():
    print("Starting packet sniffer... Press Ctrl+C to stop.")
    # Start sniffing packets and process each with the analyze_packet function
    sniff(prn=analyze_packet, store=0)

if __name__ == "__main__":
    main()
