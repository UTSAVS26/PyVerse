# author: Jivan Jamdar

import socket
from concurrent.futures import ThreadPoolExecutor
import paramiko

# Function to detect service running on open ports
def detect_service(port):
    services = {
        21: "FTP",
        22: "SSH",
        23: "Telnet",
        25: "SMTP",
        80: "HTTP",
        110: "POP3",
        143: "IMAP",
        443: "HTTPS",
        3306: "MySQL",
        5432: "PostgreSQL",
        3389: "RDP"
    }
    return services.get(port, "Unknown Service")

# Banner grabbing with SSH integration
def banner_grabbing(target_ip, port):
    try:
        if port == 22:  # SSH-specific banner grabbing
            banner = ssh_banner_grabbing(target_ip)
            return banner if banner else "No SSH banner received"

        # For non-SSH services
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(1)
        sock.connect((target_ip, port))

        # Send protocol-specific request based on common ports
        if port == 80 or port == 8080:  # HTTP
            sock.send(b"HEAD / HTTP/1.1\r\nHost: " + target_ip.encode() + b"\r\n\r\n")
        elif port == 21:  # FTP
            sock.send(b"USER anonymous\r\n")
        elif port == 25:  # SMTP
            sock.send(b"EHLO test\r\n")
        elif port == 110:  # POP3
            sock.send(b"USER test\r\n")
        elif port == 143:  # IMAP
            sock.send(b"TAG LOGIN test\r\n")
        else:

            sock.send(b"HEAD / HTTP/1.1\r\n\r\n") # HTTPS request for fallback

        banner = sock.recv(1024).decode().strip()
        sock.close()
        return banner if banner else "No banner received"
    except Exception as e:
        return f"Error grabbing banner: {e}"

# SSH-specific banner grabbing using paramiko
def ssh_banner_grabbing(target_ip):
    try:
        client = paramiko.SSHClient()
        client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        client.connect(target_ip, port=22, username='', password='', timeout=2)
        banner = client.get_transport().remote_version
        client.close()
        return banner
    except Exception as e:
        return f"SSH error: {e}"

# Function to scan a single port
def scan_port(target_ip, port):
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(1)
    result = sock.connect_ex((target_ip, port))

    if result == 0:
        service = detect_service(port)
        banner = banner_grabbing(target_ip, port)
        output = f"Port {port}:\n  - Service: {service}\n  - Banner: {banner}\n{'-'*40}"
        print(output)
    sock.close()
    return port if result == 0 else None

# Scanner with multithreading
def run_scanner(target_ip, start_port, end_port):
    print(f"\nStarting scan on {target_ip} (Ports {start_port} to {end_port})\n{'='*50}")

    with ThreadPoolExecutor(max_workers=100) as executor:
        futures = [executor.submit(scan_port, target_ip, port) for port in range(start_port, end_port + 1)]

        open_ports = [future.result() for future in futures if future.result()]

    if not open_ports:
        print("\nNo open ports found.\n")
    else:
        print(f"\nTotal Open Ports: {len(open_ports)}\n{'='*50}")

if __name__ == "__main__":
    target_ip = input("Enter target IP address: ")
    start_port = int(input("Enter start port: "))
    end_port = int(input("Enter end port: "))

    run_scanner(target_ip, start_port, end_port)
