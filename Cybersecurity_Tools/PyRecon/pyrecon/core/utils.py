"""
Utility functions for PyRecon core functionality.
"""

import ipaddress
import socket
import re
import os
from typing import List, Union, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed


def parse_target(target: str) -> List[str]:
    """
    Parse target specification into list of IP addresses.
    
    Args:
        target: Target specification (IP, domain, CIDR, or file path)
        
    Returns:
        List of IP addresses to scan
    """
    ips = []
    
    # Check if it's a file path
    if target.endswith('.txt') or os.path.exists(target):
        try:
            with open(target, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#'):
                        ips.extend(parse_target(line))
            return ips
        except FileNotFoundError:
            raise ValueError(f"Target file not found: {target}")
    
    # Check if it's a CIDR range
    if '/' in target:
        try:
            network = ipaddress.ip_network(target, strict=False)
            return [str(ip) for ip in network.hosts()]
        except ValueError:
            pass
    
    # Check if it's a domain name (not an IP address and not a file path)
    if not re.match(r'^\d+\.\d+\.\d+\.\d+$', target) and not os.path.exists(target):
        try:
            ip = socket.gethostbyname(target)
            return [ip]
        except socket.gaierror:
            raise ValueError(f"Could not resolve domain: {target}")
    
    # Single IP address
    try:
        ipaddress.ip_address(target)
        return [target]
    except ValueError:
        raise ValueError(f"Invalid IP address: {target}")


def parse_port_range(port_spec: str) -> List[int]:
    """
    Parse port specification into list of port numbers.
    
    Args:
        port_spec: Port specification (single port, range, list, or 'top-N')
        
    Returns:
        List of port numbers
    """
    ports = []
    
    # Handle 'top-N' specification
    if port_spec.startswith('top-'):
        try:
            n = int(port_spec[4:])
            return get_top_ports(n)
        except ValueError:
            raise ValueError(f"Invalid top ports specification: {port_spec}")
    
    # Handle comma-separated ports
    if ',' in port_spec:
        try:
            port_list = []
            for port_str in port_spec.split(','):
                port_str = port_str.strip()
                if '-' in port_str:
                    # Handle range within comma-separated list
                    start, end = map(int, port_str.split('-'))
                    if start < 1 or end > 65535 or start > end:
                        raise ValueError
                    port_list.extend(range(start, end + 1))
                else:
                    # Single port
                    port = int(port_str)
                    if port < 1 or port > 65535:
                        raise ValueError
                    port_list.append(port)
            return sorted(list(set(port_list)))  # Remove duplicates and sort
        except ValueError:
            raise ValueError(f"Invalid port specification: {port_spec}")
    
    # Handle port ranges
    if '-' in port_spec:
        try:
            start, end = map(int, port_spec.split('-'))
            if start < 1 or end > 65535 or start > end:
                raise ValueError
            return list(range(start, end + 1))
        except ValueError:
            raise ValueError(f"Invalid port range: {port_spec}")
    
    # Single port
    try:
        port = int(port_spec)
        if port < 1 or port > 65535:
            raise ValueError
        return [port]
    except ValueError:
        raise ValueError(f"Invalid port: {port_spec}")


def get_top_ports(n: int) -> List[int]:
    """
    Get the most commonly used ports.
    
    Args:
        n: Number of top ports to return
        
    Returns:
        List of top N ports
    """
    # Common ports in order of likelihood
    common_ports = [
        80, 443, 22, 21, 23, 25, 53, 110, 143, 993, 995,  # Web, SSH, FTP, Email
        135, 139, 445, 1433, 1521, 3306, 3389, 5432, 5900,  # Windows, Database, RDP
        6379, 8080, 8443, 9000, 27017, 9200, 11211, 6379,   # Redis, Web, MongoDB, Elasticsearch
        22, 23, 25, 53, 80, 110, 143, 443, 993, 995,        # Common services
        135, 139, 445, 993, 995, 1723, 3306, 3389, 5900,    # Windows services
        8080, 8443, 9000, 27017, 9200, 11211, 6379          # Web and database
    ]
    
    # Remove duplicates while preserving order
    seen = set()
    unique_ports = []
    for port in common_ports:
        if port not in seen:
            seen.add(port)
            unique_ports.append(port)
    
    return unique_ports[:n]


def is_port_open(host: str, port: int, timeout: float = 1.0, protocol: str = 'tcp') -> bool:
    """
    Check if a port is open on the given host.
    
    Args:
        host: Target host
        port: Port to check
        timeout: Connection timeout
        protocol: Protocol to use ('tcp' or 'udp')
        
    Returns:
        True if port is open, False otherwise
    """
    try:
        if protocol.lower() == 'tcp':
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(timeout)
            result = sock.connect_ex((host, port))
            sock.close()
            return result == 0
        elif protocol.lower() == 'udp':
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sock.settimeout(timeout)
            try:
                sock.sendto(b'', (host, port))
                sock.recvfrom(1024)
                sock.close()
                return True
            except socket.timeout:
                sock.close()
                return False
        else:
            raise ValueError(f"Unsupported protocol: {protocol}")
    except Exception:
        return False


def scan_ports_parallel(host: str, ports: List[int], max_workers: int = 100, 
                       timeout: float = 1.0, protocol: str = 'tcp') -> List[int]:
    """
    Scan ports in parallel using ThreadPoolExecutor.
    
    Args:
        host: Target host
        ports: List of ports to scan
        max_workers: Maximum number of worker threads
        timeout: Connection timeout
        protocol: Protocol to use
        
    Returns:
        List of open ports
    """
    open_ports = []
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_port = {
            executor.submit(is_port_open, host, port, timeout, protocol): port 
            for port in ports
        }
        
        for future in as_completed(future_to_port):
            port = future_to_port[future]
            try:
                if future.result():
                    open_ports.append(port)
            except Exception:
                pass
    
    return sorted(open_ports)


def format_time(seconds: float) -> str:
    """
    Format time duration in human-readable format.
    
    Args:
        seconds: Time in seconds
        
    Returns:
        Formatted time string
    """
    if seconds < 60:
        return f"{seconds:.2f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.1f}h"


def get_service_name(port: int) -> str:
    """
    Get common service name for a port.
    
    Args:
        port: Port number
        
    Returns:
        Service name
    """
    common_services = {
        21: "FTP", 22: "SSH", 23: "Telnet", 25: "SMTP", 53: "DNS",
        80: "HTTP", 110: "POP3", 143: "IMAP", 443: "HTTPS", 993: "IMAPS",
        995: "POP3S", 135: "RPC", 139: "NetBIOS", 445: "SMB", 1433: "MSSQL",
        1521: "Oracle", 3306: "MySQL", 3389: "RDP", 5432: "PostgreSQL",
        5900: "VNC", 6379: "Redis", 8080: "HTTP-Alt", 8443: "HTTPS-Alt",
        9000: "Web", 27017: "MongoDB", 9200: "Elasticsearch", 11211: "Memcached"
    }
    
    return common_services.get(port, "Unknown")


def validate_ip(ip: str) -> bool:
    """
    Validate IP address format.
    
    Args:
        ip: IP address string
        
    Returns:
        True if valid IP, False otherwise
    """
    try:
        ipaddress.ip_address(ip)
        return True
    except ValueError:
        return False


def resolve_domain(domain: str) -> Optional[str]:
    """
    Resolve domain name to IP address.
    
    Args:
        domain: Domain name
        
    Returns:
        IP address or None if resolution fails
    """
    try:
        return socket.gethostbyname(domain)
    except socket.gaierror:
        return None 