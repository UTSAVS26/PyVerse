"""
Banner grabbing and service fingerprinting module for PyRecon.
"""

import socket
import ssl
import time
import re
from typing import Dict, Optional, Any
from urllib.parse import urlparse
import base64


class BannerGrabber:
    """
    Banner grabbing and service fingerprinting utility.
    """
    
    def __init__(self, timeout: float = 3.0):
        """
        Initialize the banner grabber.
        
        Args:
            timeout: Connection timeout in seconds
        """
        self.timeout = timeout
        self.service_patterns = {
            'SSH': r'SSH-(\d+\.\d+)',
            'FTP': r'(\d{3})',
            'SMTP': r'(\d{3})',
            'HTTP': r'HTTP/(\d+\.\d+)',
            'HTTPS': r'HTTP/(\d+\.\d+)',
            'POP3': r'(\+OK)',
            'IMAP': r'(\* OK)',
            'DNS': r'',
            'MySQL': r'(\d+\.\d+\.\d+)',
            'PostgreSQL': r'PostgreSQL',
            'Redis': r'(-ERR|-OK)',
            'MongoDB': r'',
            'Elasticsearch': r'(\d+\.\d+\.\d+)',
            'Memcached': r'',
        }
    
    def grab_banner(self, host: str, port: int, protocol: str = 'tcp') -> Dict[str, Any]:
        """
        Grab banner and fingerprint service on the given port.
        
        Args:
            host: Target host
            port: Port to check
            protocol: Protocol to use
            
        Returns:
            Dictionary with banner and service information
        """
        start_time = time.time()
        
        try:
            if protocol.lower() == 'tcp':
                return self._grab_tcp_banner(host, port)
            elif protocol.lower() == 'udp':
                return self._grab_udp_banner(host, port)
            else:
                raise ValueError(f"Unsupported protocol: {protocol}")
        except Exception as e:
            return {
                'service': self._guess_service_by_port(port),
                'banner': None,
                'error': str(e),
                'response_time': time.time() - start_time
            }
    
    def _grab_tcp_banner(self, host: str, port: int) -> Dict[str, Any]:
        """
        Grab TCP banner and fingerprint service.
        
        Args:
            host: Target host
            port: Port to check
            
        Returns:
            Dictionary with banner and service information
        """
        start_time = time.time()
        
        try:
            # Create socket
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(self.timeout)
            
            # Connect
            sock.connect((host, port))
            
            # Try to get banner
            banner = self._get_banner(sock, port)
            
            # Determine service
            service = self._identify_service(port, banner)
            
            # Get additional info based on service
            additional_info = {}
            if service in ['HTTP', 'HTTPS']:
                additional_info = self._get_http_info(host, port, service)
            elif service == 'SSH':
                additional_info = self._get_ssh_info(banner)
            elif service in ['FTP', 'SMTP', 'POP3', 'IMAP']:
                additional_info = self._get_protocol_info(banner, service)
            
            sock.close()
            
            return {
                'service': service,
                'banner': banner,
                'response_time': time.time() - start_time,
                **additional_info
            }
            
        except Exception as e:
            return {
                'service': self._guess_service_by_port(port),
                'banner': None,
                'error': str(e),
                'response_time': time.time() - start_time
            }
    
    def _grab_udp_banner(self, host: str, port: int) -> Dict[str, Any]:
        """
        Grab UDP banner and fingerprint service.
        
        Args:
            host: Target host
            port: Port to check
            
        Returns:
            Dictionary with banner and service information
        """
        start_time = time.time()
        
        try:
            # Create socket
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sock.settimeout(self.timeout)
            
            # Send probe and try to get response
            banner = self._get_udp_banner(sock, host, port)
            
            # Determine service
            service = self._identify_service(port, banner)
            
            sock.close()
            
            return {
                'service': service,
                'banner': banner,
                'response_time': time.time() - start_time
            }
            
        except Exception as e:
            return {
                'service': self._guess_service_by_port(port),
                'banner': None,
                'error': str(e),
                'response_time': time.time() - start_time
            }
    
    def _get_banner(self, sock: socket.socket, port: int) -> Optional[str]:
        """
        Get banner from socket.
        
        Args:
            sock: Connected socket
            port: Port number
            
        Returns:
            Banner string or None
        """
        try:
            # Send initial probe based on port
            if port == 80:
                sock.send(b"GET / HTTP/1.0\r\n\r\n")
            elif port == 443:
                # HTTPS requires SSL handshake
                return None
            elif port == 22:
                # SSH sends banner automatically
                pass
            elif port == 21:
                sock.send(b"QUIT\r\n")
            elif port == 25:
                sock.send(b"QUIT\r\n")
            elif port == 110:
                sock.send(b"QUIT\r\n")
            elif port == 143:
                sock.send(b"a001 LOGOUT\r\n")
            else:
                # Generic probe
                sock.send(b"\r\n")
            
            # Receive response
            sock.settimeout(2.0)
            data = sock.recv(1024)
            
            if data:
                return data.decode('utf-8', errors='ignore').strip()
            
        except Exception:
            pass
        
        return None
    
    def _get_udp_banner(self, sock: socket.socket, host: str, port: int) -> Optional[str]:
        """
        Get UDP banner.
        
        Args:
            sock: UDP socket
            host: Target host
            port: Port number
            
        Returns:
            Banner string or None
        """
        try:
            # Send UDP probe
            if port == 53:
                # DNS query
                query = b'\x00\x01\x01\x00\x00\x01\x00\x00\x00\x00\x00\x00\x07example\x03com\x00\x00\x01\x00\x01'
                sock.sendto(query, (host, port))
            else:
                # Generic UDP probe
                sock.sendto(b"", (host, port))
            
            # Try to receive response
            data, addr = sock.recvfrom(1024)
            
            if data:
                return data.decode('utf-8', errors='ignore').strip()
                
        except Exception:
            pass
        
        return None
    
    def _identify_service(self, port: int, banner: Optional[str]) -> str:
        """
        Identify service based on port and banner.
        
        Args:
            port: Port number
            banner: Banner string
            
        Returns:
            Service name
        """
        # First check by port
        service = self._guess_service_by_port(port)
        
        # Refine based on banner content
        if banner:
            banner_upper = banner.upper()
            
            if 'SSH' in banner_upper:
                return 'SSH'
            elif 'FTP' in banner_upper or '220' in banner[:3]:
                return 'FTP'
            elif 'SMTP' in banner_upper or '220' in banner[:3]:
                return 'SMTP'
            elif 'HTTP' in banner_upper:
                return 'HTTP'
            elif 'POP3' in banner_upper or '+OK' in banner:
                return 'POP3'
            elif 'IMAP' in banner_upper or '* OK' in banner:
                return 'IMAP'
            elif 'MYSQL' in banner_upper:
                return 'MySQL'
            elif 'POSTGRESQL' in banner_upper:
                return 'PostgreSQL'
            elif 'REDIS' in banner_upper or '-ERR' in banner or '-OK' in banner:
                return 'Redis'
            elif 'MONGODB' in banner_upper:
                return 'MongoDB'
            elif 'ELASTICSEARCH' in banner_upper:
                return 'Elasticsearch'
            elif 'MEMCACHED' in banner_upper:
                return 'Memcached'
        
        return service
    
    def _guess_service_by_port(self, port: int) -> str:
        """
        Guess service based on port number.
        
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
    
    def _get_http_info(self, host: str, port: int, service: str) -> Dict[str, Any]:
        """
        Get HTTP/HTTPS specific information.
        
        Args:
            host: Target host
            port: Port number
            service: Service name
            
        Returns:
            Dictionary with HTTP information
        """
        info = {}
        
        try:
            if service == 'HTTPS':
                # Get TLS certificate info
                context = ssl.create_default_context()
                context.check_hostname = False
                context.verify_mode = ssl.CERT_NONE
                
                with socket.create_connection((host, port), timeout=self.timeout) as sock:
                    with context.wrap_socket(sock, server_hostname=host) as ssock:
                        cert = ssock.getpeercert()
                        
                        if cert:
                            info['tls_info'] = {
                                'subject': dict(x[0] for x in cert['subject']),
                                'issuer': dict(x[0] for x in cert['issuer']),
                                'version': cert.get('version'),
                                'serial_number': cert.get('serialNumber'),
                                'not_before': cert.get('notBefore'),
                                'not_after': cert.get('notAfter')
                            }
            
            # Get HTTP headers
            headers = self._get_http_headers(host, port, service)
            if headers:
                info['http_headers'] = headers
                
        except Exception:
            pass
        
        return info
    
    def _get_http_headers(self, host: str, port: int, service: str) -> Optional[Dict[str, str]]:
        """
        Get HTTP headers.
        
        Args:
            host: Target host
            port: Port number
            service: Service name
            
        Returns:
            Dictionary of HTTP headers
        """
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(self.timeout)
            
            if service == 'HTTPS':
                context = ssl.create_default_context()
                context.check_hostname = False
                context.verify_mode = ssl.CERT_NONE
                sock = context.wrap_socket(sock, server_hostname=host)
            
            sock.connect((host, port))
            
            # Send HTTP request
            request = f"HEAD / HTTP/1.1\r\nHost: {host}\r\nUser-Agent: PyRecon/1.0\r\n\r\n"
            sock.send(request.encode())
            
            # Receive response
            response = sock.recv(4096).decode('utf-8', errors='ignore')
            sock.close()
            
            # Parse headers
            headers = {}
            lines = response.split('\r\n')
            
            for line in lines[1:]:  # Skip status line
                if ':' in line:
                    key, value = line.split(':', 1)
                    headers[key.strip()] = value.strip()
            
            return headers
            
        except Exception:
            return None
    
    def _get_ssh_info(self, banner: Optional[str]) -> Dict[str, Any]:
        """
        Get SSH specific information.
        
        Args:
            banner: SSH banner
            
        Returns:
            Dictionary with SSH information
        """
        info = {}
        
        if banner:
            # Extract SSH version
            match = re.search(r'SSH-(\d+\.\d+)', banner)
            if match:
                info['ssh_version'] = match.group(1)
            
            # Extract software info
            if 'OpenSSH' in banner:
                info['software'] = 'OpenSSH'
            elif 'SSH' in banner:
                info['software'] = 'SSH'
        
        return info
    
    def _get_protocol_info(self, banner: Optional[str], service: str) -> Dict[str, Any]:
        """
        Get protocol-specific information.
        
        Args:
            banner: Service banner
            service: Service name
            
        Returns:
            Dictionary with protocol information
        """
        info = {}
        
        if banner:
            # Extract version information
            version_patterns = {
                'FTP': r'(\d+\.\d+)',
                'SMTP': r'(\d+\.\d+)',
                'POP3': r'(\d+\.\d+)',
                'IMAP': r'(\d+\.\d+)'
            }
            
            if service in version_patterns:
                match = re.search(version_patterns[service], banner)
                if match:
                    info['version'] = match.group(1)
            
            # Extract software information
            software_patterns = {
                'FTP': r'(vsftpd|ProFTPD|FileZilla|Pure-FTPd)',
                'SMTP': r'(Postfix|Sendmail|Exchange|Exim)',
                'POP3': r'(Dovecot|Cyrus|Courier)',
                'IMAP': r'(Dovecot|Cyrus|Courier)'
            }
            
            if service in software_patterns:
                match = re.search(software_patterns[service], banner, re.IGNORECASE)
                if match:
                    info['software'] = match.group(1)
        
        return info 