"""
OS fingerprinting module for PyRecon.
"""

import socket
import struct
import time
from typing import Optional, Dict, Any


class OSFingerprinter:
    """
    Basic OS fingerprinting utility based on TTL values and response patterns.
    """
    
    def __init__(self):
        """
        Initialize the OS fingerprinter.
        """
        # TTL values for different operating systems
        self.ttl_patterns = {
            32: "Windows",
            64: "Linux/Unix",
            128: "Windows",
            255: "Network Device"
        }
        
        # Common OS signatures
        self.os_signatures = {
            'Windows': ['Windows', 'Microsoft', 'IIS', 'Exchange'],
            'Linux': ['Linux', 'Ubuntu', 'Debian', 'CentOS', 'Red Hat', 'Apache'],
            'Unix': ['BSD', 'FreeBSD', 'OpenBSD', 'NetBSD', 'Solaris'],
            'Network': ['Cisco', 'Juniper', 'HP', 'Dell', 'Brocade']
        }
    
    def fingerprint(self, host: str, port: int, protocol: str = 'tcp') -> Optional[str]:
        """
        Perform OS fingerprinting on the target host.
        
        Args:
            host: Target host
            port: Port to check
            protocol: Protocol to use
            
        Returns:
            OS guess string or None
        """
        try:
            # Get TTL-based guess
            ttl_guess = self._get_ttl_guess(host, port, protocol)
            
            # Get banner-based guess
            banner_guess = self._get_banner_guess(host, port, protocol)
            
            # Combine guesses
            if ttl_guess and banner_guess:
                return f"{banner_guess} (TTL={ttl_guess})"
            elif ttl_guess:
                return f"{ttl_guess} (TTL-based)"
            elif banner_guess:
                return f"{banner_guess} (Banner-based)"
            else:
                return None
                
        except Exception:
            return None
    
    def _get_ttl_guess(self, host: str, port: int, protocol: str) -> Optional[str]:
        """
        Get OS guess based on TTL value.
        
        Args:
            host: Target host
            port: Port to check
            protocol: Protocol to use
            
        Returns:
            OS guess based on TTL
        """
        try:
            # Send a ping-like packet to get TTL
            ttl = self._get_ttl(host, port, protocol)
            
            if ttl:
                # Map TTL to OS
                for ttl_value, os_name in self.ttl_patterns.items():
                    if abs(ttl - ttl_value) <= 8:  # Allow some variance
                        return os_name
            
        except Exception:
            pass
        
        return None
    
    def _get_ttl(self, host: str, port: int, protocol: str) -> Optional[int]:
        """
        Get TTL value from target.
        
        Args:
            host: Target host
            port: Port to check
            protocol: Protocol to use
            
        Returns:
            TTL value or None
        """
        try:
            if protocol.lower() == 'tcp':
                return self._get_tcp_ttl(host, port)
            elif protocol.lower() == 'udp':
                return self._get_udp_ttl(host, port)
            else:
                return None
                
        except Exception:
            return None
    
    def _get_tcp_ttl(self, host: str, port: int) -> Optional[int]:
        """
        Get TTL from TCP connection.
        
        Args:
            host: Target host
            port: Port to check
            
        Returns:
            TTL value or None
        """
        try:
            # Create socket
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(2.0)
            
            # Connect and get TTL
            sock.connect((host, port))
            
            # Get socket info
            sock_info = sock.getsockopt(socket.IPPROTO_IP, socket.IP_TTL)
            sock.close()
            
            return sock_info
            
        except Exception:
            return None
    
    def _get_udp_ttl(self, host: str, port: int) -> Optional[int]:
        """
        Get TTL from UDP connection.
        
        Args:
            host: Target host
            port: Port to check
            
        Returns:
            TTL value or None
        """
        try:
            # Create socket
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sock.settimeout(2.0)
            
            # Send probe and get TTL
            sock.sendto(b"", (host, port))
            
            # Get socket info
            sock_info = sock.getsockopt(socket.IPPROTO_IP, socket.IP_TTL)
            sock.close()
            
            return sock_info
            
        except Exception:
            return None
    
    def _get_banner_guess(self, host: str, port: int, protocol: str) -> Optional[str]:
        """
        Get OS guess based on service banner.
        
        Args:
            host: Target host
            port: Port to check
            protocol: Protocol to use
            
        Returns:
            OS guess based on banner
        """
        try:
            # Get banner
            banner = self._get_banner(host, port, protocol)
            
            if banner:
                banner_upper = banner.upper()
                
                # Check for OS signatures
                for os_name, signatures in self.os_signatures.items():
                    for signature in signatures:
                        if signature.upper() in banner_upper:
                            return os_name
            
        except Exception:
            pass
        
        return None
    
    def _get_banner(self, host: str, port: int, protocol: str) -> Optional[str]:
        """
        Get banner from service.
        
        Args:
            host: Target host
            port: Port to check
            protocol: Protocol to use
            
        Returns:
            Banner string or None
        """
        try:
            if protocol.lower() == 'tcp':
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(2.0)
                sock.connect((host, port))
                
                # Send probe
                if port == 80:
                    sock.send(b"GET / HTTP/1.0\r\n\r\n")
                elif port == 22:
                    # SSH sends banner automatically
                    pass
                else:
                    sock.send(b"\r\n")
                
                # Receive response
                data = sock.recv(1024)
                sock.close()
                
                if data:
                    return data.decode('utf-8', errors='ignore').strip()
                    
            elif protocol.lower() == 'udp':
                sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                sock.settimeout(2.0)
                
                # Send probe
                sock.sendto(b"", (host, port))
                
                # Try to receive response
                data, addr = sock.recvfrom(1024)
                sock.close()
                
                if data:
                    return data.decode('utf-8', errors='ignore').strip()
                    
        except Exception:
            pass
        
        return None
    
    def get_detailed_fingerprint(self, host: str, port: int, protocol: str = 'tcp') -> Dict[str, Any]:
        """
        Get detailed OS fingerprinting information.
        
        Args:
            host: Target host
            port: Port to check
            protocol: Protocol to use
            
        Returns:
            Dictionary with detailed fingerprinting information
        """
        result = {
            'os_guess': None,
            'ttl': None,
            'banner': None,
            'confidence': 'low'
        }
        
        try:
            # Get TTL
            ttl = self._get_ttl(host, port, protocol)
            if ttl:
                result['ttl'] = ttl
                
                # Map TTL to OS
                for ttl_value, os_name in self.ttl_patterns.items():
                    if abs(ttl - ttl_value) <= 8:
                        result['os_guess'] = os_name
                        result['confidence'] = 'medium'
                        break
            
            # Get banner
            banner = self._get_banner(host, port, protocol)
            if banner:
                result['banner'] = banner
                
                # Check for OS signatures in banner
                banner_upper = banner.upper()
                for os_name, signatures in self.os_signatures.items():
                    for signature in signatures:
                        if signature.upper() in banner_upper:
                            if result['os_guess'] and result['os_guess'] == os_name:
                                result['confidence'] = 'high'
                            elif not result['os_guess']:
                                result['os_guess'] = os_name
                                result['confidence'] = 'medium'
                            break
            
            # Combine results
            if result['os_guess']:
                if result['ttl']:
                    result['os_guess'] = f"{result['os_guess']} (TTL={result['ttl']})"
            
        except Exception as e:
            result['error'] = str(e)
        
        return result
    
    def get_common_os_patterns(self) -> Dict[str, list]:
        """
        Get common OS patterns for reference.
        
        Returns:
            Dictionary of OS patterns
        """
        return {
            'TTL Patterns': self.ttl_patterns,
            'Banner Signatures': self.os_signatures
        } 