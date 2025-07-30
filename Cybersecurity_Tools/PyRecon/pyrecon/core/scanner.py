"""
Port scanner module for PyRecon.
"""

import socket
import time
import threading
from typing import List, Dict, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from .utils import (
    parse_target, parse_port_range, scan_ports_parallel, 
    get_service_name, format_time
)
from .banner_grabber import BannerGrabber
from .os_fingerprint import OSFingerprinter


@dataclass
class ScanResult:
    """Result of a port scan."""
    host: str
    port: int
    protocol: str
    status: str  # 'open', 'closed', 'filtered'
    service: str
    banner: Optional[str] = None
    os_guess: Optional[str] = None
    tls_info: Optional[Dict] = None
    response_time: Optional[float] = None


class PortScanner:
    """
    High-speed port scanner with multithreading support.
    """
    
    def __init__(self, max_workers: int = 100, timeout: float = 1.0):
        """
        Initialize the port scanner.
        
        Args:
            max_workers: Maximum number of worker threads
            timeout: Connection timeout in seconds
        """
        self.max_workers = max_workers
        self.timeout = timeout
        self.banner_grabber = BannerGrabber()
        self.os_fingerprinter = OSFingerprinter()
        self._lock = threading.Lock()
        self._results = []
        
    def scan(self, target: str, ports: str = "top-100", 
             protocol: str = "tcp", fingerprint: bool = False,
             progress_callback: Optional[callable] = None) -> List[ScanResult]:
        """
        Scan ports on the target host.
        
        Args:
            target: Target host (IP, domain, CIDR, or file)
            ports: Port specification (range, list, or 'top-N')
            protocol: Protocol to use ('tcp' or 'udp')
            fingerprint: Whether to perform service fingerprinting
            progress_callback: Callback function for progress updates
            
        Returns:
            List of scan results
        """
        start_time = time.time()
        
        # Parse target and ports
        hosts = parse_target(target)
        port_list = parse_port_range(ports)
        
        if not hosts:
            raise ValueError("No valid targets found")
        
        print(f"Starting scan of {len(hosts)} host(s) on {len(port_list)} port(s)")
        print(f"Protocol: {protocol.upper()}")
        print(f"Workers: {self.max_workers}")
        print("-" * 50)
        
        all_results = []
        
        for i, host in enumerate(hosts):
            print(f"\nScanning host {i+1}/{len(hosts)}: {host}")
            
            # Scan ports for this host
            open_ports = scan_ports_parallel(
                host, port_list, self.max_workers, self.timeout, protocol
            )
            
            # Perform fingerprinting if requested
            results = []
            if fingerprint and open_ports:
                results = self._fingerprint_ports(host, open_ports, protocol)
            else:
                # Create basic results without fingerprinting
                for port in open_ports:
                    result = ScanResult(
                        host=host,
                        port=port,
                        protocol=protocol,
                        status="open",
                        service=get_service_name(port)
                    )
                    results.append(result)
            
            all_results.extend(results)
            
            # Progress callback
            if progress_callback:
                progress_callback(i + 1, len(hosts))
        
        scan_time = time.time() - start_time
        print(f"\nScan completed in {format_time(scan_time)}")
        print(f"Found {len(all_results)} open ports")
        
        return all_results
    
    def _fingerprint_ports(self, host: str, open_ports: List[int], 
                          protocol: str) -> List[ScanResult]:
        """
        Perform service fingerprinting on open ports.
        
        Args:
            host: Target host
            open_ports: List of open ports
            protocol: Protocol used
            
        Returns:
            List of scan results with fingerprinting data
        """
        results = []
        
        with ThreadPoolExecutor(max_workers=min(self.max_workers, len(open_ports))) as executor:
            future_to_port = {
                executor.submit(self._fingerprint_single_port, host, port, protocol): port
                for port in open_ports
            }
            
            for future in as_completed(future_to_port):
                port = future_to_port[future]
                try:
                    result = future.result()
                    if result:
                        results.append(result)
                except Exception as e:
                    # Create basic result if fingerprinting fails
                    result = ScanResult(
                        host=host,
                        port=port,
                        protocol=protocol,
                        status="open",
                        service=get_service_name(port)
                    )
                    results.append(result)
        
        return sorted(results, key=lambda x: x.port)
    
    def _fingerprint_single_port(self, host: str, port: int, 
                                protocol: str) -> Optional[ScanResult]:
        """
        Fingerprint a single port.
        
        Args:
            host: Target host
            port: Port to fingerprint
            protocol: Protocol used
            
        Returns:
            Scan result with fingerprinting data
        """
        try:
            # Get banner and service info
            banner_info = self.banner_grabber.grab_banner(host, port, protocol)
            
            # Get OS fingerprint
            os_guess = self.os_fingerprinter.fingerprint(host, port, protocol)
            
            # Create result
            result = ScanResult(
                host=host,
                port=port,
                protocol=protocol,
                status="open",
                service=banner_info.get('service', get_service_name(port)),
                banner=banner_info.get('banner'),
                os_guess=os_guess,
                tls_info=banner_info.get('tls_info'),
                response_time=banner_info.get('response_time')
            )
            
            return result
            
        except Exception as e:
            # Return basic result if fingerprinting fails
            return ScanResult(
                host=host,
                port=port,
                protocol=protocol,
                status="open",
                service=get_service_name(port)
            )
    
    def quick_scan(self, target: str, ports: str = "top-100") -> List[ScanResult]:
        """
        Perform a quick scan without fingerprinting.
        
        Args:
            target: Target host
            ports: Port specification
            
        Returns:
            List of scan results
        """
        return self.scan(target, ports, fingerprint=False)
    
    def full_scan(self, target: str, ports: str = "1-1024") -> List[ScanResult]:
        """
        Perform a full scan with fingerprinting.
        
        Args:
            target: Target host
            ports: Port specification
            
        Returns:
            List of scan results
        """
        return self.scan(target, ports, fingerprint=True)
    
    def scan_udp(self, target: str, ports: str = "top-100") -> List[ScanResult]:
        """
        Perform UDP port scan.
        
        Args:
            target: Target host
            ports: Port specification
            
        Returns:
            List of scan results
        """
        return self.scan(target, ports, protocol="udp", fingerprint=True)
    
    def get_statistics(self, results: List[ScanResult]) -> Dict:
        """
        Get statistics from scan results.
        
        Args:
            results: List of scan results
            
        Returns:
            Dictionary with scan statistics
        """
        if not results:
            return {}
        
        hosts = set(r.host for r in results)
        protocols = set(r.protocol for r in results)
        services = {}
        
        for result in results:
            service = result.service
            if service in services:
                services[service] += 1
            else:
                services[service] = 1
        
        return {
            'total_ports': len(results),
            'unique_hosts': len(hosts),
            'protocols': list(protocols),
            'services': services,
            'hosts': list(hosts)
        } 