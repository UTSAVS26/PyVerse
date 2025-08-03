"""
HTTP fingerprinting plugin for PyRecon.
"""

import socket
import ssl
import re
from typing import Dict, Optional, Any
from urllib.parse import urlparse


class HTTPFingerprinter:
    """
    HTTP service fingerprinting utility.
    """
    
    def __init__(self, timeout: float = 3.0):
        """
        Initialize the HTTP fingerprinter.
        
        Args:
            timeout: Connection timeout in seconds
        """
        self.timeout = timeout
        
        # Common web server signatures
        self.server_signatures = {
            'Apache': [r'Apache', r'httpd'],
            'Nginx': [r'nginx'],
            'IIS': [r'IIS', r'Microsoft-IIS'],
            'Lighttpd': [r'lighttpd'],
            'Caddy': [r'Caddy'],
            'Cloudflare': [r'cloudflare'],
            'CDN': [r'cdn', r'cloudfront', r'fastly']
        }
        
        # Common web framework signatures
        self.framework_signatures = {
            'Django': [r'Django', r'CSRF'],
            'Flask': [r'Flask', r'Werkzeug'],
            'Express': [r'Express', r'Node.js'],
            'Rails': [r'Rails', r'Ruby'],
            'Laravel': [r'Laravel', r'PHP'],
            'Spring': [r'Spring', r'Java'],
            'ASP.NET': [r'ASP.NET', r'.NET'],
            'WordPress': [r'WordPress', r'wp-content'],
            'Drupal': [r'Drupal'],
            'Joomla': [r'Joomla']
        }
    
    def fingerprint(self, host: str, port: int, use_ssl: bool = False) -> Dict[str, Any]:
        """
        Fingerprint HTTP service on the given port.
        
        Args:
            host: Target host
            port: Port to check
            use_ssl: Whether to use HTTPS
            
        Returns:
            Dictionary with HTTP fingerprinting information
        """
        result = {
            'server': None,
            'framework': None,
            'headers': {},
            'ssl_info': None,
            'technologies': [],
            'security_headers': []
        }
        
        try:
            # Get HTTP headers
            headers = self._get_http_headers(host, port, use_ssl)
            if headers:
                result['headers'] = headers
                
                # Identify server
                result['server'] = self._identify_server(headers)
                
                # Identify framework
                result['framework'] = self._identify_framework(headers)
                
                # Check for technologies
                result['technologies'] = self._identify_technologies(headers)
                
                # Check security headers
                result['security_headers'] = self._check_security_headers(headers)
            
            # Get SSL info if using HTTPS
            if use_ssl:
                result['ssl_info'] = self._get_ssl_info(host, port)
                
        except Exception as e:
            result['error'] = str(e)
        
        return result
    
    def _get_http_headers(self, host: str, port: int, use_ssl: bool) -> Optional[Dict[str, str]]:
        """
        Get HTTP headers from the server.
        
        Args:
            host: Target host
            port: Port to check
            use_ssl: Whether to use HTTPS
            
        Returns:
            Dictionary of HTTP headers
        """
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(self.timeout)
            
            if use_ssl:
                context = ssl.create_default_context()
                context.check_hostname = False
                context.verify_mode = ssl.CERT_NONE
                sock = context.wrap_socket(sock, server_hostname=host)
            
            sock.connect((host, port))
            
            # Send HTTP request
            request = (
                f"HEAD / HTTP/1.1\r\n"
                f"Host: {host}\r\n"
                f"User-Agent: PyRecon/1.0\r\n"
                f"Accept: */*\r\n"
                f"Connection: close\r\n\r\n"
            )
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
    
    def _identify_server(self, headers: Dict[str, str]) -> Optional[str]:
        """
        Identify web server from headers.
        
        Args:
            headers: HTTP headers
            
        Returns:
            Server name or None
        """
        server_header = headers.get('Server', '').lower()
        
        for server_name, patterns in self.server_signatures.items():
            for pattern in patterns:
                if re.search(pattern, server_header, re.IGNORECASE):
                    return server_name
        
        return None
    
    def _identify_framework(self, headers: Dict[str, str]) -> Optional[str]:
        """
        Identify web framework from headers.
        
        Args:
            headers: HTTP headers
            
        Returns:
            Framework name or None
        """
        # Check various headers for framework signatures
        headers_text = ' '.join(headers.values()).lower()
        
        for framework_name, patterns in self.framework_signatures.items():
            for pattern in patterns:
                if re.search(pattern, headers_text, re.IGNORECASE):
                    return framework_name
        
        return None
    
    def _identify_technologies(self, headers: Dict[str, str]) -> list:
        """
        Identify technologies from headers.
        
        Args:
            headers: HTTP headers
            
        Returns:
            List of identified technologies
        """
        technologies = []
        headers_text = ' '.join(headers.values()).lower()
        
        # Check for common technologies
        tech_patterns = {
            'PHP': r'php',
            'Python': r'python',
            'Node.js': r'node',
            'Java': r'java',
            'Ruby': r'ruby',
            'Go': r'go',
            'React': r'react',
            'Angular': r'angular',
            'Vue.js': r'vue',
            'jQuery': r'jquery',
            'Bootstrap': r'bootstrap',
            'WordPress': r'wordpress',
            'Drupal': r'drupal',
            'Joomla': r'joomla',
            'Magento': r'magento',
            'Shopify': r'shopify',
            'WooCommerce': r'woocommerce',
            'Cloudflare': r'cloudflare',
            'AWS': r'aws',
            'Google Analytics': r'google-analytics',
            'Facebook Pixel': r'facebook',
            'Google Tag Manager': r'gtm'
        }
        
        for tech_name, pattern in tech_patterns.items():
            if re.search(pattern, headers_text, re.IGNORECASE):
                technologies.append(tech_name)
        
        return technologies
    
    def _check_security_headers(self, headers: Dict[str, str]) -> list:
        """
        Check for security headers.
        
        Args:
            headers: HTTP headers
            
        Returns:
            List of security headers found
        """
        security_headers = []
        
        security_header_names = [
            'Strict-Transport-Security',
            'Content-Security-Policy',
            'X-Frame-Options',
            'X-Content-Type-Options',
            'X-XSS-Protection',
            'Referrer-Policy',
            'Permissions-Policy',
            'X-Permitted-Cross-Domain-Policies'
        ]
        
        for header_name in security_header_names:
            if header_name in headers:
                security_headers.append(header_name)
        
        return security_headers
    
    def _get_ssl_info(self, host: str, port: int) -> Optional[Dict[str, Any]]:
        """
        Get SSL certificate information.
        
        Args:
            host: Target host
            port: Port to check
            
        Returns:
            SSL certificate information
        """
        try:
            context = ssl.create_default_context()
            context.check_hostname = False
            context.verify_mode = ssl.CERT_NONE
            
            with socket.create_connection((host, port), timeout=self.timeout) as sock:
                with context.wrap_socket(sock, server_hostname=host) as ssock:
                    cert = ssock.getpeercert()
                    
                    if cert:
                        return {
                            'subject': dict(x[0] for x in cert['subject']),
                            'issuer': dict(x[0] for x in cert['issuer']),
                            'version': cert.get('version'),
                            'serial_number': cert.get('serialNumber'),
                            'not_before': cert.get('notBefore'),
                            'not_after': cert.get('notAfter'),
                            'san': cert.get('subjectAltName', [])
                        }
                        
        except Exception:
            pass
        
        return None
    
    def get_detailed_analysis(self, host: str, port: int, use_ssl: bool = False) -> Dict[str, Any]:
        """
        Get detailed HTTP analysis.
        
        Args:
            host: Target host
            port: Port to check
            use_ssl: Whether to use HTTPS
            
        Returns:
            Detailed analysis dictionary
        """
        basic_info = self.fingerprint(host, port, use_ssl)
        
        # Add additional analysis
        analysis = {
            **basic_info,
            'recommendations': [],
            'security_score': 0
        }
        
        # Calculate security score
        security_score = 0
        
        # Points for security headers
        security_headers = basic_info.get('security_headers', [])
        security_score += len(security_headers) * 10
        
        # Points for HTTPS
        if use_ssl:
            security_score += 20
        
        # Points for modern server
        server = basic_info.get('server')
        if server in ['Nginx', 'Apache']:
            security_score += 10
        
        analysis['security_score'] = min(security_score, 100)
        
        # Generate recommendations
        recommendations = []
        
        if not use_ssl:
            recommendations.append("Enable HTTPS/SSL")
        
        if 'Strict-Transport-Security' not in security_headers:
            recommendations.append("Add HSTS header")
        
        if 'Content-Security-Policy' not in security_headers:
            recommendations.append("Add Content Security Policy")
        
        if 'X-Frame-Options' not in security_headers:
            recommendations.append("Add X-Frame-Options header")
        
        analysis['recommendations'] = recommendations
        
        return analysis 