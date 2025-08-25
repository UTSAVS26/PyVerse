"""
HTTP Honeypot Server
Simulates web services to capture web-based attacks like SQL injection, XSS, etc.
"""

import socket
import threading
import time
import hashlib
import random
import re
from typing import Dict, Any, Optional
from .base_server import BaseServer, ConnectionLog

class HTTPServer(BaseServer):
    """HTTP honeypot server implementation"""
    
    def __init__(self, port: int = 80, host: str = "0.0.0.0"):
        super().__init__("HTTP", port, host)
        self.fake_pages = {
            "/": self._generate_index_page,
            "/admin": self._generate_admin_page,
            "/login": self._generate_login_page,
            "/api/users": self._generate_api_endpoint,
            "/phpmyadmin": self._generate_phpmyadmin_page,
            "/wp-admin": self._generate_wordpress_page
        }
        self.attack_patterns = {
            "sql_injection": [
                r"(\b(union|select|insert|update|delete|drop|create|alter)\b)",
                r"(\b(and|or)\b\s+\d+\s*=\s*\d+)",
                r"(\b(and|or)\b\s+['\"]\w+['\"]\s*=\s*['\"]\w+['\"])",
                r"(--|\#|\/\*|\*\/)",
                r"(\bxp_cmdshell\b|\bsp_executesql\b)"
            ],
            "xss": [
                r"(<script[^>]*>.*?</script>)",
                r"(javascript:)",
                r"(on\w+\s*=)",
                r"(<iframe[^>]*>)",
                r"(<img[^>]*on\w+[^>]*>)"
            ],
            "path_traversal": [
                r"(\.\.\/|\.\.\\)",
                r"(\/etc\/passwd|\/etc\/shadow)",
                r"(c:\\windows\\system32)",
                r"(\.\.%2f|\.\.%5c)"
            ],
            "command_injection": [
                r"(\b(cat|ls|dir|whoami|id|pwd|uname)\b)",
                r"(\b(system|exec|shell_exec|passthru)\b)",
                r"(\b(rm|del|mkdir|touch)\b)",
                r"(\b(netcat|nc|telnet|ssh)\b)"
            ]
        }
        
    def _process_connection(self, client_socket: socket.socket, address: tuple, log: ConnectionLog) -> bool:
        """Process HTTP connection"""
        source_ip, source_port = address
        
        try:
            # Receive HTTP request
            request_data = client_socket.recv(4096).decode('utf-8', errors='ignore')
            log.payload_size = len(request_data)
            log.payload_hash = hashlib.md5(request_data.encode()).hexdigest()
            
            # Parse HTTP request
            request_lines = request_data.split('\n')
            if not request_lines:
                return False
                
            first_line = request_lines[0].strip()
            method, path, version = self._parse_request_line(first_line)
            
            # Extract headers
            headers = self._parse_headers(request_lines[1:])
            
            # Detect attacks
            attack_type, confidence = self._detect_attacks(request_data, path, headers)
            if attack_type:
                log.attack_type = attack_type
                log.confidence = confidence
                self.stats["attack_detections"] += 1
            
            # Generate response
            response = self._generate_response(method, path, headers, attack_type)
            
            # Send response
            client_socket.send(response.encode())
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error processing HTTP connection: {e}")
            return False
    
    def _parse_request_line(self, first_line: str) -> tuple:
        """Parse HTTP request line"""
        try:
            parts = first_line.split(' ')
            if len(parts) >= 3:
                return parts[0], parts[1], parts[2]
            return "GET", "/", "HTTP/1.1"
        except:
            return "GET", "/", "HTTP/1.1"
    
    def _parse_headers(self, header_lines: list) -> Dict[str, str]:
        """Parse HTTP headers"""
        headers = {}
        for line in header_lines:
            if ':' in line:
                key, value = line.split(':', 1)
                headers[key.strip().lower()] = value.strip()
        return headers
    
    def _detect_attacks(self, request_data: str, path: str, headers: Dict[str, str]) -> tuple:
        """Detect various types of attacks in the request"""
        request_lower = request_data.lower()
        path_lower = path.lower()
        
        # Check for SQL injection
        for pattern in self.attack_patterns["sql_injection"]:
            if re.search(pattern, request_lower, re.IGNORECASE):
                return "sql_injection", 0.9
        
        # Check for XSS
        for pattern in self.attack_patterns["xss"]:
            if re.search(pattern, request_lower, re.IGNORECASE):
                return "xss", 0.85
        
        # Check for path traversal
        for pattern in self.attack_patterns["path_traversal"]:
            if re.search(pattern, path_lower, re.IGNORECASE):
                return "path_traversal", 0.95
        
        # Check for command injection
        for pattern in self.attack_patterns["command_injection"]:
            if re.search(pattern, request_lower, re.IGNORECASE):
                return "command_injection", 0.8
        
        # Check for suspicious user agents
        user_agent = headers.get('user-agent', '').lower()
        suspicious_agents = ['sqlmap', 'nikto', 'nmap', 'dirb', 'gobuster', 'wfuzz']
        for agent in suspicious_agents:
            if agent in user_agent:
                return "scanning", 0.7
        
        # Check for rapid requests (DoS attempt)
        if self._is_rapid_request(headers.get('x-forwarded-for', '')):
            return "dos", 0.6
        
        return None, 0.0
    
    def _is_rapid_request(self, client_ip: str) -> bool:
        """Check if request is part of a rapid sequence (DoS indicator)"""
        # This is a simplified check - in production you'd track request timing
        return random.random() < 0.05  # 5% chance of being flagged as rapid
    
    def _generate_response(self, method: str, path: str, headers: Dict[str, str], attack_type: Optional[str]) -> str:
        """Generate HTTP response based on request and attack type"""
        
        # Determine response based on attack type
        if attack_type == "sql_injection":
            return self._generate_sql_error_response()
        elif attack_type == "xss":
            return self._generate_xss_response()
        elif attack_type == "path_traversal":
            return self._generate_403_response()
        elif attack_type == "command_injection":
            return self._generate_command_error_response()
        elif attack_type == "scanning":
            return self._generate_scanning_response()
        elif attack_type == "dos":
            return self._generate_dos_response()
        
        # Normal response
        return self._generate_normal_response(method, path)
    
    def _generate_normal_response(self, method: str, path: str) -> str:
        """Generate normal HTTP response"""
        if path in self.fake_pages:
            content = self.fake_pages[path]()
        else:
            content = self._generate_404_page()
        
        response = f"""HTTP/1.1 200 OK
Server: Apache/2.4.41 (Ubuntu)
Content-Type: text/html; charset=UTF-8
Content-Length: {len(content)}
Connection: close

{content}"""
        
        return response
    
    def _generate_sql_error_response(self) -> str:
        """Generate SQL error response to confuse attackers"""
        content = """<!DOCTYPE html>
<html>
<head><title>Database Error</title></head>
<body>
<h1>Database Connection Error</h1>
<p>Unable to connect to MySQL server at localhost (127.0.0.1)</p>
<p>Error: Access denied for user 'root'@'localhost' (using password: YES)</p>
</body>
</html>"""
        
        return f"""HTTP/1.1 500 Internal Server Error
Server: Apache/2.4.41 (Ubuntu)
Content-Type: text/html; charset=UTF-8
Content-Length: {len(content)}
Connection: close

{content}"""
    
    def _generate_xss_response(self) -> str:
        """Generate XSS response"""
        content = """<!DOCTYPE html>
<html>
<head><title>Security Alert</title></head>
<body>
<h1>Security Violation Detected</h1>
<p>Your request has been blocked due to potential security risks.</p>
<p>Please contact the administrator if you believe this is an error.</p>
</body>
</html>"""
        
        return f"""HTTP/1.1 403 Forbidden
Server: Apache/2.4.41 (Ubuntu)
Content-Type: text/html; charset=UTF-8
Content-Length: {len(content)}
Connection: close

{content}"""
    
    def _generate_403_response(self) -> str:
        """Generate 403 Forbidden response"""
        return """HTTP/1.1 403 Forbidden
Server: Apache/2.4.41 (Ubuntu)
Content-Type: text/html; charset=UTF-8
Content-Length: 0
Connection: close

"""
    
    def _generate_command_error_response(self) -> str:
        """Generate command execution error response"""
        content = """<!DOCTYPE html>
<html>
<head><title>System Error</title></head>
<body>
<h1>System Command Execution Failed</h1>
<p>The requested operation could not be completed.</p>
<p>Error: Permission denied</p>
</body>
</html>"""
        
        return f"""HTTP/1.1 500 Internal Server Error
Server: Apache/2.4.41 (Ubuntu)
Content-Type: text/html; charset=UTF-8
Content-Length: {len(content)}
Connection: close

{content}"""
    
    def _generate_scanning_response(self) -> str:
        """Generate response for scanning tools"""
        content = """<!DOCTYPE html>
<html>
<head><title>Welcome</title></head>
<body>
<h1>Welcome to Our Server</h1>
<p>This is a development server. Please be careful with automated tools.</p>
</body>
</html>"""
        
        return f"""HTTP/1.1 200 OK
Server: Apache/2.4.41 (Ubuntu)
Content-Type: text/html; charset=UTF-8
Content-Length: {len(content)}
Connection: close

{content}"""
    
    def _generate_dos_response(self) -> str:
        """Generate response for DoS attempts"""
        return """HTTP/1.1 429 Too Many Requests
Server: Apache/2.4.41 (Ubuntu)
Retry-After: 60
Content-Type: text/html; charset=UTF-8
Content-Length: 0
Connection: close

"""
    
    def _generate_index_page(self) -> str:
        """Generate fake index page"""
        return """<!DOCTYPE html>
<html>
<head><title>Welcome to Our Website</title></head>
<body>
<h1>Welcome!</h1>
<p>This is our company website. Please explore our services.</p>
<a href="/login">Login</a> | <a href="/admin">Admin</a>
</body>
</html>"""
    
    def _generate_admin_page(self) -> str:
        """Generate fake admin page"""
        return """<!DOCTYPE html>
<html>
<head><title>Admin Panel</title></head>
<body>
<h1>Administrative Panel</h1>
<p>Please log in to access administrative functions.</p>
<form action="/login" method="post">
Username: <input type="text" name="username"><br>
Password: <input type="password" name="password"><br>
<input type="submit" value="Login">
</form>
</body>
</html>"""
    
    def _generate_login_page(self) -> str:
        """Generate fake login page"""
        return """<!DOCTYPE html>
<html>
<head><title>Login</title></head>
<body>
<h1>User Login</h1>
<form action="/auth" method="post">
Username: <input type="text" name="username"><br>
Password: <input type="password" name="password"><br>
<input type="submit" value="Login">
</form>
</body>
</html>"""
    
    def _generate_api_endpoint(self) -> str:
        """Generate fake API endpoint"""
        return """{"status": "success", "message": "API endpoint working", "data": []}"""
    
    def _generate_phpmyadmin_page(self) -> str:
        """Generate fake phpMyAdmin page"""
        return """<!DOCTYPE html>
<html>
<head><title>phpMyAdmin</title></head>
<body>
<h1>phpMyAdmin</h1>
<p>Database administration interface</p>
<form action="/phpmyadmin/index.php" method="post">
Server: <input type="text" name="pma_servername" value="localhost"><br>
Username: <input type="text" name="pma_username"><br>
Password: <input type="password" name="pma_password"><br>
<input type="submit" value="Go">
</form>
</body>
</html>"""
    
    def _generate_wordpress_page(self) -> str:
        """Generate fake WordPress admin page"""
        return """<!DOCTYPE html>
<html>
<head><title>WordPress Admin</title></head>
<body>
<h1>WordPress Administration</h1>
<p>Please log in to access the WordPress admin panel.</p>
<form action="/wp-login.php" method="post">
Username: <input type="text" name="log"><br>
Password: <input type="password" name="pwd"><br>
<input type="submit" value="Log In">
</form>
</body>
</html>"""
    
    def _generate_404_page(self) -> str:
        """Generate 404 page"""
        return """<!DOCTYPE html>
<html>
<head><title>404 Not Found</title></head>
<body>
<h1>404 - Page Not Found</h1>
<p>The requested page could not be found.</p>
</body>
</html>"""
    
    def get_attack_stats(self) -> Dict[str, Any]:
        """Get attack statistics"""
        attack_counts = {}
        for log in self.logs:
            if log.attack_type:
                attack_counts[log.attack_type] = attack_counts.get(log.attack_type, 0) + 1
        
        return {
            "total_attacks": sum(attack_counts.values()),
            "attack_types": attack_counts,
            "most_common_attack": max(attack_counts.items(), key=lambda x: x[1])[0] if attack_counts else None
        }
