"""
SSH Honeypot Server
Simulates SSH service to capture brute force and other SSH-based attacks
"""

import socket
import threading
import time
import hashlib
import random
from typing import Dict, Any
from .base_server import BaseServer, ConnectionLog

class SSHServer(BaseServer):
    """SSH honeypot server implementation"""
    
    def __init__(self, port: int = 22, host: str = "0.0.0.0"):
        super().__init__("SSH", port, host)
        self.fake_users = {
            "admin": "admin123",
            "root": "password",
            "user": "user123",
            "test": "test123"
        }
        self.login_attempts = {}
        self.max_attempts = 3
        self.lockout_duration = 300  # 5 minutes
        
    def _process_connection(self, client_socket: socket.socket, address: tuple, log: ConnectionLog) -> bool:
        """Process SSH connection"""
        source_ip, source_port = address
        
        try:
            # Send SSH banner
            banner = "SSH-2.0-OpenSSH_8.2p1 Ubuntu-4ubuntu0.2\r\n"
            client_socket.send(banner.encode())
            
            # Receive client version
            client_version = client_socket.recv(1024).decode().strip()
            log.payload_size = len(client_version)
            log.payload_hash = hashlib.md5(client_version.encode()).hexdigest()
            
            # Simulate SSH handshake
            self._simulate_ssh_handshake(client_socket)
            
            # Simulate authentication attempts
            auth_success = self._simulate_authentication(client_socket, source_ip, log)
            
            if auth_success:
                # Simulate successful login
                self._simulate_shell_session(client_socket)
                return True
            else:
                # Simulate failed login
                self._simulate_failed_login(client_socket)
                return False
                
        except Exception as e:
            self.logger.error(f"Error processing SSH connection: {e}")
            return False
    
    def _simulate_ssh_handshake(self, client_socket: socket.socket):
        """Simulate SSH protocol handshake"""
        try:
            # Receive key exchange init
            key_exchange = client_socket.recv(1024)
            
            # Send key exchange response
            response = b"\x00\x00\x00\x14\x06\x14\x00\x00\x00\x0c\x00\x00\x00\x07\x00\x00\x00\x00\x00\x00\x00\x00"
            client_socket.send(response)
            
            # Receive new keys
            new_keys = client_socket.recv(1024)
            
            # Send new keys response
            new_keys_response = b"\x00\x00\x00\x0c\x15"
            client_socket.send(new_keys_response)
            
        except Exception as e:
            self.logger.debug(f"SSH handshake error: {e}")
    
    def _simulate_authentication(self, client_socket: socket.socket, source_ip: str, log: ConnectionLog) -> bool:
        """Simulate SSH authentication process"""
        try:
            # Check if IP is locked out
            if self._is_ip_locked_out(source_ip):
                self.logger.info(f"Blocked connection from locked out IP: {source_ip}")
                log.attack_type = "brute_force"
                log.confidence = 0.9
                self.stats["attack_detections"] += 1
                return False
            
            # Simulate multiple authentication attempts
            for attempt in range(random.randint(1, 5)):
                # Receive auth request
                auth_data = client_socket.recv(1024)
                
                # Simulate processing delay
                time.sleep(random.uniform(0.1, 0.5))
                
                # Send auth failure response
                auth_failure = b"\x00\x00\x00\x0c\x33"
                client_socket.send(auth_failure)
                
                # Track failed attempts
                if source_ip not in self.login_attempts:
                    self.login_attempts[source_ip] = 0
                self.login_attempts[source_ip] += 1
                
                # Check for brute force
                if self.login_attempts[source_ip] >= self.max_attempts:
                    self._lockout_ip(source_ip)
                    log.attack_type = "brute_force"
                    log.confidence = 0.95
                    self.stats["attack_detections"] += 1
                    return False
            
            # Simulate successful authentication (rare)
            if random.random() < 0.1:  # 10% chance of "success"
                auth_success = b"\x00\x00\x00\x0c\x32"
                client_socket.send(auth_success)
                return True
            
            return False
            
        except Exception as e:
            self.logger.debug(f"Authentication simulation error: {e}")
            return False
    
    def _simulate_shell_session(self, client_socket: socket.socket):
        """Simulate interactive shell session"""
        try:
            # Send shell prompt
            prompt = b"root@honeypot:~# "
            client_socket.send(prompt)
            
            # Simulate command processing
            for _ in range(random.randint(1, 3)):
                command = client_socket.recv(1024).decode()
                
                # Simulate command execution
                time.sleep(random.uniform(0.5, 2.0))
                
                # Send fake command output
                output = self._generate_fake_output(command)
                client_socket.send(output.encode())
                client_socket.send(prompt)
            
        except Exception as e:
            self.logger.debug(f"Shell session error: {e}")
    
    def _simulate_failed_login(self, client_socket: socket.socket):
        """Simulate failed login response"""
        try:
            # Send connection closed message
            close_msg = b"Connection closed by remote host.\r\n"
            client_socket.send(close_msg)
            
        except Exception as e:
            self.logger.debug(f"Failed login simulation error: {e}")
    
    def _generate_fake_output(self, command: str) -> str:
        """Generate fake command output"""
        command = command.strip().lower()
        
        if "ls" in command:
            return "file1.txt  file2.txt  directory1  directory2\r\n"
        elif "pwd" in command:
            return "/root\r\n"
        elif "whoami" in command:
            return "root\r\n"
        elif "ps" in command:
            return "  PID TTY          TIME CMD\r\n  123 pts/0    00:00:00 bash\r\n  456 pts/0    00:00:00 ps\r\n"
        elif "netstat" in command:
            return "Active Internet connections (w/o servers)\r\nProto Recv-Q Send-Q Local Address           Foreign Address         State\r\n"
        else:
            return f"bash: {command}: command not found\r\n"
    
    def _is_ip_locked_out(self, source_ip: str) -> bool:
        """Check if IP is currently locked out"""
        if source_ip in self.login_attempts:
            attempts = self.login_attempts[source_ip]
            if attempts >= self.max_attempts:
                last_attempt_time = self.login_attempts.get(f"{source_ip}_time", 0)
                if time.time() - last_attempt_time < self.lockout_duration:
                    return True
        return False
    
    def _lockout_ip(self, source_ip: str):
        """Lock out an IP address"""
        self.login_attempts[f"{source_ip}_time"] = time.time()
        self.logger.warning(f"Locked out IP {source_ip} due to brute force attempts")
    
    def get_brute_force_stats(self) -> Dict[str, Any]:
        """Get brute force attack statistics"""
        locked_ips = [ip for ip in self.login_attempts.keys() if ip.endswith("_time")]
        total_attempts = sum([v for k, v in self.login_attempts.items() if not k.endswith("_time")])
        return {
            "total_attempts": total_attempts,
            "locked_ips": len(locked_ips),
            "current_lockouts": len([ip for ip in locked_ips if self._is_ip_locked_out(ip.replace("_time", ""))])
        }
