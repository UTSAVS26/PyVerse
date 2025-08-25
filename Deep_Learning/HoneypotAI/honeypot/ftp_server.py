"""
FTP Honeypot Server
Simulates FTP service to capture FTP-based attacks and unauthorized access attempts
"""

import socket
import threading
import time
import hashlib
import random
import re
from typing import Dict, Any, Optional
from .base_server import BaseServer, ConnectionLog

class FTPServer(BaseServer):
    """FTP honeypot server implementation"""
    
    def __init__(self, port: int = 21, host: str = "0.0.0.0"):
        super().__init__("FTP", port, host)
        self.fake_users = {
            "anonymous": "",
            "ftp": "ftp",
            "admin": "admin123",
            "user": "password"
        }
        self.login_attempts = {}
        self.max_attempts = 3
        self.lockout_duration = 300  # 5 minutes
        self.current_user = None
        self.authenticated = False
        
    def _process_connection(self, client_socket: socket.socket, address: tuple, log: ConnectionLog) -> bool:
        """Process FTP connection"""
        source_ip, source_port = address
        
        try:
            # Send FTP welcome banner
            welcome = "220 Welcome to FTP Server (vsFTPd 3.0.3)\r\n"
            client_socket.send(welcome.encode())
            
            # Main FTP command loop
            while self.running:
                try:
                    # Receive command
                    command_data = client_socket.recv(1024).decode('utf-8', errors='ignore').strip()
                    if not command_data:
                        break
                    
                    log.payload_size += len(command_data)
                    log.payload_hash = hashlib.md5(command_data.encode()).hexdigest()
                    
                    # Parse command
                    command_parts = command_data.split(' ')
                    command = command_parts[0].upper() if command_parts else ""
                    args = ' '.join(command_parts[1:]) if len(command_parts) > 1 else ""
                    
                    # Process command
                    response = self._process_command(command, args, source_ip, log)
                    client_socket.send(response.encode())
                    
                    # Check if connection should be closed
                    if command in ["QUIT", "BYE"]:
                        break
                        
                except Exception as e:
                    self.logger.debug(f"Error processing FTP command: {e}")
                    break
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error processing FTP connection: {e}")
            return False
    
    def _process_command(self, command: str, args: str, source_ip: str, log: ConnectionLog) -> str:
        """Process individual FTP commands"""
        
        # Check for attack patterns
        attack_type, confidence = self._detect_attacks(command, args)
        if attack_type:
            log.attack_type = attack_type
            log.confidence = confidence
            self.stats["attack_detections"] += 1
        
        # Process commands
        if command == "USER":
            return self._handle_user_command(args)
        elif command == "PASS":
            return self._handle_pass_command(args, source_ip, log)
        elif command == "QUIT":
            return "221 Goodbye.\r\n"
        elif command == "SYST":
            return "215 UNIX Type: L8\r\n"
        elif command == "FEAT":
            return "211-Features:\r\n MDTM\r\n REST STREAM\r\n SIZE\r\n211 End\r\n"
        elif command == "PWD":
            return "257 \"/\" is current directory.\r\n"
        elif command == "TYPE":
            return "200 Type set to I.\r\n"
        elif command == "PASV":
            return "227 Entering Passive Mode (127,0,0,1,123,45).\r\n"
        elif command == "LIST":
            return self._handle_list_command()
        elif command == "CWD":
            return "250 Directory successfully changed.\r\n"
        elif command == "RETR":
            return "550 Failed to open file.\r\n"
        elif command == "STOR":
            return "550 Permission denied.\r\n"
        elif command == "DELE":
            return "550 Permission denied.\r\n"
        elif command == "MKD":
            return "550 Permission denied.\r\n"
        elif command == "RMD":
            return "550 Permission denied.\r\n"
        elif command == "HELP":
            return "214-The following commands are recognized.\r\n USER PASS QUIT SYST FEAT PWD TYPE PASV LIST CWD RETR STOR DELE MKD RMD HELP\r\n214 Help OK.\r\n"
        else:
            return "500 Unknown command.\r\n"
    
    def _handle_user_command(self, username: str) -> str:
        """Handle USER command"""
        self.current_user = username
        if username.lower() in self.fake_users:
            return "331 Please specify the password.\r\n"
        else:
            return "331 Please specify the password.\r\n"
    
    def _handle_pass_command(self, password: str, source_ip: str, log: ConnectionLog) -> str:
        """Handle PASS command"""
        if not self.current_user:
            return "503 Login with USER first.\r\n"
        
        # Check if IP is locked out
        if self._is_ip_locked_out(source_ip):
            self.logger.info(f"Blocked connection from locked out IP: {source_ip}")
            log.attack_type = "brute_force"
            log.confidence = 0.9
            return "530 Login incorrect.\r\n"
        
        # Check credentials
        expected_password = self.fake_users.get(self.current_user.lower(), "")
        if password == expected_password:
            self.authenticated = True
            return "230 Login successful.\r\n"
        else:
            # Track failed attempts
            if source_ip not in self.login_attempts:
                self.login_attempts[source_ip] = 0
            self.login_attempts[source_ip] += 1
            
            # Check for brute force
            if self.login_attempts[source_ip] >= self.max_attempts:
                self._lockout_ip(source_ip)
                log.attack_type = "brute_force"
                log.confidence = 0.95
                return "530 Login incorrect.\r\n"
            
            return "530 Login incorrect.\r\n"
    
    def _handle_list_command(self) -> str:
        """Handle LIST command"""
        if not self.authenticated:
            return "530 Please login with USER and PASS.\r\n"
        
        # Generate fake directory listing
        fake_files = [
            "drwxr-xr-x 2 ftp ftp 4096 Jan 1 12:00 .",
            "drwxr-xr-x 2 ftp ftp 4096 Jan 1 12:00 ..",
            "-rw-r--r-- 1 ftp ftp 1024 Jan 1 12:00 readme.txt",
            "-rw-r--r-- 1 ftp ftp 2048 Jan 1 12:00 config.txt",
            "drwxr-xr-x 2 ftp ftp 4096 Jan 1 12:00 public"
        ]
        
        response = "150 Here comes the directory listing.\r\n"
        response += "\r\n".join(fake_files) + "\r\n"
        response += "226 Directory send OK.\r\n"
        
        return response
    
    def _detect_attacks(self, command: str, args: str) -> tuple:
        """Detect various types of attacks in FTP commands"""
        command_lower = command.lower()
        args_lower = args.lower()
        full_command = f"{command} {args}".lower()
        
        # Check for command injection
        injection_patterns = [
            r"(;|\||&|\$\(|\`|\$\{)",
            r"(cat|ls|dir|whoami|id|pwd|uname)",
            r"(system|exec|shell_exec|passthru)",
            r"(rm|del|mkdir|touch|echo)"
        ]
        
        for pattern in injection_patterns:
            if re.search(pattern, full_command, re.IGNORECASE):
                return "command_injection", 0.85
        
        # Check for path traversal
        traversal_patterns = [
            r"(\.\.\/|\.\.\\)",
            r"(\/etc\/passwd|\/etc\/shadow)",
            r"(c:\\windows\\system32)",
            r"(\.\.%2f|\.\.%5c)"
        ]
        
        for pattern in traversal_patterns:
            if re.search(pattern, args_lower, re.IGNORECASE):
                return "path_traversal", 0.9
        
        # Check for anonymous access abuse
        if command == "USER" and args.lower() == "anonymous":
            return "anonymous_access", 0.6
        
        # Check for excessive commands (DoS)
        if self._is_excessive_commands():
            return "dos", 0.7
        
        return None, 0.0
    
    def _is_excessive_commands(self) -> bool:
        """Check if there are excessive commands (DoS indicator)"""
        # This is a simplified check - in production you'd track command frequency
        return random.random() < 0.03  # 3% chance of being flagged as excessive
    
    def _is_ip_locked_out(self, source_ip: str) -> bool:
        """Check if IP is currently locked out"""
        if source_ip in self.login_attempts:
            last_attempt_time = self.login_attempts.get(f"{source_ip}_time", 0)
            if time.time() - last_attempt_time < self.lockout_duration:
                return True
        return False
    
    def _lockout_ip(self, source_ip: str):
        """Lock out an IP address"""
        self.login_attempts[f"{source_ip}_time"] = time.time()
        self.logger.warning(f"Locked out IP {source_ip} due to brute force attempts")
    
    def get_ftp_stats(self) -> Dict[str, Any]:
        """Get FTP-specific statistics"""
        locked_ips = [ip for ip in self.login_attempts.keys() if ip.endswith("_time")]
        total_attempts = sum([v for k, v in self.login_attempts.items() if not k.endswith("_time")])
        return {
            "total_attempts": total_attempts,
            "locked_ips": len(locked_ips),
            "current_lockouts": len([ip for ip in locked_ips if self._is_ip_locked_out(ip.replace("_time", ""))]),
            "anonymous_access_attempts": len([log for log in self.logs if log.attack_type == "anonymous_access"])
        }
