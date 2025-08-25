"""
Tests for HoneypotAI Honeypot Module
"""

import pytest
import time
import socket
import threading
from unittest.mock import Mock, patch, MagicMock
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from honeypot import HoneypotManager, SSHServer, HTTPServer, FTPServer
from honeypot.base_server import BaseServer, ConnectionLog

class TestBaseServer:
    """Test base server functionality"""
    
    def test_base_server_initialization(self):
        """Test base server initialization"""
        server = SSHServer(port=2222)
        
        assert server.service_name == "SSH"
        assert server.port == 2222
        assert server.host == "0.0.0.0"
        assert not server.running
        assert server.socket is None
        assert len(server.logs) == 0
    
    def test_base_server_stats(self):
        """Test base server statistics"""
        server = SSHServer(port=2222)
        
        stats = server.get_stats()
        assert "total_connections" in stats
        assert "successful_connections" in stats
        assert "failed_connections" in stats
        assert "attack_detections" in stats
        assert stats["total_connections"] == 0
    
    def test_base_server_logs(self):
        """Test base server logging"""
        server = SSHServer(port=2222)
        
        # Test empty logs
        logs = server.get_logs()
        assert len(logs) == 0
        
        # Test logs with limit
        logs = server.get_logs(limit=10)
        assert len(logs) == 0
    
    def test_base_server_clear_logs(self):
        """Test clearing logs"""
        server = SSHServer(port=2222)
        
        # Add some mock logs
        server.logs = [Mock(), Mock(), Mock()]
        assert len(server.logs) == 3
        
        server.clear_logs()
        assert len(server.logs) == 0

class TestSSHServer:
    """Test SSH server functionality"""
    
    def test_ssh_server_initialization(self):
        """Test SSH server initialization"""
        server = SSHServer(port=2222)
        
        assert server.service_name == "SSH"
        assert server.port == 2222
        assert "admin" in server.fake_users
        assert "root" in server.fake_users
        assert server.max_attempts == 3
        assert server.lockout_duration == 300
    
    def test_ssh_brute_force_detection(self):
        """Test SSH brute force detection"""
        server = SSHServer(port=2222)
        
        # Simulate multiple failed attempts
        server.login_attempts["192.168.1.1"] = 4  # More than max_attempts (3)
        server.login_attempts["192.168.1.1_time"] = time.time()  # Set lockout time
        
        # Should be locked out after 3 attempts
        assert server._is_ip_locked_out("192.168.1.1")
    
    def test_ssh_brute_force_stats(self):
        """Test SSH brute force statistics"""
        server = SSHServer(port=2222)
        
        # Add some mock attempts
        server.login_attempts["192.168.1.1"] = 5
        server.login_attempts["192.168.1.2"] = 3
        server.login_attempts["192.168.1.1_time"] = time.time()
        
        stats = server.get_brute_force_stats()
        assert stats["total_attempts"] == 8
        assert stats["locked_ips"] == 1
    
    def test_ssh_fake_output_generation(self):
        """Test SSH fake command output generation"""
        server = SSHServer(port=2222)
        
        # Test different commands
        assert "file1.txt" in server._generate_fake_output("ls")
        assert "/root" in server._generate_fake_output("pwd")
        assert "root" in server._generate_fake_output("whoami")
        assert "bash:" in server._generate_fake_output("invalid_command")

class TestHTTPServer:
    """Test HTTP server functionality"""
    
    def test_http_server_initialization(self):
        """Test HTTP server initialization"""
        server = HTTPServer(port=8080)
        
        assert server.service_name == "HTTP"
        assert server.port == 8080
        assert "/" in server.fake_pages
        assert "/admin" in server.fake_pages
        assert "sql_injection" in server.attack_patterns
    
    def test_http_sql_injection_detection(self):
        """Test HTTP SQL injection detection"""
        server = HTTPServer(port=8080)
        
        # Test SQL injection patterns
        request_data = "SELECT * FROM users WHERE id=1 OR 1=1"
        path = "/login"
        headers = {}
        
        attack_type, confidence = server._detect_attacks(request_data, path, headers)
        assert attack_type == "sql_injection"
        assert confidence == 0.9
    
    def test_http_xss_detection(self):
        """Test HTTP XSS detection"""
        server = HTTPServer(port=8080)
        
        # Test XSS patterns
        request_data = "<script>alert('xss')</script>"
        path = "/search"
        headers = {}
        
        attack_type, confidence = server._detect_attacks(request_data, path, headers)
        assert attack_type == "xss"
        assert confidence == 0.85
    
    def test_http_path_traversal_detection(self):
        """Test HTTP path traversal detection"""
        server = HTTPServer(port=8080)
        
        # Test path traversal patterns
        request_data = "GET /../../../etc/passwd HTTP/1.1"
        path = "/../../../etc/passwd"
        headers = {}
        
        attack_type, confidence = server._detect_attacks(request_data, path, headers)
        assert attack_type == "path_traversal"
        assert confidence == 0.95
    
    def test_http_attack_stats(self):
        """Test HTTP attack statistics"""
        server = HTTPServer(port=8080)
        
        # Add some mock logs with attacks
        mock_log1 = Mock()
        mock_log1.attack_type = "sql_injection"
        mock_log2 = Mock()
        mock_log2.attack_type = "xss"
        mock_log3 = Mock()
        mock_log3.attack_type = "sql_injection"
        
        server.logs = [mock_log1, mock_log2, mock_log3]
        
        stats = server.get_attack_stats()
        assert stats["total_attacks"] == 3
        assert stats["attack_types"]["sql_injection"] == 2
        assert stats["attack_types"]["xss"] == 1
        assert stats["most_common_attack"] == "sql_injection"

class TestFTPServer:
    """Test FTP server functionality"""
    
    def test_ftp_server_initialization(self):
        """Test FTP server initialization"""
        server = FTPServer(port=2121)
        
        assert server.service_name == "FTP"
        assert server.port == 2121
        assert "anonymous" in server.fake_users
        assert "ftp" in server.fake_users
        assert server.max_attempts == 3
    
    def test_ftp_command_injection_detection(self):
        """Test FTP command injection detection"""
        server = FTPServer(port=2121)
        
        # Test command injection patterns
        command = "USER"
        args = "test; cat /etc/passwd"
        
        attack_type, confidence = server._detect_attacks(command, args)
        assert attack_type == "command_injection"
        assert confidence == 0.85
    
    def test_ftp_anonymous_access_detection(self):
        """Test FTP anonymous access detection"""
        server = FTPServer(port=2121)
        
        # Test anonymous access
        command = "USER"
        args = "anonymous"
        
        attack_type, confidence = server._detect_attacks(command, args)
        assert attack_type == "anonymous_access"
        assert confidence == 0.6
    
    def test_ftp_ftp_stats(self):
        """Test FTP statistics"""
        server = FTPServer(port=2121)
        
        # Add some mock data
        server.login_attempts["192.168.1.1"] = 5
        server.login_attempts["192.168.1.1_time"] = time.time()
        
        mock_log1 = Mock()
        mock_log1.attack_type = "anonymous_access"
        server.logs = [mock_log1]
        
        stats = server.get_ftp_stats()
        assert stats["total_attempts"] == 5
        assert stats["locked_ips"] == 1
        assert stats["anonymous_access_attempts"] == 1

class TestHoneypotManager:
    """Test honeypot manager functionality"""
    
    def test_honeypot_manager_initialization(self):
        """Test honeypot manager initialization"""
        manager = HoneypotManager()
        
        assert len(manager.servers) == 0
        assert not manager.running
        assert len(manager.logs) == 0
        assert "ssh" in manager.default_config
        assert "http" in manager.default_config
        assert "ftp" in manager.default_config
    
    def test_deploy_service(self):
        """Test service deployment"""
        manager = HoneypotManager()
        
        # Mock the server start method
        with patch.object(SSHServer, 'start', return_value=True):
            success = manager.deploy_service("ssh", port=2222)
            assert success
            assert "ssh" in manager.servers
    
    def test_deploy_all_services(self):
        """Test deploying all services"""
        manager = HoneypotManager()
        
        # Mock all server start methods
        with patch.object(SSHServer, 'start', return_value=True), \
             patch.object(HTTPServer, 'start', return_value=True), \
             patch.object(FTPServer, 'start', return_value=True):
            
            success = manager.deploy_all_services()
            assert success
            assert manager.running
            assert len(manager.servers) == 3
    
    def test_get_service_status(self):
        """Test getting service status"""
        manager = HoneypotManager()
        
        # Mock server
        mock_server = Mock()
        mock_server.get_stats.return_value = {"total_connections": 10}
        mock_server.is_running.return_value = True
        mock_server.get_brute_force_stats.return_value = {"total_attempts": 5}
        
        manager.servers["ssh"] = mock_server
        
        status = manager.get_service_status("ssh")
        assert status["total_connections"] == 10
        assert status["running"] is True
        assert status["service_type"] == "ssh"
    
    def test_get_all_services_status(self):
        """Test getting all services status"""
        manager = HoneypotManager()
        
        # Mock servers
        mock_ssh = Mock()
        mock_ssh.get_stats.return_value = {"total_connections": 10}
        mock_ssh.is_running.return_value = True
        mock_ssh.get_brute_force_stats.return_value = {"total_attempts": 5}
        
        mock_http = Mock()
        mock_http.get_stats.return_value = {"total_connections": 20}
        mock_http.is_running.return_value = True
        mock_http.get_attack_stats.return_value = {"total_attacks": 3}
        
        manager.servers["ssh"] = mock_ssh
        manager.servers["http"] = mock_http
        
        status = manager.get_all_services_status()
        assert "ssh" in status
        assert "http" in status
        assert status["ssh"]["total_connections"] == 10
        assert status["http"]["total_connections"] == 20
    
    def test_get_overall_stats(self):
        """Test getting overall statistics"""
        manager = HoneypotManager()
        
        # Mock logs
        mock_log1 = Mock()
        mock_log1.attack_type = "sql_injection"
        mock_log2 = Mock()
        mock_log2.attack_type = "xss"
        mock_log3 = Mock()
        mock_log3.attack_type = "sql_injection"
        
        manager.logs = [mock_log1, mock_log2, mock_log3]
        
        stats = manager.get_overall_stats()
        assert stats["total_connections"] == 0
        assert stats["total_attacks"] == 0
        assert "attack_breakdown" in stats
        assert stats["attack_breakdown"]["sql_injection"] == 2
        assert stats["attack_breakdown"]["xss"] == 1
        assert stats["most_common_attack"] == "sql_injection"
    
    def test_export_logs(self):
        """Test log export functionality"""
        manager = HoneypotManager()
        
        # Mock logs
        mock_log = {
            "timestamp": "2023-01-01T00:00:00",
            "source_ip": "192.168.1.1",
            "service": "ssh",
            "attack_type": "brute_force"
        }
        
        manager.logs = [mock_log]
        
        # Test JSON export
        with patch('builtins.open', create=True) as mock_open:
            success = manager.export_logs("test_logs.json", "json")
            assert success
    
    def test_available_services(self):
        """Test getting available services"""
        manager = HoneypotManager()
        
        services = manager.get_available_services()
        assert "ssh" in services
        assert "http" in services
        assert "ftp" in services
    
    def test_deployed_services(self):
        """Test getting deployed services"""
        manager = HoneypotManager()
        
        # Initially no services deployed
        deployed = manager.get_deployed_services()
        assert len(deployed) == 0
        
        # Add a mock service
        manager.servers["ssh"] = Mock()
        deployed = manager.get_deployed_services()
        assert "ssh" in deployed
        assert len(deployed) == 1

if __name__ == "__main__":
    pytest.main([__file__])
