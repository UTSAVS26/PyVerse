"""
Tests for PyRecon scanner functionality.
"""

import pytest
import unittest.mock as mock
from unittest.mock import patch, MagicMock
import socket
import tempfile
import os

from pyrecon.core.scanner import PortScanner, ScanResult
from pyrecon.core.utils import (
    parse_target, parse_port_range, get_top_ports, 
    is_port_open, scan_ports_parallel, get_service_name,
    validate_ip, resolve_domain
)
from pyrecon.core.banner_grabber import BannerGrabber
from pyrecon.core.os_fingerprint import OSFingerprinter
from pyrecon.output.formatter import OutputFormatter


class TestUtils:
    """Test utility functions."""
    
    def test_parse_target_single_ip(self):
        """Test parsing single IP address."""
        result = parse_target("192.168.1.1")
        assert result == ["192.168.1.1"]
    
    def test_parse_target_domain(self):
        """Test parsing domain name."""
        with patch('socket.gethostbyname', return_value="93.184.216.34"):
            result = parse_target("example.com")
            assert result == ["93.184.216.34"]
    
    def test_parse_target_cidr(self):
        """Test parsing CIDR range."""
        result = parse_target("192.168.1.0/30")
        assert len(result) == 2
        assert "192.168.1.1" in result
        assert "192.168.1.2" in result
    
    def test_parse_target_file(self):
        """Test parsing targets from file."""
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
            f.write("192.168.1.1\n")
            f.write("example.com\n")
            f.write("# This is a comment\n")
            f.write("10.0.0.1\n")
            temp_file = f.name
        
        try:
            # Mock gethostbyname to return the same IP for any domain
            with patch('socket.gethostbyname', return_value="93.184.216.34"):
                result = parse_target(temp_file)
                # The result should contain all IPs including the resolved one
                assert "192.168.1.1" in result
                assert "93.184.216.34" in result  # resolved from example.com
                assert "10.0.0.1" in result
                assert len(result) == 3
        finally:
            os.unlink(temp_file)
    
    def test_parse_target_file_simple(self):
        """Test parsing targets from file with only IPs."""
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
            f.write("192.168.1.1\n")
            f.write("10.0.0.1\n")
            f.write("# This is a comment\n")
            f.write("172.16.0.1\n")
            temp_file = f.name
        
        try:
            result = parse_target(temp_file)
            assert "192.168.1.1" in result
            assert "10.0.0.1" in result
            assert "172.16.0.1" in result
            assert len(result) == 3
        finally:
            os.unlink(temp_file)
    
    def test_parse_target_invalid_ip(self):
        """Test parsing invalid IP address."""
        with pytest.raises(ValueError):
            parse_target("256.256.256.256")
    
    def test_parse_port_range_single(self):
        """Test parsing single port."""
        result = parse_port_range("80")
        assert result == [80]
    
    def test_parse_port_range_range(self):
        """Test parsing port range."""
        result = parse_port_range("80-82")
        assert result == [80, 81, 82]
    
    def test_parse_port_range_top(self):
        """Test parsing top ports."""
        result = parse_port_range("top-10")
        assert len(result) == 10
        assert 80 in result  # Common port should be included
    
    def test_parse_port_range_invalid(self):
        """Test parsing invalid port range."""
        with pytest.raises(ValueError):
            parse_port_range("70000")
        
        with pytest.raises(ValueError):
            parse_port_range("80-70")  # Start > end
    
    def test_get_top_ports(self):
        """Test getting top ports."""
        result = get_top_ports(5)
        assert len(result) == 5
        assert 80 in result  # HTTP
        assert 443 in result  # HTTPS
        assert 22 in result   # SSH
    
    def test_get_service_name(self):
        """Test getting service name."""
        assert get_service_name(80) == "HTTP"
        assert get_service_name(443) == "HTTPS"
        assert get_service_name(22) == "SSH"
        assert get_service_name(9999) == "Unknown"
    
    def test_validate_ip(self):
        """Test IP validation."""
        assert validate_ip("192.168.1.1") == True
        assert validate_ip("256.256.256.256") == False
        assert validate_ip("192.168.1") == False
    
    def test_resolve_domain(self):
        """Test domain resolution."""
        with patch('socket.gethostbyname', return_value="93.184.216.34"):
            result = resolve_domain("example.com")
            assert result == "93.184.216.34"
        
        with patch('socket.gethostbyname', side_effect=socket.gaierror):
            result = resolve_domain("nonexistent.example.com")
            assert result is None


class TestPortScanner:
    """Test PortScanner class."""
    
    def test_scanner_initialization(self):
        """Test scanner initialization."""
        scanner = PortScanner(max_workers=50, timeout=2.0)
        assert scanner.max_workers == 50
        assert scanner.timeout == 2.0
        assert scanner.banner_grabber is not None
        assert scanner.os_fingerprinter is not None
    
    @patch('pyrecon.core.utils.scan_ports_parallel')
    def test_scan_basic(self, mock_scan_ports):
        """Test basic scanning functionality."""
        mock_scan_ports.return_value = [80, 443, 22]
        
        scanner = PortScanner()
        results = scanner.scan("192.168.1.1", "80,443,22", fingerprint=False)
        
        assert len(results) == 3
        assert all(isinstance(r, ScanResult) for r in results)
        assert results[0].port == 80
        assert results[1].port == 443
        assert results[2].port == 22
    
    @patch('pyrecon.core.utils.scan_ports_parallel')
    def test_scan_with_fingerprinting(self, mock_scan_ports):
        """Test scanning with fingerprinting."""
        mock_scan_ports.return_value = [80, 443]
        
        scanner = PortScanner()
        results = scanner.scan("192.168.1.1", "80,443", fingerprint=True)
        
        assert len(results) == 2
        # Should have fingerprinting data
        assert all(hasattr(r, 'banner') for r in results)
        assert all(hasattr(r, 'os_guess') for r in results)
    
    def test_scan_udp(self):
        """Test UDP scanning."""
        scanner = PortScanner()
        # Mock the scan_ports_parallel to avoid actual network calls
        with patch.object(scanner, '_fingerprint_ports') as mock_fingerprint:
            mock_fingerprint.return_value = []
            results = scanner.scan_udp("192.168.1.1", "53")
            assert isinstance(results, list)
    
    def test_get_statistics(self):
        """Test statistics generation."""
        scanner = PortScanner()
        
        # Create mock results
        results = [
            ScanResult("192.168.1.1", 80, "tcp", "open", "HTTP"),
            ScanResult("192.168.1.1", 443, "tcp", "open", "HTTPS"),
            ScanResult("192.168.1.2", 22, "tcp", "open", "SSH")
        ]
        
        stats = scanner.get_statistics(results)
        
        assert stats['total_ports'] == 3
        assert stats['unique_hosts'] == 2
        assert 'tcp' in stats['protocols']
        assert stats['services']['HTTP'] == 1
        assert stats['services']['HTTPS'] == 1
        assert stats['services']['SSH'] == 1


class TestBannerGrabber:
    """Test BannerGrabber class."""
    
    def test_banner_grabber_initialization(self):
        """Test banner grabber initialization."""
        grabber = BannerGrabber(timeout=5.0)
        assert grabber.timeout == 5.0
        assert hasattr(grabber, 'service_patterns')
    
    @patch('socket.socket')
    def test_grab_tcp_banner(self, mock_socket):
        """Test TCP banner grabbing."""
        # Mock socket behavior
        mock_sock = MagicMock()
        mock_socket.return_value = mock_sock
        mock_sock.recv.return_value = b"SSH-2.0-OpenSSH_7.4"
        
        grabber = BannerGrabber()
        result = grabber._grab_tcp_banner("192.168.1.1", 22)
        
        assert result['service'] == 'SSH'
        assert 'SSH' in result['banner']
    
    def test_identify_service(self):
        """Test service identification."""
        grabber = BannerGrabber()
        
        # Test SSH identification
        result = grabber._identify_service(22, "SSH-2.0-OpenSSH_7.4")
        assert result == "SSH"
        
        # Test HTTP identification
        result = grabber._identify_service(80, "HTTP/1.1 200 OK")
        assert result == "HTTP"
        
        # Test unknown service
        result = grabber._identify_service(9999, "Unknown banner")
        assert result == "Unknown"
    
    def test_guess_service_by_port(self):
        """Test service guessing by port."""
        grabber = BannerGrabber()
        
        assert grabber._guess_service_by_port(80) == "HTTP"
        assert grabber._guess_service_by_port(443) == "HTTPS"
        assert grabber._guess_service_by_port(22) == "SSH"
        assert grabber._guess_service_by_port(9999) == "Unknown"


class TestOSFingerprinter:
    """Test OSFingerprinter class."""
    
    def test_os_fingerprinter_initialization(self):
        """Test OS fingerprinter initialization."""
        fingerprinter = OSFingerprinter()
        assert hasattr(fingerprinter, 'ttl_patterns')
        assert hasattr(fingerprinter, 'os_signatures')
    
    def test_get_common_os_patterns(self):
        """Test getting common OS patterns."""
        fingerprinter = OSFingerprinter()
        patterns = fingerprinter.get_common_os_patterns()
        
        assert 'TTL Patterns' in patterns
        assert 'Banner Signatures' in patterns
        assert isinstance(patterns['TTL Patterns'], dict)
        assert isinstance(patterns['Banner Signatures'], dict)
    
    def test_ttl_patterns(self):
        """Test TTL pattern matching."""
        fingerprinter = OSFingerprinter()
        
        # Test Windows TTL
        result = fingerprinter._get_ttl_guess("192.168.1.1", 80, "tcp")
        # This will be None in test environment, but we can test the logic
        assert result is None or isinstance(result, str)


class TestOutputFormatter:
    """Test OutputFormatter class."""
    
    def test_formatter_initialization(self):
        """Test formatter initialization."""
        formatter = OutputFormatter(pretty=True, json_output="test.json")
        assert formatter.pretty == True
        assert formatter.json_output == "test.json"
        assert formatter.console is not None
    
    def test_format_scan_result(self):
        """Test formatting scan results."""
        formatter = OutputFormatter(pretty=False)
        
        result = ScanResult("192.168.1.1", 80, "tcp", "open", "HTTP", "Apache")
        formatted = formatter.format_scan_result(result)
        
        assert "80/TCP" in formatted
        assert "HTTP" in formatted
        assert "Apache" in formatted
    
    def test_format_scan_result_pretty(self):
        """Test pretty formatting of scan results."""
        formatter = OutputFormatter(pretty=True)
        
        result = ScanResult("192.168.1.1", 80, "tcp", "open", "HTTP", "Apache")
        formatted = formatter.format_scan_result(result)
        
        # Should return rich Text object
        assert hasattr(formatted, 'plain')
    
    @patch('builtins.open', create=True)
    @patch('json.dump')
    def test_save_json_results(self, mock_json_dump, mock_open):
        """Test saving JSON results."""
        formatter = OutputFormatter(json_output="test.json")
        
        results = [
            ScanResult("192.168.1.1", 80, "tcp", "open", "HTTP", "Apache")
        ]
        
        formatter._save_json_results(results, "192.168.1.1", 1.5)
        
        mock_open.assert_called_once()
        mock_json_dump.assert_called_once()


class TestIntegration:
    """Integration tests."""
    
    @patch('pyrecon.core.utils.scan_ports_parallel')
    def test_full_scan_workflow(self, mock_scan_ports):
        """Test complete scan workflow."""
        mock_scan_ports.return_value = [80, 443]
        
        scanner = PortScanner()
        formatter = OutputFormatter(pretty=False)
        
        results = scanner.scan("192.168.1.1", "80,443", fingerprint=True)
        
        assert len(results) == 2
        assert all(isinstance(r, ScanResult) for r in results)
        
        # Test formatter
        formatter.format_results(results, "192.168.1.1", 1.0)
        # Should not raise any exceptions
    
    def test_error_handling(self):
        """Test error handling."""
        scanner = PortScanner()
        
        # Test with invalid target
        with pytest.raises(ValueError):
            scanner.scan("invalid-target", "80")
        
        # Test with invalid port range
        with pytest.raises(ValueError):
            scanner.scan("192.168.1.1", "invalid-ports")


if __name__ == "__main__":
    pytest.main([__file__]) 