# 🔍 PyRecon: High-Speed Port Scanner & Service Fingerprinter

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Production%20Ready-brightgreen.svg)]()

A fast, multithreaded Python-based TCP/UDP port scanner with intelligent service and OS fingerprinting capabilities. PyRecon is designed for ethical hacking, security research, and network reconnaissance tasks.

## 📋 Table of Contents

- [Features](#-features)
- [Installation](#-installation)
- [Quick Start](#-quick-start)
- [Usage](#-usage)
- [Examples](#-examples)
- [Advanced Features](#-advanced-features)
- [Configuration](#-configuration)
- [API Reference](#-api-reference)
- [Testing](#-testing)
- [Contributing](#-contributing)
- [Security](#-security)
- [License](#-license)

## 🚀 Features

### Core Scanning Capabilities
- **⚡ High-Speed Scanning**: Multithreaded TCP and UDP port scanning across IP ranges
- **🎯 Flexible Targeting**: Support for single IPs, domains, CIDR ranges, and target files
- **🔢 Smart Port Specification**: Single ports, ranges, lists, and top-N common ports
- **🔄 Protocol Support**: Both TCP and UDP scanning with configurable timeouts

### Service Fingerprinting
- **🧠 Intelligent Detection**: Automatic service identification based on banners and protocols
- **📡 Protocol Parsing**: HTTP, HTTPS, SSH, FTP, SMTP, DNS, and more
- **🔍 Banner Grabbing**: Detailed service information and version detection
- **🌐 Web Technology Stack**: HTTP server, framework, and technology detection

### OS Fingerprinting
- **💻 TTL Analysis**: Operating system detection based on Time-To-Live values
- **🔬 Pattern Matching**: OS identification through response patterns
- **📊 Confidence Scoring**: Reliability assessment of fingerprinting results

### Output & Reporting
- **🎨 Rich Terminal Output**: Beautiful, colorized output with tables and panels
- **📄 JSON Export**: Structured data export for programmatic analysis
- **📊 Statistics**: Service distribution and scan statistics
- **⏱️ Progress Tracking**: Real-time progress updates during scanning

### Plugin System
- **🔌 Extensible Architecture**: Easy to add new fingerprinting modules
- **🌐 HTTP Fingerprinting**: Web server and framework detection
- **🔒 TLS Certificate Analysis**: SSL certificate parsing and security assessment

## 📦 Installation

### Prerequisites
- Python 3.8 or higher
- pip (Python package installer)

### Method 1: Direct Installation
```bash
# Clone the repository
git clone <repository-url>
cd PyRecon

# Install dependencies
pip install -r requirements.txt

# Run installation test
python test_installation.py
```

### Method 2: Development Installation
```bash
# Clone and install in development mode
git clone <repository-url>
cd PyRecon
pip install -e .

# Install development dependencies
pip install -r requirements.txt
```

### Method 3: Using pip (when published)
```bash
pip install pyrecon
```

## 🚀 Quick Start

### Basic Port Scan
```bash
# Scan common ports on a single host
python -c "from pyrecon.cli.main import cli; cli()" scan 192.168.1.1 --top-ports 100

# Scan specific ports
python -c "from pyrecon.cli.main import cli; cli()" scan example.com -p 80,443,22,21
```

### Scan with Fingerprinting
```bash
# Full scan with service and OS fingerprinting
python -c "from pyrecon.cli.main import cli; cli()" scan target.com -p 1-1024 --fingerprint
```

### Save Results
```bash
# Export results to JSON
python -c "from pyrecon.cli.main import cli; cli()" scan target.com --json results.json
```

## 📖 Usage

### Command Line Interface

PyRecon provides a comprehensive CLI with multiple commands:

#### Main Scan Command
```bash
pyrecon scan <target> [OPTIONS]
```

**Arguments:**
- `target`: IP address, domain name, CIDR range, or file path

**Options:**
- `-p, --ports`: Port specification (default: "top-100")
- `--protocol`: Protocol to use - tcp/udp (default: "tcp")
- `--fingerprint`: Enable service fingerprinting
- `--pretty`: Use pretty terminal output (default: True)
- `--json`: Save results to JSON file
- `--workers`: Maximum worker threads (default: 100)
- `--timeout`: Connection timeout in seconds (default: 1.0)
- `-f, --file`: Read targets from file

#### Quick Commands
```bash
# Quick scan without fingerprinting
pyrecon quick <target> [OPTIONS]

# Full scan with fingerprinting
pyrecon full <target> [OPTIONS]

# UDP scan
pyrecon udp <target> [OPTIONS]

# Show version
pyrecon version

# Show help
pyrecon help
```

### Port Specifications

PyRecon supports multiple port specification formats:

```bash
# Single port
-p 80

# Port range
-p 1-1024

# Port list
-p 80,443,8080,8443

# Top N common ports
-p top-100

# Mixed specification
-p 80,443,8080-8090,top-10
```

### Target Specifications

Support for various target formats:

```bash
# Single IP address
192.168.1.1

# Domain name
example.com

# CIDR range
192.168.1.0/24

# File with targets
-f targets.txt
```

## 📝 Examples

### Basic Scanning Examples

#### 1. Quick Network Scan
```bash
# Scan common ports on a network
python -c "from pyrecon.cli.main import cli; cli()" scan 192.168.1.0/24 --top-ports 50
```

#### 2. Web Server Analysis
```bash
# Comprehensive web server scan
python -c "from pyrecon.cli.main import cli; cli()" scan example.com -p 80,443,8080,8443 --fingerprint --json web_scan.json
```

#### 3. Service Discovery
```bash
# Find all services on a host
python -c "from pyrecon.cli.main import cli; cli()" scan target.com -p 1-65535 --fingerprint
```

#### 4. UDP Service Detection
```bash
# Scan UDP services
python -c "from pyrecon.cli.main import cli; cli()" scan 10.0.0.1 --protocol udp -p 53,67,68,161,162
```

### Advanced Usage Examples

#### 5. Multiple Target Scan
```bash
# Create targets file
echo "192.168.1.1" > targets.txt
echo "example.com" >> targets.txt
echo "10.0.0.0/24" >> targets.txt

# Scan all targets
python -c "from pyrecon.cli.main import cli; cli()" scan -f targets.txt --fingerprint
```

#### 6. Custom Configuration
```bash
# High-speed scan with custom settings
python -c "from pyrecon.cli.main import cli; cli()" scan target.com \
  -p top-1000 \
  --workers 200 \
  --timeout 0.5 \
  --fingerprint \
  --json detailed_scan.json
```

#### 7. Security Assessment
```bash
# Comprehensive security scan
python -c "from pyrecon.cli.main import cli; cli()" scan target.com \
  -p 21,22,23,25,53,80,110,143,443,993,995,1433,1521,3306,3389,5432,5900,6379,8080,8443,27017 \
  --fingerprint \
  --json security_report.json
```

## 🔧 Advanced Features

### Plugin System

PyRecon includes a plugin system for extended functionality:

#### HTTP Fingerprinting Plugin
```python
from pyrecon.plugins.http_fingerprint import HTTPFingerprinter

fingerprinter = HTTPFingerprinter()
result = fingerprinter.fingerprint("example.com", 443, use_ssl=True)
print(f"Server: {result['server']}")
print(f"Framework: {result['framework']}")
print(f"Technologies: {result['technologies']}")
```

#### TLS Certificate Analysis
```python
from pyrecon.plugins.tls_parser import TLSParser

parser = TLSParser()
cert_info = parser.parse_certificate("example.com", 443)
print(f"Subject: {cert_info['subject']}")
print(f"Issuer: {cert_info['issuer']}")
print(f"Security Score: {cert_info['security_analysis']['security_score']}")
```

### Programmatic Usage

#### Basic Scanner Usage
```python
from pyrecon.core.scanner import PortScanner
from pyrecon.output.formatter import OutputFormatter

# Initialize scanner
scanner = PortScanner(max_workers=100, timeout=1.0)

# Perform scan
results = scanner.scan("192.168.1.1", "top-100", fingerprint=True)

# Format output
formatter = OutputFormatter(pretty=True)
formatter.format_results(results, "192.168.1.1", 1.5)
```

#### Custom Banner Grabbing
```python
from pyrecon.core.banner_grabber import BannerGrabber

grabber = BannerGrabber(timeout=3.0)
banner_info = grabber.grab_banner("example.com", 80, "tcp")
print(f"Service: {banner_info['service']}")
print(f"Banner: {banner_info['banner']}")
```

## ⚙️ Configuration

### Environment Variables
```bash
# Set default timeout
export PYRECON_TIMEOUT=2.0

# Set default workers
export PYRECON_WORKERS=50

# Enable debug mode
export PYRECON_DEBUG=1
```

### Configuration File
Create a `pyrecon.conf` file:
```ini
[scanner]
default_timeout = 1.0
default_workers = 100
default_protocol = tcp

[output]
default_pretty = true
default_json_output = false

[plugins]
enable_http_fingerprint = true
enable_tls_parser = true
```

## 📊 Output Formats

### Terminal Output
```
🔍 PyRecon Scan Results
Target: example.com
Scan Time: 2.34s
Open Ports: 3

┌──────┬──────────┬──────────┬─────────────────────────────┬─────────────┐
│ Port │ Protocol │ Service  │ Banner                      │ OS Guess    │
├──────┼──────────┼──────────┼─────────────────────────────┼─────────────┤
│  80  │   TCP    │  HTTP    │ Apache/2.4.41 (Ubuntu)     │ Linux       │
│ 443  │   TCP    │  HTTPS   │ nginx/1.18.0               │ Linux       │
│  22  │   TCP    │   SSH    │ SSH-2.0-OpenSSH_8.2p1     │ Linux       │
└──────┴──────────┴──────────┴─────────────────────────────┴─────────────┘
```

### JSON Output
```json
{
  "target": "example.com",
  "timestamp": "2024-01-15T10:30:45Z",
  "scan_time": 2.34,
  "open_ports": [
    {
      "port": 80,
      "protocol": "tcp",
      "service": "HTTP",
      "banner": "Apache/2.4.41 (Ubuntu)",
      "os_guess": "Linux (TTL=64)",
      "response_time": 0.023
    }
  ]
}
```

## 🧪 Testing

### Run All Tests
```bash
# Run comprehensive test suite
python -m pytest tests/ -v

# Run specific test categories
python -m pytest tests/test_scanner.py::TestUtils -v
python -m pytest tests/test_scanner.py::TestPortScanner -v
```

### Installation Test
```bash
# Verify installation
python test_installation.py
```

### Demo Script
```bash
# Run interactive demo
python demo.py
```

## 🔒 Security Considerations

### Ethical Usage
- **Authorization Required**: Only scan networks you own or have explicit permission to test
- **Educational Purpose**: This tool is designed for security research and education
- **Legal Compliance**: Ensure compliance with local laws and regulations

### Best Practices
- **Rate Limiting**: Use appropriate timeouts and worker limits
- **Network Impact**: Be mindful of network load during scanning
- **Documentation**: Keep records of authorized scanning activities

### Responsible Disclosure
- Report security vulnerabilities to appropriate parties
- Follow responsible disclosure timelines
- Provide detailed technical information for fixes

## 🤝 Contributing

### Development Setup
```bash
# Clone repository
git clone <repository-url>
cd PyRecon

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install development dependencies
pip install -r requirements.txt
pip install -e .

# Run tests
python -m pytest tests/ -v
```

### Code Style
- Follow PEP 8 guidelines
- Use type hints for all functions
- Write comprehensive docstrings
- Add tests for new features

### Pull Request Process
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Ensure all tests pass
6. Submit a pull request

## 📚 API Reference

### Core Classes

#### PortScanner
```python
class PortScanner:
    def __init__(self, max_workers: int = 100, timeout: float = 1.0)
    def scan(self, target: str, ports: str = "top-100", 
             protocol: str = "tcp", fingerprint: bool = False) -> List[ScanResult]
    def quick_scan(self, target: str, ports: str = "top-100") -> List[ScanResult]
    def full_scan(self, target: str, ports: str = "1-1024") -> List[ScanResult]
    def scan_udp(self, target: str, ports: str = "top-100") -> List[ScanResult]
```

#### BannerGrabber
```python
class BannerGrabber:
    def __init__(self, timeout: float = 3.0)
    def grab_banner(self, host: str, port: int, protocol: str = 'tcp') -> Dict[str, Any]
```

#### OutputFormatter
```python
class OutputFormatter:
    def __init__(self, pretty: bool = True, json_output: Optional[str] = None)
    def format_results(self, results: List[ScanResult], target: str, scan_time: float) -> None
```

### Data Structures

#### ScanResult
```python
@dataclass
class ScanResult:
    host: str
    port: int
    protocol: str
    status: str  # 'open', 'closed', 'filtered'
    service: str
    banner: Optional[str] = None
    os_guess: Optional[str] = None
    tls_info: Optional[Dict] = None
    response_time: Optional[float] = None
```

## 📈 Performance

### Benchmarks
- **Scan Speed**: ~1000 ports/second on localhost
- **Memory Usage**: ~50MB for typical scans
- **Network Efficiency**: Configurable timeouts and connection limits
- **Concurrency**: Up to 200 concurrent workers

### Optimization Tips
- Use appropriate worker counts for your network
- Adjust timeouts based on network conditions
- Use port ranges instead of individual ports for large scans
- Enable fingerprinting only when needed

## 🐛 Troubleshooting

### Common Issues

#### Import Errors
```bash
# Ensure all dependencies are installed
pip install -r requirements.txt

# Check Python version
python --version  # Should be 3.8+
```

#### Network Timeouts
```bash
# Increase timeout for slow networks
python -c "from pyrecon.cli.main import cli; cli()" scan target.com --timeout 5.0
```

#### Permission Errors
```bash
# On Linux/macOS, you might need sudo for certain scans
sudo python -c "from pyrecon.cli.main import cli; cli()" scan target.com
```

### Debug Mode
```bash
# Enable debug output
export PYRECON_DEBUG=1
python -c "from pyrecon.cli.main import cli; cli()" scan target.com
```

## 👨‍💻 Author

**Shivansh Katiyar** - SSOC Participant
- **Project**: PyRecon - High-Speed Port Scanner & Service Fingerprinter


*Remember: With great power comes great responsibility. Always scan ethically and legally* 