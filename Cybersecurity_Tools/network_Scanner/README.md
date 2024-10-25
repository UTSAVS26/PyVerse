<!DOCTYPE html>
<html lang="en">

<body>
<div class="container">
    <h1>🔍 Network Vulnerability Scanner</h1>
    <p>A powerful yet simple network vulnerability scanner built in Python! This tool is designed to help identify open ports, detect running services, and retrieve service banners to assist with vulnerability assessment. Featuring multithreading for efficiency and SSH support for secure banner grabbing, it’s ideal for network admins, security enthusiasts, and developers.</p>

   <h2>📋 Features</h2>
    <ul>
        <li>🔎 <strong>Port Scanning</strong>: Scans a specified range of ports on a target IP to identify open ports.</li>
        <li>🛠 <strong>Service Detection</strong>: Identifies common services (HTTP, FTP, SSH, etc.) on open ports.</li>
        <li>📜 <strong>Banner Grabbing</strong>: Retrieves banners from services to reveal potential vulnerabilities.</li>
        <li>🔐 <strong>SSH Banner Fetching</strong>: Uses paramiko for secure SSH banner retrieval.</li>
        <li>⚡ <strong>Multithreaded Scanning</strong>: Scans multiple ports concurrently to maximize speed and efficiency.</li>
    </ul>
    <h2>🚀 Benefits</h2>
    <ul>
        <li><strong>Enhanced Network Security</strong>: Quickly identify open ports and services on your network, making it easier to lock down unused or unnecessary services.</li>
        <li><strong>Risk Assessment</strong>: Fetches banners that may contain version information, helping you spot potentially vulnerable services and take action.</li>
        <li><strong>Speed & Efficiency</strong>: Utilizing multithreading, the scanner covers a large range of ports in a short time, providing fast, actionable results.</li>
        <li><strong>Customizable</strong>: The script can be adapted or expanded based on your needs, offering flexibility for both basic and advanced users.</li>
        <li><strong>Educational Value</strong>: Great for learning the fundamentals of network security, service detection, and banner grabbing!</li>
    </ul>
    <h2>🛠 Tech Stack</h2>
    <p><strong>Python</strong>: Core programming language for the scanner<br>
    <strong>Socket Module</strong>: Manages network connections and performs port scanning<br>
    <strong>ThreadPoolExecutor</strong>: Handles multithreading for efficient scanning<br>
    <strong>Paramiko</strong>: Enables SSH connections and secure banner retrieval on port 22</p>

 <h2>📦 Installation</h2>
    <p><strong>1. Clone the repository:</strong></p>
    <pre><code>https://github.com/UTSAVS26/PyVerse/tree/main/Cybersecurity_Tools
    </code></pre>
     <pre><code>cd network_Scanner
    </code></pre>
    <p><strong>2. Install dependencies:</strong></p>
    <pre><code>pip install paramiko</code></pre>
    <pre><code>pip install scapy</code></pre>

  <h2>🚀 Usage</h2>
  <p>Run the scanner by providing the target IP address and the range of ports to scan.</p>
    <pre><code>python main.py</code></pre>
    <p><strong>Example Run:</strong></p>
    <div class="code-snippet">
        <img src="scanner_screenshot.png" alt="Network Vulnerability Scanner Screenshot" width="600">
    </div>

<h2>⚠️ Important Notes</h2>
<ul>
        <li><strong>Use Responsibly</strong>: Scanning networks without permission is prohibited by law.</li>
        <li><strong>SSH Support</strong>: Ensure paramiko is installed via <code>pip install paramiko</code> for SSH banner grabbing to work.</li>
        <li><strong>Limitations</strong>: Some services may block or prevent banner retrieval to limit exposure.</li>
</ul>
