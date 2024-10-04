## **CLI Based Port Scanner**

### **Disclaimer**

This code is purely meant for learning purposes, and not meant to be used with malicious intent.

### 🎯 **Goal**

Python Project - This CLI tool allows the user to scan specified ports of a host.

Modules Used:
    1. optparse - to enable command-line arguments in the terminal and their parsing
    2. socket - to interact with the host and their ports
	

### 🧾 **Description**

This CLI tool allows the user to scan specified ports of a host, and can be run in a terminal. The user can enter the host's IP address, and the port numbers (separated by comma - do not include any whitespace between commas and numbers). The tool scans ports one by one and returns a summary of findings for each port.

### 🧮 **What I had done!**

1. Imported required libraries.
2. Set up the arguments for the tool using optparse (-H/--Host and -p/--port).
3. Read the user input, and parse them accordingly. 
4. Stored the host IP in a variable, and port numbers in a list.
5. Wrote function for sending a message to a port and checking its response.
6. Wrote a function to iterate through the list of port numbers, and initiate the above scan function for each port.

### 📢 **Conclusion**

On using the help option, we get the following output:
```
Usage: port-scanner.py [options]

Options:
  -h, --help            show this help message and exit
  -H TGHOST, --Host=TGHOST
                        specify target host
  -p TGPORT, --port=TGPORT
                        specify target port[s] separated by comma
```

After entering a valid IP address for `certifiedhacker.com` and 3 ports - namely 21, 22, 80, the following output is displayed:
```
[+] Scan Results for: box5331.bluehost.com

Scanning port 21
[+] 21/tcp open
[+] 220---------- Welcome to Pure-FTPd [privsep] [TLS] ----------
220-You are user number 5 of 150 allo

Scanning port 22
[+] 22/tcp open
[+] SSH-2.0-OpenSSH_7.4
Protocol mismatch.


Scanning port 80
[+] 80/tcp open
[+] HTTP/1.1 400 Bad Request
Date: Sat, 14 Sep 2024 06:07:12 GMT
Server: Apache
Content-Length: 347
```

### ✒️ **Your Signature**

`Shreelu Santosh`
[GitHub Profile](https://github.com/ShreeluSantosh)


	
	
