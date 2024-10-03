#import libraries
import optparse
from socket import *

#function to connect to host and start scanning the specific port
def connScan(tgHost, tgPort):
    try:
        connSkt = socket(AF_INET, SOCK_STREAM)
        connSkt.connect((tgHost, tgPort))
        connSkt.send(b'HelloPython\r\n')
        results = connSkt.recv(100)
        print(f'[+] {tgPort}/tcp open')
        print(f'[+] {results.decode()}')
        connSkt.close()
    except:
        print(f'[-] {tgPort}/tcp closed')

#function to resolve the host and start port scan
def portScan(tgHost, tgPorts):
    try:
        tgIP = gethostbyname(tgHost)
    except:
        print(f"[-] Cannot resolve '{tgHost}': Unknown host")
        return
    try:
        tgName = gethostbyaddr(tgIP)
        print(f"\n[+] Scan Results for: {tgName[0]}")
    except:
        print(f"\n[+] Scan Results for: {tgIP}")
    setdefaulttimeout(1)
    for tgPort in tgPorts:
        print(f"\nScanning port {tgPort}")
        connScan(tgHost, tgPort)

#main function
def main():
    parser = optparse.OptionParser()
    parser.add_option('-H', "--Host", dest='tgHost', type='string', help='specify target host')
    parser.add_option('-p', "--port", dest='tgPort', type='string', help='specify target port[s] separated by comma')
    (options, args) = parser.parse_args()
    tgHost = options.tgHost
    tgPorts = str(options.tgPort).split(",")
    if (tgHost == None) | (tgPorts[0] == None):
        print("[-] You must specify a target host and port[s]")
        exit(0)
    tgPorts = [int(port) for port in tgPorts]
    portScan(tgHost, tgPorts)

if __name__ == "__main__":
    main()