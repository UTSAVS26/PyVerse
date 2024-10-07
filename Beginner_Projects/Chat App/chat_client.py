import socket
import threading

def receive_messages(client_socket):
    """Receive messages from the server."""
    while True:
        try:
            message = client_socket.recv(1024).decode('utf-8')
            if message:
                print(f"\n{message}")
            else:
                break
        except:
            break

# Set up the client
client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client.connect(('127.0.0.1', 12345))  # Connect to the server

# Start a thread to listen for messages
threading.Thread(target=receive_messages, args=(client,)).start()

while True:
    message = input("You: ")
    if message.lower() == 'exit':
        break
    client.send(message.encode('utf-8'))

client.close()

