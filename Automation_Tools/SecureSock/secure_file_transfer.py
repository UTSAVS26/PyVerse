import asyncio
import websockets
import os
import hashlib
import json
import sys
import secrets
import jwt
import zipfile
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import padding
from tqdm import tqdm
from PyQt5.QtWidgets import QApplication, QFileDialog, QVBoxLayout, QPushButton, QWidget, QLabel, QLineEdit

CHUNK_SIZE = 1024 * 1024  # 1MB
SECRET = 'supersecretkey'

# Helper functions

def sha256_file(filepath):
    hash_sha256 = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_sha256.update(chunk)
    return hash_sha256.hexdigest()

def get_cipher(key, iv):
    return Cipher(algorithms.AES(key), modes.CBC(iv), backend=default_backend())

def encrypt_chunk(chunk, key, iv):
    cipher = get_cipher(key, iv)
    padder = padding.PKCS7(128).padder()
    padded_data = padder.update(chunk) + padder.finalize()
    encryptor = cipher.encryptor()
    return encryptor.update(padded_data) + encryptor.finalize()


def decrypt_chunk(chunk, key, iv):
    cipher = get_cipher(key, iv)
    decryptor = cipher.decryptor()
    padded_data = decryptor.update(chunk) + decryptor.finalize()
    unpadder = padding.PKCS7(128).unpadder()
    return unpadder.update(padded_data) + unpadder.finalize()
def compress_file(filepath):
    zip_path = filepath + ".zip"
    with zipfile.ZipFile(zip_path, 'w') as zipf:
        zipf.write(filepath, os.path.basename(filepath))
    return zip_path

# Server

async def handle_client(websocket):
    print("Client connected")
    token = await websocket.recv()
    try:
        jwt.decode(token, SECRET, algorithms=['HS256'])
    except jwt.InvalidTokenError:
        await websocket.close()
        print("Invalid token. Connection closed.")
        return

    metadata_json = await websocket.recv()
    metadata = json.loads(metadata_json)

    filename = metadata['filename']
    filesize = metadata['filesize']

    # Sanitize filename to prevent path traversal
    filename = os.path.basename(filename)
    # Remove any remaining path separators
    filename = filename.replace('/', '_').replace('\\', '_')

    # Limit file size to prevent DoS (e.g., 100MB)
    MAX_FILE_SIZE = 100 * 1024 * 1024
    if filesize > MAX_FILE_SIZE:
        await websocket.send("File too large")
        await websocket.close()
        return

    sha256_hash = metadata['sha256']
    key = bytes.fromhex(metadata['key'])
    iv = bytes.fromhex(metadata['iv'])

    cipher = get_cipher(key, iv)
    output_path = "received_" + filename
    received_bytes = 0

    if os.path.exists(output_path):
        received_bytes = os.path.getsize(output_path)

    await websocket.send(str(received_bytes))

    with open(output_path, 'ab') as f:
        with tqdm(total=filesize, initial=received_bytes, unit='B', unit_scale=True, desc='Receiving') as pbar:
            while received_bytes < filesize:
                encrypted_chunk = await websocket.recv()
                decrypted_chunk = decrypt_chunk(encrypted_chunk, cipher)
                f.write(decrypted_chunk)
                received_bytes += len(decrypted_chunk)
                pbar.update(len(decrypted_chunk))

    print("File received. Verifying hash...")
    received_hash = sha256_file(output_path)
    if received_hash == sha256_hash:
        print("Success: File integrity verified.")
    else:
        print("Warning: File hash mismatch!")

# Client

async def send_file(uri, filepath, key, iv, token):
    cipher = get_cipher(key, iv)
    filepath = compress_file(filepath)
    filesize = os.path.getsize(filepath)
    sha256 = sha256_file(filepath)

    metadata = {
        'filename': os.path.basename(filepath),
        'filesize': filesize,
        'sha256': sha256,
        'key': key.hex(),
        'iv': iv.hex(),
    }

    async with websockets.connect(uri, max_size=None) as websocket:
        await websocket.send(token)
        await websocket.send(json.dumps(metadata))
        offset = int(await websocket.recv())

        with open(filepath, 'rb') as f:
            f.seek(offset)
            with tqdm(total=filesize, initial=offset, unit='B', unit_scale=True, desc='Sending') as pbar:
                while chunk := f.read(CHUNK_SIZE):
                    encrypted_chunk = encrypt_chunk(chunk, cipher)
                    await websocket.send(encrypted_chunk)
                    pbar.update(len(chunk))

        print("File sent successfully.")

# GUI
class FileTransferGUI(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("SecureSock File Sender")
        self.layout = QVBoxLayout()

        self.label = QLabel("Select a file to send:")
        self.pass_label = QLabel("Enter password:")
        self.pass_input = QLineEdit()
        self.pass_input.setEchoMode(QLineEdit.Password)
        self.button = QPushButton("Choose File and Send")
        self.button.clicked.connect(self.choose_file)

        self.layout.addWidget(self.label)
        self.layout.addWidget(self.pass_label)
        self.layout.addWidget(self.pass_input)
        self.layout.addWidget(self.button)
        self.setLayout(self.layout)

    def choose_file(self):
        filepath, _ = QFileDialog.getOpenFileName(self, "Choose file")
        if filepath:
            key = secrets.token_bytes(32)
            iv = secrets.token_bytes(16)
            password = self.pass_input.text()
            if password:
                token = jwt.encode({'user': 'client'}, SECRET, algorithm='HS256')
                asyncio.get_event_loop().run_until_complete(
                    send_file('ws://localhost:8765', filepath, key, iv, token)
                )

# Entry point
if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == 'server':
        start_server = websockets.serve(handle_client, 'localhost', 8765, max_size=None)
        asyncio.get_event_loop().run_until_complete(start_server)
        print("Server running at ws://localhost:8765")
        asyncio.get_event_loop().run_forever()

    elif len(sys.argv) > 1 and sys.argv[1] == 'client':
        filepath = sys.argv[2]
        key = secrets.token_bytes(32)
        iv = secrets.token_bytes(16)
        token = jwt.encode({'user': 'cli'}, SECRET, algorithm='HS256')
        asyncio.get_event_loop().run_until_complete(send_file('ws://localhost:8765', filepath, key, iv, token))

    else:
        app = QApplication(sys.argv)
        gui = FileTransferGUI()
        gui.show()
        sys.exit(app.exec_())
