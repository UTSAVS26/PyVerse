import asyncio
import websockets
from client.utils import encrypt, decrypt, derive_key
from auth import check_token
import subprocess
import argparse
import json

clients = {}

async def handle_connection(websocket, path, token, key):
    encrypted_auth = await websocket.recv()
    try:
        decrypted_data = decrypt(encrypted_auth, key)
        auth_data = json.loads(decrypted_data.decode('utf-8'))
    except Exception as e:
        await websocket.send(encrypt(b"AUTH_FAILED", key))
        return
    if not check_token(auth_data.get("token"), token):
        await websocket.send(encrypt("AUTH_FAILED", key))
        return
    await websocket.send(encrypt("AUTH_SUCCESS", key))
    while True:
        try:
            enc_cmd = await websocket.recv()
            cmd = decrypt(enc_cmd, key)
            output = subprocess.getoutput(cmd)
            await websocket.send(encrypt(output, key))
        except websockets.ConnectionClosed:
            break

async def main(host, port, token, password):
    key = derive_key(password)
    async with websockets.serve(lambda ws, path: handle_connection(ws, path, token, key), host, port):
        print(f"[*] Server running on {host}:{port}")
        await asyncio.Future()  # Run forever

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--token", required=True)
    parser.add_argument("--password", required=True)
    args = parser.parse_args()
    asyncio.run(main(args.host, args.port, args.token, args.password))
