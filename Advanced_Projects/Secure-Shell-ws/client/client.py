import asyncio
import websockets
from utils import encrypt, decrypt, derive_key
import argparse
import getpass
import json

async def run_client(uri, token, key):
    async with websockets.connect(uri) as websocket:
        await websocket.send(encrypt(json.dumps({"token": token}), key))
        response = decrypt(await websocket.recv(), key)
        if response != "AUTH_SUCCESS":
            print("Authentication failed.")
            return
        print("[*] Connected to remote shell. Type 'exit' to quit.")
        while True:
            cmd = input(">>> ")
            if cmd.lower() == "exit":
                break
            await websocket.send(encrypt(cmd, key))
            result = decrypt(await websocket.recv(), key)
            print(result)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", required=True)
    parser.add_argument("--port", required=True)
    parser.add_argument("--token", required=True)
    parser.add_argument("--password", required=False)
    args = parser.parse_args()
    password = args.password or getpass.getpass("Password: ")
    key = derive_key(password)
    asyncio.run(run_client(f"ws://{args.host}:{args.port}", args.token, key))
