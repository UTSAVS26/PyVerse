import asyncio
import websockets
from utils import encrypt, decrypt, derive_key
import argparse
import getpass
import json

async def run_client(uri, token, key):
    try:
        async with websockets.connect(uri) as websocket:
            try:
                await websocket.send(encrypt(json.dumps({"token": token}).encode(), key))
                response_data = await websocket.recv()
                response = decrypt(response_data, key).decode()
                if response != "AUTH_SUCCESS":
                    print("Authentication failed.")
                    return
                print("[*] Connected to remote shell. Type 'exit' to quit.")
                while True:
                    try:
                        cmd = input(">>> ")
                        if cmd.lower() == "exit":
                            break
                        await websocket.send(encrypt(cmd.encode(), key))
                        result_data = await websocket.recv()
                        result = decrypt(result_data, key).decode()
                        print(result)
                    except (KeyboardInterrupt, EOFError):
                        break
                    except Exception as e:
                        print(f"Error processing command: {e}")
            except Exception as e:
                print(f"Communication error: {e}")
    except Exception as e:
        print(f"Connection error: {e}")
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
