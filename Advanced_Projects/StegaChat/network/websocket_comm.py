import asyncio
import websockets
import os
import json
import ssl
import logging
from typing import Dict, Set, Optional, Callable

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SecureWebSocketServer:
    def __init__(self, host: str = "0.0.0.0", port: int = 8765, 
                 secret_key: Optional[str] = None, use_ssl: bool = False):
        self.host = host
        self.port = port
        self.secret_key = secret_key or os.urandom(32).hex()
        self.use_ssl = use_ssl
        self.clients: Dict[str, websockets.WebSocketServerProtocol] = {}
        self.message_handlers: Set[Callable] = set()
        
    def add_message_handler(self, handler: Callable):
        """Add a message handler function"""
        self.message_handlers.add(handler)
    
    def remove_message_handler(self, handler: Callable):
        """Remove a message handler function"""
        self.message_handlers.discard(handler)
    
    async def authenticate_client(self, websocket, path):
        """Authenticate incoming client connection"""
        try:
            # Simple token-based authentication
            auth_message = await websocket.recv()
            auth_data = json.loads(auth_message)
            
            if auth_data.get('type') == 'auth':
                token = auth_data.get('token')
                if token == self.secret_key:
                    client_id = auth_data.get('client_id', f"client_{len(self.clients)}")
                    self.clients[client_id] = websocket
                    await websocket.send(json.dumps({
                        'type': 'auth_success',
                        'client_id': client_id
                    }))
                    logger.info(f"Client {client_id} authenticated successfully")
                    return client_id
                else:
                    await websocket.send(json.dumps({
                        'type': 'auth_failed',
                        'message': 'Invalid token'
                    }))
                    return None
            else:
                await websocket.send(json.dumps({
                    'type': 'auth_failed',
                    'message': 'Authentication required'
                }))
                return None
        except Exception as e:
            logger.error(f"Authentication error: {e}")
            return None
    
    async def handle_client(self, websocket, path):
        """Handle individual client connection"""
        client_id = await self.authenticate_client(websocket, path)
        if not client_id:
            return
            
        try:
            async for message in websocket:
                try:
                    # Parse message
                    data = json.loads(message)
                    message_type = data.get('type', 'message')
                    
                    # Handle different message types
                    if message_type == 'message':
                        await self.broadcast_message(client_id, data.get('content', ''))
                    elif message_type == 'file':
                        await self.handle_file_message(client_id, data)
                    elif message_type == 'ping':
                        await websocket.send(json.dumps({'type': 'pong'}))
                    
                    # Call registered handlers
                    for handler in self.message_handlers:
                        try:
                            await handler(client_id, data)
                        except Exception as e:
                            logger.error(f"Handler error: {e}")
                            
                except json.JSONDecodeError:
                    logger.warning(f"Invalid JSON from client {client_id}")
                except Exception as e:
                    logger.error(f"Message handling error: {e}")
                    
        except websockets.exceptions.ConnectionClosed:
            logger.info(f"Client {client_id} disconnected")
        except Exception as e:
            logger.error(f"Client {client_id} error: {e}")
        finally:
            # Clean up client
            if client_id in self.clients:
                del self.clients[client_id]
    
    async def broadcast_message(self, sender_id: str, content: str):
        """Broadcast message to all other clients"""
        message = {
            'type': 'message',
            'sender': sender_id,
            'content': content,
            'timestamp': asyncio.get_event_loop().time()
        }
        
        for client_id, client_ws in self.clients.items():
            if client_id != sender_id:
                try:
                    await client_ws.send(json.dumps(message))
                except Exception as e:
                    logger.error(f"Failed to send to {client_id}: {e}")
    
    async def handle_file_message(self, sender_id: str, data: dict):
        """Handle file transfer messages"""
        file_data = {
            'type': 'file',
            'sender': sender_id,
            'filename': data.get('filename'),
            'content': data.get('content'),
            'timestamp': asyncio.get_event_loop().time()
        }
        
        for client_id, client_ws in self.clients.items():
            if client_id != sender_id:
                try:
                    await client_ws.send(json.dumps(file_data))
                except Exception as e:
                    logger.error(f"Failed to send file to {client_id}: {e}")
    
    def start_server(self):
        """Start the WebSocket server"""
        try:
            if self.use_ssl:
                # Create SSL context (you would need proper certificates in production)
                ssl_context = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
                if os.path.exists('cert.pem') and os.path.exists('key.pem'):
                    ssl_context.load_cert_chain('cert.pem', 'key.pem')
                else:
                    logger.warning("SSL certificates not found. SSL disabled.")
                    return websockets.serve(self.handle_client, self.host, self.port)
                return websockets.serve(self.handle_client, self.host, self.port, ssl=ssl_context)
            else:
                return websockets.serve(self.handle_client, self.host, self.port)
        except Exception as e:
            logger.error(f"Failed to start server: {e}")
            raise

class SecureWebSocketClient:
    def __init__(self, uri: str, client_id: str, secret_key: str):
        self.uri = uri
        self.client_id = client_id
        self.secret_key = secret_key
        self.websocket = None
        self.connected = False
        
    async def connect(self):
        """Connect to the WebSocket server"""
        try:
            self.websocket = await websockets.connect(self.uri)
            
            # Authenticate
            auth_message = {
                'type': 'auth',
                'client_id': self.client_id,
                'token': self.secret_key
            }
            await self.websocket.send(json.dumps(auth_message))
            
            response = await self.websocket.recv()
            auth_response = json.loads(response)
            
            if auth_response.get('type') == 'auth_success':
                self.connected = True
                logger.info(f"Connected to server as {self.client_id}")
                return True
            else:
                logger.error(f"Authentication failed: {auth_response.get('message')}")
                return False
                
        except Exception as e:
            logger.error(f"Connection failed: {e}")
            return False
    
    async def send_message(self, content: str):
        """Send a text message"""
        if not self.connected or not self.websocket:
            raise ConnectionError("Not connected to server")
        
        message = {
            'type': 'message',
            'content': content
        }
        await self.websocket.send(json.dumps(message))
    
    async def send_file(self, filename: str, content: str):
        """Send a file message"""
        if not self.connected or not self.websocket:
            raise ConnectionError("Not connected to server")
        
        message = {
            'type': 'file',
            'filename': filename,
            'content': content
        }
        await self.websocket.send(json.dumps(message))
    
    async def receive_messages(self, message_handler: Callable):
        """Receive and handle incoming messages"""
        if not self.connected or not self.websocket:
            raise ConnectionError("Not connected to server")
        
        try:
            async for message in self.websocket:
                try:
                    data = json.loads(message)
                    await message_handler(data)
                except json.JSONDecodeError:
                    logger.warning("Received invalid JSON")
                except Exception as e:
                    logger.error(f"Message handling error: {e}")
        except websockets.exceptions.ConnectionClosed:
            self.connected = False
            logger.info("Connection closed")
        except Exception as e:
            logger.error(f"Receive error: {e}")
            self.connected = False
    
    async def disconnect(self):
        """Disconnect from the server"""
        if self.websocket:
            await self.websocket.close()
            self.connected = False
            logger.info("Disconnected from server")

# Legacy functions for backward compatibility
async def send_message(message: str, uri: str = "ws://localhost:8765", 
                      client_id: str = "default_client", secret_key: Optional[str] = None):
    """Legacy function to send a single message"""
    if secret_key is None:
        raise ValueError("Secret key is required for secure communication")
    
    client = SecureWebSocketClient(uri, client_id, secret_key)
    if await client.connect():
        await client.send_message(message)
        await client.disconnect()
    else:
        raise ConnectionError("Failed to connect to server")

def start_server(host: str = "0.0.0.0", port: int = 8765, 
                secret_key: Optional[str] = None, use_ssl: bool = False):
    """Legacy function to start server"""
    server = SecureWebSocketServer(host, port, secret_key, use_ssl)
    return server.start_server()
