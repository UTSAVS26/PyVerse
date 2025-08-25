"""
Base Server Class for Honeypot Services
Provides common functionality for all honeypot servers
"""

import socket
import threading
import time
import logging
import json
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Callable
from dataclasses import dataclass, asdict
from datetime import datetime
import structlog

logger = structlog.get_logger(__name__)

@dataclass
class ConnectionLog:
    """Data class for logging connection attempts"""
    timestamp: str
    source_ip: str
    source_port: int
    service: str
    payload_size: int
    payload_hash: str
    connection_duration: float
    success: bool
    attack_type: Optional[str] = None
    confidence: float = 0.0

class BaseServer(ABC):
    """Abstract base class for all honeypot servers"""
    
    def __init__(self, service_name: str, port: int, host: str = "0.0.0.0"):
        self.service_name = service_name
        self.port = port
        self.host = host
        self.socket = None
        self.running = False
        self.thread = None
        self.connections = []
        self.logs = []
        self.callbacks = {}
        
        # Configure logging
        self.logger = structlog.get_logger(f"honeypot.{service_name}")
        
        # Statistics
        self.stats = {
            "total_connections": 0,
            "successful_connections": 0,
            "failed_connections": 0,
            "attack_detections": 0,
            "start_time": None,
            "uptime": 0
        }
    
    def start(self) -> bool:
        """Start the honeypot server"""
        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.socket.bind((self.host, self.port))
            self.socket.listen(5)
            
            self.running = True
            self.stats["start_time"] = datetime.now()
            self.thread = threading.Thread(target=self._run_server, daemon=True)
            self.thread.start()
            
            self.logger.info(f"Started {self.service_name} honeypot on {self.host}:{self.port}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to start {self.service_name} server: {e}")
            return False
    
    def stop(self) -> bool:
        """Stop the honeypot server"""
        try:
            self.running = False
            if self.socket:
                self.socket.close()
            if self.thread:
                self.thread.join(timeout=5)
            
            self.logger.info(f"Stopped {self.service_name} honeypot")
            return True
            
        except Exception as e:
            self.logger.error(f"Error stopping {self.service_name} server: {e}")
            return False
    
    def _run_server(self):
        """Main server loop"""
        while self.running:
            try:
                client_socket, address = self.socket.accept()
                self.stats["total_connections"] += 1
                
                # Create connection handler thread
                conn_thread = threading.Thread(
                    target=self._handle_connection,
                    args=(client_socket, address),
                    daemon=True
                )
                conn_thread.start()
                self.connections.append(conn_thread)
                
            except Exception as e:
                if self.running:
                    self.logger.error(f"Error accepting connection: {e}")
    
    def _handle_connection(self, client_socket: socket.socket, address: tuple):
        """Handle individual client connection"""
        start_time = time.time()
        source_ip, source_port = address
        
        try:
            # Log connection attempt
            connection_log = ConnectionLog(
                timestamp=datetime.now().isoformat(),
                source_ip=source_ip,
                source_port=source_port,
                service=self.service_name,
                payload_size=0,
                payload_hash="",
                connection_duration=0,
                success=False
            )
            
            # Handle the connection (implemented by subclasses)
            success = self._process_connection(client_socket, address, connection_log)
            
            # Update log
            connection_log.connection_duration = time.time() - start_time
            connection_log.success = success
            
            if success:
                self.stats["successful_connections"] += 1
            else:
                self.stats["failed_connections"] += 1
            
            # Store log
            self.logs.append(connection_log)
            
            # Trigger callbacks
            self._trigger_callbacks("connection", connection_log)
            
        except Exception as e:
            self.logger.error(f"Error handling connection from {source_ip}:{source_port}: {e}")
        finally:
            try:
                client_socket.close()
            except:
                pass
    
    @abstractmethod
    def _process_connection(self, client_socket: socket.socket, address: tuple, log: ConnectionLog) -> bool:
        """Process the connection - must be implemented by subclasses"""
        pass
    
    def register_callback(self, event: str, callback: Callable):
        """Register a callback for specific events"""
        if event not in self.callbacks:
            self.callbacks[event] = []
        self.callbacks[event].append(callback)
    
    def _trigger_callbacks(self, event: str, data: Any):
        """Trigger registered callbacks for an event"""
        if event in self.callbacks:
            for callback in self.callbacks[event]:
                try:
                    callback(data)
                except Exception as e:
                    self.logger.error(f"Error in callback: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get current server statistics"""
        stats = self.stats.copy()
     def get_stats(self) -> Dict[str, Any]:
         """Get current server statistics"""
         stats = self.stats.copy()
-        if stats["start_time"]:
        # Ensure start_time is ISO-serialized for transport, but still compute uptime if it's a real datetime
        if self.stats["start_time"]:
            # Convert datetime to ISO string (or leave it unchanged if already serialized)
            stats["start_time"] = (
                self.stats["start_time"].isoformat()
                if isinstance(self.stats["start_time"], datetime)
                else self.stats["start_time"]
            )
            # Compute uptime only when start_time is a datetime; otherwise default to 0
            stats["uptime"] = (
                (datetime.now() - self.stats["start_time"]).total_seconds()
                if isinstance(self.stats["start_time"], datetime)
                else 0
            )
         stats["active_connections"] = len([c for c in self.connections if c.is_alive()])
         stats["total_logs"] = len(self.logs)
        return stats
    
    def get_logs(self, limit: Optional[int] = None) -> list:
        """Get connection logs"""
        logs = [asdict(log) for log in self.logs]
        if limit:
            logs = logs[-limit:]
        return logs
    
    def clear_logs(self):
        """Clear stored logs"""
        self.logs.clear()
        self.logger.info("Cleared connection logs")
    
    def is_running(self) -> bool:
        """Check if server is running"""
        return self.running
