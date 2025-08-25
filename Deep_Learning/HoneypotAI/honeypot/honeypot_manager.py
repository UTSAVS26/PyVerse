"""
Honeypot Manager
Coordinates all honeypot services and provides unified management interface
"""

import threading
import time
import json
from typing import Dict, List, Any, Optional, Callable
from datetime import datetime
import structlog

from .ssh_server import SSHServer
from .http_server import HTTPServer
from .ftp_server import FTPServer

logger = structlog.get_logger(__name__)

class HoneypotManager:
    """Main honeypot manager that coordinates all services"""
    
    def __init__(self):
        self.servers = {}
        self.running = False
        self.logs = []
        self.callbacks = {}
        self.stats = {
            "total_connections": 0,
            "total_attacks": 0,
            "start_time": None,
            "uptime": 0
        }
        
        # Configure logging
        self.logger = structlog.get_logger("honeypot.manager")
        
        # Default service configurations
        self.default_config = {
            "ssh": {"port": 2222, "host": "0.0.0.0"},
            "http": {"port": 8080, "host": "0.0.0.0"},
            "ftp": {"port": 2121, "host": "0.0.0.0"}
        }
    
    def deploy_service(self, service_type: str, port: Optional[int] = None, host: str = "0.0.0.0") -> bool:
        """Deploy a honeypot service"""
        try:
            if service_type.lower() == "ssh":
                server = SSHServer(port or self.default_config["ssh"]["port"], host)
            elif service_type.lower() == "http":
                server = HTTPServer(port or self.default_config["http"]["port"], host)
            elif service_type.lower() == "ftp":
                server = FTPServer(port or self.default_config["ftp"]["port"], host)
            else:
                self.logger.error(f"Unknown service type: {service_type}")
                return False
            
            # Register connection callback
            server.register_callback("connection", self._on_connection)
            
            # Start the server
            if server.start():
                self.servers[service_type.lower()] = server
                self.logger.info(f"Deployed {service_type} honeypot on {host}:{server.port}")
                return True
            else:
                self.logger.error(f"Failed to start {service_type} server")
                return False
                
        except Exception as e:
            self.logger.error(f"Error deploying {service_type} service: {e}")
            return False
    
    def deploy_all_services(self) -> bool:
        """Deploy all default honeypot services"""
        success = True
        for service_type, config in self.default_config.items():
            if not self.deploy_service(service_type, config["port"], config["host"]):
                success = False
        
        if success:
            self.running = True
            self.stats["start_time"] = datetime.now()
            self.logger.info("All honeypot services deployed successfully")
        
        return success
    
    def stop_service(self, service_type: str) -> bool:
        """Stop a specific honeypot service"""
        service_type = service_type.lower()
        if service_type in self.servers:
            server = self.servers[service_type]
            if server.stop():
                del self.servers[service_type]
                self.logger.info(f"Stopped {service_type} honeypot")
                return True
            else:
                self.logger.error(f"Failed to stop {service_type} server")
                return False
        else:
            self.logger.warning(f"Service {service_type} not found")
            return False
    
    def stop_all_services(self) -> bool:
        """Stop all honeypot services"""
        success = True
        for service_type in list(self.servers.keys()):
            if not self.stop_service(service_type):
                success = False
        
        if success:
            self.running = False
            self.logger.info("All honeypot services stopped")
        
        return success
    
    def get_service_status(self, service_type: str) -> Dict[str, Any]:
        """Get status of a specific service"""
        service_type = service_type.lower()
        if service_type in self.servers:
            server = self.servers[service_type]
            status = server.get_stats()
            status["service_type"] = service_type
            status["running"] = server.is_running()
            
            # Add service-specific stats
            if service_type == "ssh":
                status.update(server.get_brute_force_stats())
            elif service_type == "http":
                status.update(server.get_attack_stats())
            elif service_type == "ftp":
                status.update(server.get_ftp_stats())
            
            return status
        else:
            return {"error": f"Service {service_type} not found"}
    
    def get_all_services_status(self) -> Dict[str, Any]:
        """Get status of all services"""
        status = {}
        for service_type in self.servers:
            status[service_type] = self.get_service_status(service_type)
        return status
    
    def get_service_logs(self, service_type: str, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Get logs from a specific service"""
        service_type = service_type.lower()
        if service_type in self.servers:
            return self.servers[service_type].get_logs(limit)
        else:
            return []
    
    def get_all_logs(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Get logs from all services"""
        all_logs = []
        for service_type in self.servers:
            service_logs = self.get_service_logs(service_type, limit)
            for log in service_logs:
                log["service_type"] = service_type
            all_logs.extend(service_logs)
        
        # Sort by timestamp
        all_logs.sort(key=lambda x: x.get("timestamp", ""))
        return all_logs
    
    def clear_logs(self, service_type: Optional[str] = None):
        """Clear logs from services"""
        if service_type:
            service_type = service_type.lower()
            if service_type in self.servers:
                self.servers[service_type].clear_logs()
                self.logger.info(f"Cleared logs for {service_type}")
        else:
            for server in self.servers.values():
                server.clear_logs()
            self.logger.info("Cleared logs for all services")
    
    def register_callback(self, event: str, callback: Callable):
        """Register a callback for specific events"""
        if event not in self.callbacks:
            self.callbacks[event] = []
        self.callbacks[event].append(callback)
    
    def _on_connection(self, connection_log):
        """Handle connection events from servers"""
        # Add to global logs
        self.logs.append(connection_log)
        
        # Update statistics
        self.stats["total_connections"] += 1
        if connection_log.attack_type:
            self.stats["total_attacks"] += 1
        
        # Trigger callbacks
        self._trigger_callbacks("connection", connection_log)
        
        # Log attack detection
        if connection_log.attack_type:
            self.logger.warning(
                f"Attack detected: {connection_log.attack_type} from {connection_log.source_ip} "
                f"on {connection_log.service} (confidence: {connection_log.confidence})"
            )
    
    def _trigger_callbacks(self, event: str, data: Any):
        """Trigger registered callbacks for an event"""
        if event in self.callbacks:
            for callback in self.callbacks[event]:
                try:
                    callback(data)
                except Exception as e:
                    self.logger.error(f"Error in callback: {e}")
    
    def get_overall_stats(self) -> Dict[str, Any]:
        """Get overall honeypot statistics"""
        stats = self.stats.copy()
        
        # Calculate uptime
        if stats["start_time"]:
            stats["uptime"] = (datetime.now() - stats["start_time"]).total_seconds()
        
        # Aggregate service stats
        stats["active_services"] = len([s for s in self.servers.values() if s.is_running()])
        stats["total_services"] = len(self.servers)
        
        # Attack statistics
        attack_counts = {}
        for log in self.logs:
            if log.attack_type:
                attack_counts[log.attack_type] = attack_counts.get(log.attack_type, 0) + 1
        
        stats["attack_breakdown"] = attack_counts
        stats["most_common_attack"] = max(attack_counts.items(), key=lambda x: x[1])[0] if attack_counts else None
        
        # Service-specific stats
        stats["service_stats"] = {}
        for service_type, server in self.servers.items():
            stats["service_stats"][service_type] = server.get_stats()
        
        return stats
    
    def export_logs(self, filename: str, format: str = "json"):
        """Export logs to file"""
        try:
            all_logs = self.get_all_logs()
            
            if format.lower() == "json":
                with open(filename, 'w') as f:
                    json.dump(all_logs, f, indent=2, default=str)
            elif format.lower() == "csv":
                import csv
                if all_logs:
                    with open(filename, 'w', newline='') as f:
                        writer = csv.DictWriter(f, fieldnames=all_logs[0].keys())
                        writer.writeheader()
                        writer.writerows(all_logs)
            else:
                self.logger.error(f"Unsupported export format: {format}")
                return False
            
            self.logger.info(f"Exported {len(all_logs)} logs to {filename}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error exporting logs: {e}")
            return False
    
    def is_running(self) -> bool:
        """Check if honeypot manager is running"""
        return self.running and any(server.is_running() for server in self.servers.values())
    
    def get_available_services(self) -> List[str]:
        """Get list of available service types"""
        return list(self.default_config.keys())
    
    def get_deployed_services(self) -> List[str]:
        """Get list of currently deployed services"""
        return list(self.servers.keys())
