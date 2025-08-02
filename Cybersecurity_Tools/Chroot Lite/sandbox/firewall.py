"""
Network isolation functionality for sandboxed processes.
"""

import os
import subprocess
import logging
from typing import Optional, List, Dict, Any

logger = logging.getLogger(__name__)


class NetworkFirewall:
    """Manages network isolation for sandboxed processes."""
    
    def __init__(self, block_network: bool = True):
        """
        Initialize network firewall.
        
        Args:
            block_network: Whether to block network access by default
        """
        self.block_network = block_network
        self.blocked_pids: List[int] = []
        
    def block_process_network(self, pid: int) -> bool:
        """
        Block network access for a specific process.
        
        Args:
            pid: Process ID to block network access for
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Try using iptables to block the process
            if self._block_with_iptables(pid):
                self.blocked_pids.append(pid)
                logger.info(f"Blocked network access for PID {pid}")
                return True
            
            # Fallback: try using firewall-cmd
            if self._block_with_firewalld(pid):
                self.blocked_pids.append(pid)
                logger.info(f"Blocked network access for PID {pid}")
                return True
                
            logger.warning(f"Could not block network access for PID {pid} - no supported firewall found")
            return False
            
        except Exception as e:
            logger.error(f"Failed to block network access for PID {pid}: {e}")
            return False
    
    def _block_with_iptables(self, pid: int) -> bool:
        """
        Block network access using iptables.
        
        Args:
            pid: Process ID to block
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Check if iptables is available
            result = subprocess.run(['which', 'iptables'], 
                                  capture_output=True, text=True, timeout=5)
            if result.returncode != 0:
                return False
            
            # Create a chain for this process
            chain_name = f"SANDBOX_{pid}"
            
            # Create the chain
            subprocess.run(['iptables', '-t', 'filter', '-N', chain_name], 
                         check=True, timeout=10)
            
            # Add rule to block all outgoing traffic from this process
            subprocess.run([
                'iptables', '-t', 'filter', '-A', 'OUTPUT', 
                '-m', 'owner', '--pid-owner', str(pid), 
                '-j', chain_name
            ], check=True, timeout=10)
            
            # Add rule to drop all traffic in the chain
            subprocess.run([
                'iptables', '-t', 'filter', '-A', chain_name, 
                '-j', 'DROP'
            ], check=True, timeout=10)
            
            return True
            
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError):
            return False
    
    def _block_with_firewalld(self, pid: int) -> bool:
        """
        Block network access using firewalld.
        
        Args:
            pid: Process ID to block
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Check if firewall-cmd is available
            result = subprocess.run(['which', 'firewall-cmd'], 
                                  capture_output=True, text=True, timeout=5)
            if result.returncode != 0:
                return False
            
            # Create a rich rule to block the process
            rule = f"rule pid user=\"{os.getenv('USER', 'root')}\" drop"
            
            subprocess.run([
                'firewall-cmd', '--add-rich-rule', rule,
                '--permanent'
            ], check=True, timeout=10)
            
            subprocess.run(['firewall-cmd', '--reload'], 
                         check=True, timeout=10)
            
            return True
            
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError):
            return False
    
    def unblock_process_network(self, pid: int) -> bool:
        """
        Unblock network access for a specific process.
        
        Args:
            pid: Process ID to unblock network access for
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if pid in self.blocked_pids:
                self.blocked_pids.remove(pid)
            
            # Try to remove iptables rules
            if self._unblock_with_iptables(pid):
                logger.info(f"Unblocked network access for PID {pid}")
                return True
            
            # Try to remove firewalld rules
            if self._unblock_with_firewalld(pid):
                logger.info(f"Unblocked network access for PID {pid}")
                return True
            
            logger.warning(f"Could not unblock network access for PID {pid}")
            return False
            
        except Exception as e:
            logger.error(f"Failed to unblock network access for PID {pid}: {e}")
            return False
    
    def _unblock_with_iptables(self, pid: int) -> bool:
        """
        Unblock network access using iptables.
        
        Args:
            pid: Process ID to unblock
            
        Returns:
            True if successful, False otherwise
        """
        try:
            chain_name = f"SANDBOX_{pid}"
            
            # Remove rules from OUTPUT chain
            subprocess.run([
                'iptables', '-t', 'filter', '-D', 'OUTPUT', 
                '-m', 'owner', '--pid-owner', str(pid), 
                '-j', chain_name
            ], check=False, timeout=10)
            
            # Flush and delete the chain
            subprocess.run(['iptables', '-t', 'filter', '-F', chain_name], 
                         check=False, timeout=10)
            subprocess.run(['iptables', '-t', 'filter', '-X', chain_name], 
                         check=False, timeout=10)
            
            return True
            
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
            return False
    
    def _unblock_with_firewalld(self, pid: int) -> bool:
        """
        Unblock network access using firewalld.
        
        Args:
            pid: Process ID to unblock
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Remove the rich rule
            rule = f"rule pid user=\"{os.getenv('USER', 'root')}\" drop"
            
            subprocess.run([
                'firewall-cmd', '--remove-rich-rule', rule,
                '--permanent'
            ], check=False, timeout=10)
            
            subprocess.run(['firewall-cmd', '--reload'], 
                         check=False, timeout=10)
            
            return True
            
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
            return False
    
    def get_network_status(self, pid: int) -> Dict[str, Any]:
        """
        Get network access status for a process.
        
        Args:
            pid: Process ID to check
            
        Returns:
            Dictionary with network status information
        """
        is_blocked = pid in self.blocked_pids
        
        return {
            'pid': pid,
            'blocked': is_blocked,
            'status': 'blocked' if is_blocked else 'allowed'
        }
    
    def cleanup(self) -> None:
        """Clean up all blocked processes."""
        for pid in self.blocked_pids.copy():
            self.unblock_process_network(pid)
        
        logger.info("Cleaned up all network blocks")
    
    def is_network_available(self) -> bool:
        """
        Check if network isolation is available on this system.
        
        Returns:
            True if network isolation is available, False otherwise
        """
        try:
            # Check for iptables
            result = subprocess.run(['which', 'iptables'], 
                                  capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                return True
            
            # Check for firewalld
            result = subprocess.run(['which', 'firewall-cmd'], 
                                  capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                return True
            
            return False
            
        except Exception:
            return False 