"""
Clipboard monitoring module for ClipCrypt.

Handles clipboard monitoring and automatic storage of new entries.
"""

import time
import threading
from typing import Optional, Callable
import pyperclip
from .storage import StorageManager


class ClipboardMonitor:
    """Monitors clipboard for changes and automatically stores new entries."""
    
    def __init__(self, storage_manager: StorageManager, 
                 callback: Optional[Callable[[str, int], None]] = None):
        """Initialize the clipboard monitor.
        
        Args:
            storage_manager: Storage manager instance
            callback: Optional callback function called when new entry is added
        """
        self.storage_manager = storage_manager
        self.callback = callback
        self._monitoring = False
        self._monitor_thread: Optional[threading.Thread] = None
        self._last_content = ""
        self._check_interval = 1.0  # seconds
        
    def start_monitoring(self) -> None:
        """Start monitoring the clipboard for changes."""
        if self._monitoring:
            return
        
        self._monitoring = True
        self._monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._monitor_thread.start()
        print("🕵️  Watching clipboard...")
    
    def stop_monitoring(self) -> None:
        """Stop monitoring the clipboard."""
        self._monitoring = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=2.0)
        print("🛑 Clipboard monitoring stopped.")
    
    def _monitor_loop(self) -> None:
        """Main monitoring loop."""
        while self._monitoring:
            try:
                # Get current clipboard content
                current_content = pyperclip.paste()
                
                # Check if content has changed and is not empty
                if (current_content != self._last_content and 
                    current_content.strip() and 
                    len(current_content) > 1):  # Ignore single characters
                    
                    # Store the new content
                    entry_id = self.storage_manager.add_entry(current_content)
                    self._last_content = current_content
                    
                    # Print status
                    print(f"🕵️  New item detected ({len(current_content)} chars)")
                    print("🔐 Saved (Encrypted ✔)")
                    
                    # Call callback if provided
                    if self.callback:
                        self.callback(current_content, entry_id)
                
                time.sleep(self._check_interval)
                
            except Exception as e:
                print(f"Error monitoring clipboard: {e}")
                time.sleep(self._check_interval)
    
    def is_monitoring(self) -> bool:
        """Check if monitoring is active.
        
        Returns:
            True if monitoring is active
        """
        return self._monitoring
    
    def set_check_interval(self, seconds: float) -> None:
        """Set the check interval for clipboard monitoring.
        
        Args:
            seconds: Interval in seconds between clipboard checks
        """
        if seconds > 0:
            self._check_interval = seconds
    
    def get_current_content(self) -> str:
        """Get the current clipboard content.
        
        Returns:
            Current clipboard content
        """
        try:
            return pyperclip.paste()
        except Exception as e:
            print(f"Error getting clipboard content: {e}")
            return ""
    
    def set_content(self, content: str) -> bool:
        """Set clipboard content.
        
        Args:
            content: Content to set in clipboard
            
        Returns:
            True if successful
        """
        try:
            pyperclip.copy(content)
            return True
        except Exception as e:
            print(f"Error setting clipboard content: {e}")
            return False 