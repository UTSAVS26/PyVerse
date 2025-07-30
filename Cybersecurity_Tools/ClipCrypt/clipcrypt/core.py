"""
Core ClipCrypt module.

Main interface for the ClipCrypt clipboard manager.
"""

import os
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from datetime import datetime

from .storage import StorageManager
from .clipboard import ClipboardMonitor


class ClipCrypt:
    """Main ClipCrypt application class."""
    
    def __init__(self, config_dir: Optional[Path] = None):
        """Initialize ClipCrypt.
        
        Args:
            config_dir: Optional custom configuration directory
        """
        if config_dir is None:
            config_dir = self._get_default_config_dir()
        
        self.config_dir = config_dir
        self.console = Console()
        self.storage_manager = StorageManager(config_dir)
        self.clipboard_monitor = ClipboardMonitor(self.storage_manager)
    
    def _get_default_config_dir(self) -> Path:
        """Get the default configuration directory based on OS."""
        if sys.platform == "win32":
            base_dir = Path(os.environ.get("APPDATA", ""))
        elif sys.platform == "darwin":
            base_dir = Path.home() / "Library" / "Application Support"
        else:
            base_dir = Path.home() / ".config"
        
        return base_dir / "ClipCrypt"
    
    def start_monitoring(self) -> None:
        """Start clipboard monitoring."""
        self.clipboard_monitor.start_monitoring()
        
        try:
            # Keep the monitoring running
            while self.clipboard_monitor.is_monitoring():
                import time
                time.sleep(1)
        except KeyboardInterrupt:
            self.clipboard_monitor.stop_monitoring()
    
    def list_entries(self, limit: Optional[int] = None) -> None:
        """List all clipboard entries.
        
        Args:
            limit: Maximum number of entries to show
        """
        entries = self.storage_manager.list_entries(limit)
        
        if not entries:
            self.console.print("No clipboard entries found.")
            return
        
        table = Table(title="Clipboard History")
        table.add_column("ID", style="cyan", no_wrap=True)
        table.add_column("Timestamp", style="green")
        table.add_column("Size", style="yellow")
        table.add_column("Tags", style="magenta")
        table.add_column("Source", style="blue")
        
        for entry in entries:
            # Format timestamp
            try:
                dt = datetime.fromisoformat(entry['timestamp'])
                timestamp = dt.strftime("%Y-%m-%d %H:%M")
            except:
                timestamp = entry['timestamp']
            
            # Format tags
            tags = ", ".join(entry['tags']) if entry['tags'] else "-"
            
            # Format source
            source = entry['source'] or "-"
            
            table.add_row(
                str(entry['id']),
                timestamp,
                f"{entry['size']} chars",
                tags,
                source
            )
        
        self.console.print(table)
    
    def get_entry(self, entry_id: int) -> None:
        """Get and display a specific entry.
        
        Args:
            entry_id: The ID of the entry to retrieve
        """
        entry = self.storage_manager.get_entry(entry_id)
        
        if not entry:
            self.console.print(f"[red]Entry {entry_id} not found.[/red]")
            return
        
        # Format timestamp
        try:
            dt = datetime.fromisoformat(entry['timestamp'])
            timestamp = dt.strftime("%Y-%m-%d %H:%M:%S")
        except:
            timestamp = entry['timestamp']
        
        # Create content panel
        content_text = Text(entry['content'])
        content_panel = Panel(
            content_text,
            title=f"Entry {entry_id}",
            border_style="green"
        )
        
        # Create metadata panel
        metadata = f"""
ID: {entry['id']}
Timestamp: {timestamp}
Size: {entry['size']} characters
Tags: {', '.join(entry['tags']) if entry['tags'] else 'None'}
Source: {entry['source'] or 'Unknown'}
        """.strip()
        
        metadata_panel = Panel(
            metadata,
            title="Metadata",
            border_style="blue"
        )
        
        self.console.print(content_panel)
        self.console.print(metadata_panel)
    
    def search_entries(self, query: str) -> None:
        """Search entries by content.
        
        Args:
            query: Search query string
        """
        results = self.storage_manager.search_entries(query)
        
        if not results:
            self.console.print(f"No entries found matching '{query}'.")
            return
        
        self.console.print(f"[green]Found {len(results)} results:[/green]")
        
        for i, (entry_id, content) in enumerate(results, 1):
            # Truncate content for display
            display_content = content[:100] + "..." if len(content) > 100 else content
            
            panel = Panel(
                display_content,
                title=f"{i}. Entry {entry_id}",
                border_style="yellow"
            )
            self.console.print(panel)
    
    def delete_entry(self, entry_id: int) -> None:
        """Delete an entry.
        
        Args:
            entry_id: The ID of the entry to delete
        """
        if self.storage_manager.delete_entry(entry_id):
            self.console.print(f"[green]Entry {entry_id} deleted successfully.[/green]")
        else:
            self.console.print(f"[red]Entry {entry_id} not found.[/red]")
    
    def add_tag(self, entry_id: int, tag: str) -> None:
        """Add a tag to an entry.
        
        Args:
            entry_id: The ID of the entry
            tag: The tag to add
        """
        if self.storage_manager.add_tag(entry_id, tag):
            self.console.print(f"[green]Tag '{tag}' added to entry {entry_id}.[/green]")
        else:
            self.console.print(f"[red]Entry {entry_id} not found.[/red]")
    
    def remove_tag(self, entry_id: int, tag: str) -> None:
        """Remove a tag from an entry.
        
        Args:
            entry_id: The ID of the entry
            tag: The tag to remove
        """
        if self.storage_manager.remove_tag(entry_id, tag):
            self.console.print(f"[green]Tag '{tag}' removed from entry {entry_id}.[/green]")
        else:
            self.console.print(f"[red]Entry {entry_id} not found or tag not present.[/red]")
    
    def get_entries_by_tag(self, tag: str) -> None:
        """Get all entries with a specific tag.
        
        Args:
            tag: The tag to filter by
        """
        entries = self.storage_manager.get_entries_by_tag(tag)
        
        if not entries:
            self.console.print(f"No entries found with tag '{tag}'.")
            return
        
        self.console.print(f"[green]Found {len(entries)} entries with tag '{tag}':[/green]")
        
        table = Table(title=f"Entries with tag '{tag}'")
        table.add_column("ID", style="cyan", no_wrap=True)
        table.add_column("Timestamp", style="green")
        table.add_column("Size", style="yellow")
        table.add_column("Tags", style="magenta")
        
        for entry in entries:
            try:
                dt = datetime.fromisoformat(entry['timestamp'])
                timestamp = dt.strftime("%Y-%m-%d %H:%M")
            except:
                timestamp = entry['timestamp']
            
            tags = ", ".join(entry['tags']) if entry['tags'] else "-"
            
            table.add_row(
                str(entry['id']),
                timestamp,
                f"{entry['size']} chars",
                tags
            )
        
        self.console.print(table)
    
    def clear_all(self) -> None:
        """Clear all entries."""
        self.console.print("[yellow]Are you sure you want to delete all entries? (y/N):[/yellow]")
        response = input().strip().lower()
        
        if response in ['y', 'yes']:
            self.storage_manager.clear_all()
            self.console.print("[green]All entries cleared.[/green]")
        else:
            self.console.print("[blue]Operation cancelled.[/blue]")
    
    def show_stats(self) -> None:
        """Show storage statistics."""
        stats = self.storage_manager.get_stats()
        
        stats_text = f"""
Total Entries: {stats['total_entries']}
Total Size: {stats['total_size_bytes']} bytes
Unique Tags: {stats['unique_tags']}
Tags: {', '.join(stats['tags']) if stats['tags'] else 'None'}
        """.strip()
        
        panel = Panel(
            stats_text,
            title="Storage Statistics",
            border_style="cyan"
        )
        self.console.print(panel)
    
    def copy_to_clipboard(self, entry_id: int) -> None:
        """Copy an entry back to the clipboard.
        
        Args:
            entry_id: The ID of the entry to copy
        """
        entry = self.storage_manager.get_entry(entry_id)
        
        if not entry:
            self.console.print(f"[red]Entry {entry_id} not found.[/red]")
            return
        
        if self.clipboard_monitor.set_content(entry['content']):
            self.console.print(f"[green]Entry {entry_id} copied to clipboard.[/green]")
        else:
            self.console.print(f"[red]Failed to copy entry {entry_id} to clipboard.[/red]") 