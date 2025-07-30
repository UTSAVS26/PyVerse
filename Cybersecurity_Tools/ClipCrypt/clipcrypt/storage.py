"""
Storage module for ClipCrypt.

Handles encrypted storage and retrieval of clipboard entries
using JSON format with metadata.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from .encryption import EncryptionManager


class StorageManager:
    """Manages encrypted storage of clipboard entries."""
    
    def __init__(self, config_dir: Path):
        """Initialize the storage manager.
        
        Args:
            config_dir: Directory to store encrypted data
        """
        self.config_dir = config_dir
        self.config_dir.mkdir(parents=True, exist_ok=True)
        self.data_file = config_dir / "clipboard_data.json"
        self.encryption_manager = EncryptionManager(config_dir)
        self._entries: List[Dict[str, Any]] = []
        self._load_entries()
    
    def _load_entries(self) -> None:
        """Load encrypted entries from storage."""
        if self.data_file.exists():
            try:
                with open(self.data_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self._entries = data.get('entries', [])
            except Exception as e:
                print(f"Warning: Could not load existing data: {e}")
                self._entries = []
        else:
            self._entries = []
    
    def _save_entries(self) -> None:
        """Save encrypted entries to storage."""
        try:
            data = {
                'version': '1.0',
                'created': datetime.now().isoformat(),
                'entries': self._entries
            }
            with open(self.data_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"Error saving data: {e}")
    
    def add_entry(self, content: str, tags: Optional[List[str]] = None, 
                  source: Optional[str] = None) -> int:
        """Add a new clipboard entry.
        
        Args:
            content: The clipboard content to store
            tags: Optional list of tags for categorization
            source: Optional source application name
            
        Returns:
            The ID of the newly created entry
        """
        # Encrypt the content
        encrypted_data = self.encryption_manager.encrypt(content)
        
        # Create entry with metadata
        entry = {
            'id': len(self._entries) + 1,
            'timestamp': datetime.now().isoformat(),
            'content': encrypted_data,
            'tags': tags or [],
            'source': source,
            'size': len(content)
        }
        
        self._entries.append(entry)
        self._save_entries()
        
        return entry['id']
    
    def get_entry(self, entry_id: int) -> Optional[Dict[str, Any]]:
        """Get a specific entry by ID.
        
        Args:
            entry_id: The ID of the entry to retrieve
            
        Returns:
            Entry dictionary with decrypted content, or None if not found
        """
        for entry in self._entries:
            if entry['id'] == entry_id:
                # Decrypt the content
                decrypted_content = self.encryption_manager.decrypt(
                    entry['content']['encrypted_data'],
                    entry['content']['nonce']
                )
                
                # Return entry with decrypted content
                result = entry.copy()
                result['content'] = decrypted_content
                return result
        
        return None
    
    def list_entries(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """List all entries with basic metadata.
        
        Args:
            limit: Maximum number of entries to return
            
        Returns:
            List of entry metadata (without decrypted content)
        """
        entries = self._entries.copy()
        if limit:
            entries = entries[-limit:]
        
        # Return entries without decrypted content for security
        return [
            {
                'id': entry['id'],
                'timestamp': entry['timestamp'],
                'tags': entry['tags'],
                'source': entry['source'],
                'size': entry['size']
            }
            for entry in entries
        ]
    
    def search_entries(self, query: str) -> List[Tuple[int, str]]:
        """Search entries by content.
        
        Args:
            query: Search query string
            
        Returns:
            List of tuples (entry_id, decrypted_content) for matching entries
        """
        results = []
        query_lower = query.lower()
        
        for entry in self._entries:
            try:
                # Decrypt content for search
                decrypted_content = self.encryption_manager.decrypt(
                    entry['content']['encrypted_data'],
                    entry['content']['nonce']
                )
                
                # Check if query matches content
                if query_lower in decrypted_content.lower():
                    results.append((entry['id'], decrypted_content))
            except Exception as e:
                print(f"Warning: Could not decrypt entry {entry['id']}: {e}")
                continue
        
        return results
    
    def delete_entry(self, entry_id: int) -> bool:
        """Delete an entry by ID.
        
        Args:
            entry_id: The ID of the entry to delete
            
        Returns:
            True if entry was deleted, False if not found
        """
        for i, entry in enumerate(self._entries):
            if entry['id'] == entry_id:
                del self._entries[i]
                self._save_entries()
                return True
        return False
    
    def add_tag(self, entry_id: int, tag: str) -> bool:
        """Add a tag to an entry.
        
        Args:
            entry_id: The ID of the entry
            tag: The tag to add
            
        Returns:
            True if tag was added, False if entry not found
        """
        for entry in self._entries:
            if entry['id'] == entry_id:
                if tag not in entry['tags']:
                    entry['tags'].append(tag)
                    self._save_entries()
                return True
        return False
    
    def remove_tag(self, entry_id: int, tag: str) -> bool:
        """Remove a tag from an entry.
        
        Args:
            entry_id: The ID of the entry
            tag: The tag to remove
            
        Returns:
            True if tag was removed, False if entry not found
        """
        for entry in self._entries:
            if entry['id'] == entry_id:
                if tag in entry['tags']:
                    entry['tags'].remove(tag)
                    self._save_entries()
                return True
        return False
    
    def get_entries_by_tag(self, tag: str) -> List[Dict[str, Any]]:
        """Get all entries with a specific tag.
        
        Args:
            tag: The tag to filter by
            
        Returns:
            List of entries with the specified tag
        """
        return [
            {
                'id': entry['id'],
                'timestamp': entry['timestamp'],
                'tags': entry['tags'],
                'source': entry['source'],
                'size': entry['size']
            }
            for entry in self._entries
            if tag in entry['tags']
        ]
    
    def clear_all(self) -> None:
        """Clear all entries."""
        self._entries = []
        self._save_entries()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get storage statistics.
        
        Returns:
            Dictionary with storage statistics
        """
        total_size = sum(entry['size'] for entry in self._entries)
        all_tags = set()
        for entry in self._entries:
            all_tags.update(entry['tags'])
        
        return {
            'total_entries': len(self._entries),
            'total_size_bytes': total_size,
            'unique_tags': len(all_tags),
            'tags': list(all_tags)
        } 