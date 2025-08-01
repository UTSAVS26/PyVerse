"""
Tests for the storage module.
"""

import pytest
import tempfile
import shutil
import json
from pathlib import Path
from datetime import datetime
from clipcrypt.storage import StorageManager


class TestStorageManager:
    """Test cases for StorageManager."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for testing."""
        temp_dir = Path(tempfile.mkdtemp())
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def storage_manager(self, temp_dir):
        """Create a StorageManager instance for testing."""
        return StorageManager(temp_dir)
    
    def test_initialization(self, storage_manager, temp_dir):
        """Test StorageManager initialization."""
        assert storage_manager.config_dir == temp_dir
        assert storage_manager.data_file == temp_dir / "clipboard_data.json"
        assert storage_manager.encryption_manager is not None
    
    def test_add_entry(self, storage_manager):
        """Test adding a new entry."""
        content = "Test clipboard content"
        entry_id = storage_manager.add_entry(content)
        
        assert entry_id == 1
        assert len(storage_manager._entries) == 1
        
        entry = storage_manager._entries[0]
        assert entry['id'] == 1
        assert entry['size'] == len(content)
        assert entry['tags'] == []
        assert entry['source'] is None
        assert 'timestamp' in entry
        assert storage_manager.encryption_manager.is_encrypted(entry['content'])
    
    def test_add_entry_with_tags_and_source(self, storage_manager):
        """Test adding an entry with tags and source."""
        content = "Test content with tags"
        tags = ["test", "code"]
        source = "test_app"
        
        entry_id = storage_manager.add_entry(content, tags, source)
        
        assert entry_id == 1
        entry = storage_manager._entries[0]
        assert entry['tags'] == tags
        assert entry['source'] == source
    
    def test_get_entry(self, storage_manager):
        """Test retrieving an entry."""
        content = "Test content for retrieval"
        entry_id = storage_manager.add_entry(content)
        
        retrieved = storage_manager.get_entry(entry_id)
        
        assert retrieved is not None
        assert retrieved['id'] == entry_id
        assert retrieved['content'] == content
        assert retrieved['size'] == len(content)
    
    def test_get_nonexistent_entry(self, storage_manager):
        """Test retrieving a non-existent entry."""
        retrieved = storage_manager.get_entry(999)
        assert retrieved is None
    
    def test_list_entries(self, storage_manager):
        """Test listing entries."""
        # Add multiple entries
        storage_manager.add_entry("First entry")
        storage_manager.add_entry("Second entry")
        storage_manager.add_entry("Third entry")
        
        entries = storage_manager.list_entries()
        
        assert len(entries) == 3
        assert entries[0]['id'] == 1
        assert entries[1]['id'] == 2
        assert entries[2]['id'] == 3
        
        # Check that content is not included in list
        for entry in entries:
            assert 'content' not in entry
    
    def test_list_entries_with_limit(self, storage_manager):
        """Test listing entries with a limit."""
        # Add multiple entries
        storage_manager.add_entry("First entry")
        storage_manager.add_entry("Second entry")
        storage_manager.add_entry("Third entry")
        
        entries = storage_manager.list_entries(limit=2)
        
        assert len(entries) == 2
        assert entries[0]['id'] == 2  # Last 2 entries
        assert entries[1]['id'] == 3
    
    def test_search_entries(self, storage_manager):
        """Test searching entries."""
        # Add entries with different content
        storage_manager.add_entry("Hello world")
        storage_manager.add_entry("Python programming")
        storage_manager.add_entry("JavaScript code")
        
        # Search for "world"
        results = storage_manager.search_entries("world")
        assert len(results) == 1
        assert results[0][0] == 1  # entry_id
        assert "world" in results[0][1].lower()  # content
        
        # Search for "code"
        results = storage_manager.search_entries("code")
        assert len(results) == 1
        assert any("JavaScript" in result[1] for result in results)
    
    def test_search_case_insensitive(self, storage_manager):
        """Test that search is case insensitive."""
        storage_manager.add_entry("Hello WORLD")
        storage_manager.add_entry("hello world")
        
        results = storage_manager.search_entries("world")
        assert len(results) == 2
        
        results = storage_manager.search_entries("WORLD")
        assert len(results) == 2
    
    def test_delete_entry(self, storage_manager):
        """Test deleting an entry."""
        storage_manager.add_entry("First entry")
        storage_manager.add_entry("Second entry")
        
        # Delete first entry
        assert storage_manager.delete_entry(1) is True
        assert len(storage_manager._entries) == 1
        assert storage_manager._entries[0]['id'] == 2
        
        # Try to delete non-existent entry
        assert storage_manager.delete_entry(999) is False
    
    def test_add_tag(self, storage_manager):
        """Test adding a tag to an entry."""
        storage_manager.add_entry("Test content")
        
        assert storage_manager.add_tag(1, "test") is True
        assert storage_manager.add_tag(1, "code") is True
        
        entry = storage_manager.get_entry(1)
        assert "test" in entry['tags']
        assert "code" in entry['tags']
        
        # Try to add tag to non-existent entry
        assert storage_manager.add_tag(999, "test") is False
    
    def test_remove_tag(self, storage_manager):
        """Test removing a tag from an entry."""
        storage_manager.add_entry("Test content", tags=["test", "code"])
        
        assert storage_manager.remove_tag(1, "test") is True
        assert storage_manager.remove_tag(1, "code") is True
        
        entry = storage_manager.get_entry(1)
        assert entry['tags'] == []
        
        # Try to remove tag from non-existent entry
        assert storage_manager.remove_tag(999, "test") is False
    
    def test_get_entries_by_tag(self, storage_manager):
        """Test getting entries by tag."""
        storage_manager.add_entry("First entry", tags=["code"])
        storage_manager.add_entry("Second entry", tags=["test"])
        storage_manager.add_entry("Third entry", tags=["code", "test"])
        
        code_entries = storage_manager.get_entries_by_tag("code")
        assert len(code_entries) == 2
        assert code_entries[0]['id'] == 1
        assert code_entries[1]['id'] == 3
        
        test_entries = storage_manager.get_entries_by_tag("test")
        assert len(test_entries) == 2
        assert test_entries[0]['id'] == 2
        assert test_entries[1]['id'] == 3
    
    def test_clear_all(self, storage_manager):
        """Test clearing all entries."""
        storage_manager.add_entry("First entry")
        storage_manager.add_entry("Second entry")
        
        assert len(storage_manager._entries) == 2
        
        storage_manager.clear_all()
        
        assert len(storage_manager._entries) == 0
        assert storage_manager.list_entries() == []
    
    def test_get_stats(self, storage_manager):
        """Test getting storage statistics."""
        storage_manager.add_entry("First entry", tags=["code"])
        storage_manager.add_entry("Second entry", tags=["test"])
        storage_manager.add_entry("Third entry", tags=["code", "test"])
        
        stats = storage_manager.get_stats()
        
        assert stats['total_entries'] == 3
        assert stats['total_size_bytes'] == len("First entry") + len("Second entry") + len("Third entry")
        assert stats['unique_tags'] == 2
        assert "code" in stats['tags']
        assert "test" in stats['tags']
    
    def test_data_persistence(self, temp_dir):
        """Test that data persists between StorageManager instances."""
        # Create first manager and add data
        manager1 = StorageManager(temp_dir)
        manager1.add_entry("Persistent data", tags=["test"])
        
        # Create second manager and check data
        manager2 = StorageManager(temp_dir)
        entries = manager2.list_entries()
        
        assert len(entries) == 1
        assert entries[0]['id'] == 1
        assert "test" in entries[0]['tags']
        
        # Check that data can be retrieved
        entry = manager2.get_entry(1)
        assert entry['content'] == "Persistent data"
    
    def test_corrupted_data_handling(self, temp_dir):
        """Test handling of corrupted data files."""
        # Create a corrupted data file
        data_file = temp_dir / "clipboard_data.json"
        with open(data_file, 'w') as f:
            f.write("invalid json content")
        
        # Should handle corruption gracefully
        manager = StorageManager(temp_dir)
        assert len(manager._entries) == 0 