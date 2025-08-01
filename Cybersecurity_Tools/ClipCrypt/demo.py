#!/usr/bin/env python3
"""
Demo script for ClipCrypt.

This script demonstrates the basic functionality of ClipCrypt
without requiring clipboard access.
"""

import tempfile
from pathlib import Path
from clipcrypt import ClipCrypt


def main():
    """Run the ClipCrypt demo."""
    print("🔐 ClipCrypt Demo")
    print("=" * 50)
    
    # Create a temporary directory for demo
    temp_dir = Path(tempfile.mkdtemp())
    print(f"Using temporary directory: {temp_dir}")
    
    # Initialize ClipCrypt
    clipcrypt = ClipCrypt(temp_dir)
    
    # Demo 1: Add some test entries
    print("\n📝 Adding test entries...")
    clipcrypt.storage_manager.add_entry("Hello, ClipCrypt!", tags=["demo", "greeting"])
    clipcrypt.storage_manager.add_entry("This is a test entry for demonstration.", tags=["demo", "test"])
    clipcrypt.storage_manager.add_entry("Python is awesome!", tags=["demo", "code"])
    clipcrypt.storage_manager.add_entry("ClipCrypt provides secure clipboard management.", tags=["demo", "feature"])
    
    print("✅ Added 4 test entries")
    
    # Demo 2: List entries
    print("\n📋 Listing all entries:")
    clipcrypt.list_entries()
    
    # Demo 3: Search functionality
    print("\n🔍 Searching for 'test':")
    clipcrypt.search_entries("test")
    
    print("\n🔍 Searching for 'Python':")
    clipcrypt.search_entries("Python")
    
    # Demo 4: Get specific entry
    print("\n📄 Getting entry 1:")
    clipcrypt.get_entry(1)
    
    # Demo 5: Add tags
    print("\n🏷️  Adding tags:")
    clipcrypt.add_tag(1, "important")
    clipcrypt.add_tag(2, "important")
    
    # Demo 6: List entries by tag
    print("\n📋 Entries with 'demo' tag:")
    clipcrypt.get_entries_by_tag("demo")
    
    print("\n📋 Entries with 'important' tag:")
    clipcrypt.get_entries_by_tag("important")
    
    # Demo 7: Show statistics
    print("\n📊 Storage statistics:")
    clipcrypt.show_stats()
    
    # Demo 8: Copy to clipboard (simulated)
    print("\n📋 Copying entry 1 to clipboard (simulated):")
    clipcrypt.copy_to_clipboard(1)
    
    # Demo 9: Delete an entry
    print("\n🗑️  Deleting entry 3:")
    clipcrypt.delete_entry(3)
    
    print("\n📋 Listing entries after deletion:")
    clipcrypt.list_entries()
    
    # Demo 10: Show final statistics
    print("\n📊 Final statistics:")
    clipcrypt.show_stats()
    
    print("\n🎉 Demo completed!")
    print(f"Data stored in: {temp_dir}")
    print("You can inspect the encrypted data files in the temporary directory.")


if __name__ == "__main__":
    main() 