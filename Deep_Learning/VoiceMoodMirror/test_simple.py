#!/usr/bin/env python3
"""
Simple test script to verify basic functionality
"""

import sys
import os

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_imports():
    """Test if we can import our modules."""
    try:
        from flashgenie.pdf_parser.extract_text import PDFTextExtractor
        print("✓ PDFTextExtractor imported successfully")
        
        from flashgenie.preprocessing.clean_text import TextCleaner
        print("✓ TextCleaner imported successfully")
        
        from flashgenie.preprocessing.chunker import TextChunker
        print("✓ TextChunker imported successfully")
        
        return True
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False

def test_basic_functionality():
    """Test basic functionality without external dependencies."""
    try:
        from flashgenie.preprocessing.clean_text import TextCleaner
        from flashgenie.preprocessing.chunker import TextChunker
        
        # Test text cleaning
        cleaner = TextCleaner()
        test_text = "  This   is   a   test   text.  "
        cleaned = cleaner.clean_text(test_text)
        print(f"✓ Text cleaning works: '{cleaned}'")
        
        # Test text chunking
        chunker = TextChunker()
        chunks = chunker.chunk_by_sentences(test_text)
        print(f"✓ Text chunking works: {len(chunks)} chunks")
        
        return True
    except Exception as e:
        print(f"✗ Basic functionality error: {e}")
        return False

if __name__ == "__main__":
    print("Testing FlashGenie basic functionality...")
    print("=" * 50)
    
    success = True
    success &= test_imports()
    success &= test_basic_functionality()
    
    print("=" * 50)
    if success:
        print("✓ All basic tests passed!")
    else:
        print("✗ Some tests failed!")
    
    sys.exit(0 if success else 1)
