#!/usr/bin/env python3
"""
Debug script for footnote removal
"""

import sys
import os

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from flashgenie.preprocessing.clean_text import TextCleaner

def debug_footnote_removal():
    """Debug the footnote removal functionality."""
    
    # Sample text from the test
    sample_text = """
    This is a sample document with multiple    spaces.
    
    It contains references and footnotes.
    
    1. This is a footnote.
    a) Another footnote.
    * Yet another footnote.
    
    References:
    Smith, J. (2020). Sample paper.
    Jones, A. (2021). Another paper.
    
    Bibliography:
    Brown, B. (2019). Third paper.
    """
    
    print("Original text:")
    print(sample_text)
    
    # Create cleaner
    cleaner = TextCleaner(remove_references=True, remove_footnotes=True)
    
    # Test footnote removal
    result = cleaner._remove_footnotes(sample_text)
    print("\nAfter footnote removal:")
    print(result)
    
    # Test full cleaning
    full_result = cleaner.clean_text(sample_text)
    print("\nAfter full cleaning:")
    print(full_result)
    
    # Check if specific footnotes are removed
    print(f"\n'1. This is a footnote.' in result: {'1. This is a footnote.' in full_result}")
    print(f"'a) Another footnote.' in result: {'a) Another footnote.' in full_result}")
    print(f"'* Yet another footnote.' in result: {'* Yet another footnote.' in full_result}")

if __name__ == "__main__":
    debug_footnote_removal()
