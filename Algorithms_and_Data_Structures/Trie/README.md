# Trie (Prefix Tree) Data Structure

A comprehensive implementation of the Trie data structure in Python, designed for efficient string storage, search, and prefix operations.

## Overview

A **Trie** (pronounced "try") is a tree-like data structure that stores a dynamic set of strings where the keys are usually strings. Each node in the tree represents a single character, and paths from the root to leaf nodes represent complete words. Tries are particularly useful for applications involving string prefix matching, such as autocomplete systems, spell checkers, and dictionary implementations.

## Features

### Core Operations
- **Insert**: Add words to the trie in O(L) time
- **Search**: Find complete words in O(L) time  
- **Prefix Matching**: Check if any word starts with a given prefix
- **Delete**: Remove words while maintaining trie structure
- **Autocomplete**: Get all words starting with a specific prefix

### Advanced Features
- **Case-insensitive operations**: Automatic lowercase conversion
- **Word counting**: Track total words and frequency
- **Longest Common Prefix**: Find the longest prefix shared by all words
- **Memory efficient deletion**: Clean up unused nodes automatically
- **Comprehensive error handling**: Robust input validation

## Time and Space Complexity

| Operation | Time Complexity | Space Complexity |
|-----------|-----------------|------------------|
| Insert | O(L) | O(L) |
| Search | O(L) | O(1) |
| Delete | O(L) | O(1) |
| Prefix Check | O(P) | O(1) |
| Get All Words with Prefix | O(P + N) | O(N) |

Where:
- L = length of the word
- P = length of the prefix  
- N = number of nodes in the subtree

**Overall Space Complexity**: O(ALPHABET_SIZE × N × L) where N is the number of words and L is the average word length.

## Installation

Simply copy the `trie.py` file to your project directory and import the Trie class:

