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
from trie import Trie

## Quick Start

from trie import Trie

Create a new Trie
trie = Trie()

Insert words
trie.insert("apple")
trie.insert("app")
trie.insert("application")

Search for words
print(trie.search("app")) # True
print(trie.search("appl")) # False (not a complete word)

Check prefixes
print(trie.starts_with("app")) # True
print(trie.starts_with("xyz")) # False

Get autocomplete suggestions
suggestions = trie.get_all_words_with_prefix("app")
print(suggestions) # ['app', 'apple', 'application']

Delete words
trie.delete("app")
print(trie.search("app")) # False


## Detailed Usage Examples

### Basic Operations

trie = Trie()

Insert words
words = ["cat", "car", "card", "care", "careful"]
for word in words:
trie.insert(word)

Check trie contents
print(f"Total words: {len(trie)}") # 5
print(f"All words: {trie.get_all_words()}") # ['car', 'card', 'care', 'careful', 'cat']

Search operations
print("car" in trie) # True (using contains)
print(trie.search("car")) # True
print(trie.search("ca")) # False


### Autocomplete System

trie = Trie()

Build vocabulary
programming_terms = [
"python", "programming", "program", "programmer",
"java", "javascript", "algorithm", "array"
]

for term in programming_terms:
trie.insert(term)

Autocomplete function
def autocomplete(prefix, max_suggestions=5):
suggestions = trie.get_all_words_with_prefix(prefix)
return suggestions[:max_suggestions]

Usage
print(autocomplete("prog")) # ['program', 'programmer', 'programming']
print(autocomplete("java")) # ['java', 'javascript']

### Spell Checker

trie = Trie()

Build dictionary
dictionary = ["hello", "world", "python", "programming", "computer"]
for word in dictionary:
trie.insert(word)

def spell_check(word):
if trie.search(word):
return f"✓ '{word}' is spelled correctly"
else:
# Simple suggestion: words with same prefix
suggestions = trie.get_all_words_with_prefix(word[:3])
if suggestions:
return f"✗ '{word}' not found. Suggestions: {suggestions[:3]}"
return f"✗ '{word}' not found. No suggestions available."

Test spell checking
print(spell_check("python")) # ✓ 'python' is spelled correctly
print(spell_check("pythom")) # ✗ 'pythom' not found. Suggestions: ['python']


### Advanced Features

trie = Trie()
words = ["test", "testing", "tester", "tea", "teach", "teacher"]

for word in words:
trie.insert(word)

Advanced operations
print(f"Longest common prefix: '{trie.longest_common_prefix()}'") # 'te'
print(f"Words with 'test' prefix: {trie.count_words_with_prefix('test')}") # 3
print(f"Words with 'tea' prefix: {trie.count_words_with_prefix('tea')}") # 3

Deletion with cleanup
print(f"Before deletion: {trie.get_all_words()}")
trie.delete("test")
print(f"After deleting 'test': {trie.get_all_words()}")

## Real-World Applications

### 1. Search Engine Autocomplete
class SearchAutocomplete:
def init(self):
self.trie = Trie()

def add_search_term(self, term):
    self.trie.insert(term)

def get_suggestions(self, partial_query, limit=10):
    return self.trie.get_all_words_with_prefix(partial_query)[:limit]

Usage
search = SearchAutocomplete()
search.add_search_term("machine learning")
search.add_search_term("machine translation")
search.add_search_term("machine vision")

print(search.get_suggestions("machine")) # All machine-related terms

### 2. IP Routing Table
class IPRouter:
def init(self):
self.trie = Trie()

def add_route(self, ip_prefix, next_hop):
    # Store route info in trie (simplified example)
    self.trie.insert(ip_prefix)

def longest_prefix_match(self, ip_address):
    # Find longest matching prefix for routing
    for i in range(len(ip_address), 0, -1):
        prefix = ip_address[:i]
        if self.trie.starts_with(prefix):
            return prefix
    return None

### 3. Dictionary with Prefix Operations
class SmartDictionary:
def init(self):
self.trie = Trie()

def add_word(self, word):
    self.trie.insert(word)

def is_word(self, word):
    return self.trie.search(word)

def words_starting_with(self, prefix):
    return self.trie.get_all_words_with_prefix(prefix)

def word_count(self):
    return len(self.trie)

## API Reference

### Trie Class

#### `__init__()`
Creates an empty Trie.

#### `insert(word: str) -> bool`
Inserts a word into the Trie. Returns `True` if successful, `False` for invalid input.

#### `search(word: str) -> bool`
Searches for a complete word. Returns `True` if word exists, `False` otherwise.

#### `starts_with(prefix: str) -> bool`
Checks if any word starts with the given prefix. Returns `True` if prefix exists.

#### `delete(word: str) -> bool`  
Deletes a word from the Trie. Returns `True` if word was deleted, `False` if not found.

#### `get_all_words() -> List[str]`
Returns a list of all words stored in the Trie.

#### `get_all_words_with_prefix(prefix: str) -> List[str]`
Returns all words that start with the given prefix.

#### `count_words() -> int`
Returns the total number of words in the Trie.

#### `count_words_with_prefix(prefix: str) -> int`
Returns the count of words starting with the given prefix.

#### `longest_common_prefix() -> str`
Returns the longest common prefix of all words in the Trie.

### Magic Methods

#### `__len__() -> int`
Returns the number of words in the Trie.

#### `__contains__(word: str) -> bool`
Enables using the `in` operator to check if a word exists.

#### `__str__() -> str`
Returns a human-readable string representation.

#### `__repr__() -> str` 
Returns a developer-friendly string representation.

## Testing

Run the comprehensive test suite:


The test suite includes:
- Unit tests for all methods
- Edge case testing  
- Performance benchmarks
- Error handling validation
- Unicode and special character support

## Performance Characteristics

The Trie implementation is optimized for:

- **Fast prefix operations**: O(P) where P is prefix length
- **Memory efficient deletion**: Automatic cleanup of unused nodes
- **Case-insensitive operations**: Built-in lowercase normalization
- **Robust error handling**: Graceful handling of edge cases

### Benchmarks
- Insert 1,000 words: ~0.01 seconds
- Search 1,000 words: ~0.005 seconds  
- Memory usage: Approximately 50-100 bytes per word (depending on overlap)

## Best Practices

1. **Case Sensitivity**: The implementation is case-insensitive by default. All words are converted to lowercase.

2. **Input Validation**: Always validate input strings. The implementation handles `None` and empty strings gracefully.

3. **Memory Management**: The delete operation automatically cleans up unused nodes to prevent memory leaks.

4. **Large Datasets**: For very large dictionaries (>1M words), consider implementing disk-based storage or compression.

5. **Thread Safety**: This implementation is not thread-safe. Use appropriate locking mechanisms in concurrent environments.

## Limitations

- **Memory Usage**: Can be memory-intensive for datasets with low prefix overlap
- **Thread Safety**: Not thread-safe without external synchronization  
- **Character Set**: Optimized for ASCII characters; Unicode support is basic
- **Serialization**: No built-in serialization support for persistence

## Contributing

Contributions are welcome! Please ensure:

1. All tests pass: `python test_trie.py`
2. Code follows PEP 8 style guidelines
3. New features include corresponding tests
4. Documentation is updated for API changes

## License

This implementation is part of the PyVerse project and follows the same licensing terms.

## References

- [Trie Data Structure - Wikipedia](https://en.wikipedia.org/wiki/Trie)
- [Introduction to Algorithms, CLRS](https://mitpress.mit.edu/books/introduction-algorithms-third-edition)
- [Algorithms, 4th Edition - Sedgewick & Wayne](https://algs4.cs.princeton.edu/home/)

