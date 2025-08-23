# Manacher's Algorithm

## Overview

Manacher's Algorithm is an efficient algorithm for finding the longest palindromic substring in a string in O(n) time and O(n) space. It uses the concept of palindrome centers and radius expansion to achieve linear time complexity, making it the optimal solution for palindromic substring problems.

## Theory

### Key Concepts

1. **Palindrome Centers**: Points around which palindromes are centered
2. **Palindrome Radius**: Distance from center to palindrome boundary
3. **Mirror Property**: Symmetry around palindrome centers
4. **Expansion**: Extending palindrome boundaries

### Core Operations

1. **Preprocessing**: Add special characters to handle even-length palindromes
2. **Center Expansion**: Expand around each potential center
3. **Mirror Optimization**: Use symmetry to avoid redundant calculations
4. **Radius Tracking**: Maintain palindrome radius at each position

### Mathematical Foundation

- **Time Complexity**: O(n) where n is string length
- **Space Complexity**: O(n) for radius array
- **Mirror Property**: P[i] = min(R - i, P[mirror])
- **Expansion**: Continue expanding while characters match

## Applications

1. **Longest Palindromic Substring**: Find LPS in linear time
2. **Palindrome Counting**: Count all palindromic substrings
3. **String Processing**: Text analysis and processing
4. **Competitive Programming**: Fast palindrome solutions
5. **DNA Analysis**: Biological sequence analysis
6. **Pattern Recognition**: Symmetry detection

## Algorithm Implementation

### Core Functions

```python
class ManachersAlgorithm:
    def __init__(self, text):
        self.text = text
        self.processed_text = self._preprocess(text)
        self.palindrome_radius = []
        self._compute_palindromes()
    
    def _preprocess(self, text):
        """Add special characters for even-length palindromes"""
        
    def _compute_palindromes(self):
        """Compute palindrome radius for each position"""
        
    def get_longest_palindrome(self):
        """Get longest palindromic substring"""
        
    def get_all_palindromes(self):
        """Get all palindromic substrings"""
        
    def count_palindromes(self):
        """Count total number of palindromic substrings"""
        
    def is_palindrome(self, start, end):
        """Check if substring is palindrome"""
```

## Usage Examples

### Basic Operations

```python
# Create Manacher's algorithm instance
text = "babad"
ma = ManachersAlgorithm(text)

# Get longest palindrome
longest = ma.get_longest_palindrome()  # Returns "bab" or "aba"

# Count palindromes
count = ma.count_palindromes()  # Returns total count

# Check specific substring
# Check specific substring
is_pal = ma.is_palindrome(0, 3)  # Check if "bab" is palindrome
```

### Advanced Operations

```python
# Get all palindromes
all_palindromes = ma.get_all_palindromes()

# Get palindrome statistics
stats = ma.get_palindrome_statistics()

# Find palindromes of specific length
length_3_palindromes = ma.get_palindromes_of_length(3)
```

## Advanced Features

### 1. Multiple String Support
- Compare palindromes across strings
- Find common palindromic patterns
- Efficient batch processing

### 2. Palindrome Classification
- Even-length palindromes
- Odd-length palindromes
- Palindromic prefixes/suffixes

### 3. Memory Optimization
- Sparse representation
- Compression techniques
- Cache-friendly implementation

### 4. Performance Monitoring
- Construction time analysis
- Query performance tracking
- Memory usage optimization

## Performance Analysis

### Time Complexity
- **Preprocessing**: O(n) for string length n
- **Palindrome Computation**: O(n)
- **Query Operations**: O(1) for precomputed data
- **Total Time**: O(n)

### Space Complexity
- **Storage**: O(n) for radius array
- **Preprocessed String**: O(n) additional space
- **Memory Efficiency**: Good for large strings

### Memory Usage
- **Efficient**: Only stores necessary information
- **Cache Friendly**: Good locality for queries
- **Compact**: Minimal memory overhead

## Visualization

### Algorithm Process
- Visual representation of expansion process
- Show mirror property utilization
- Highlight palindrome centers

### Palindrome Detection
- Animate palindrome expansion
- Show radius calculation
- Visualize symmetry properties

### Performance Analysis
- Show computation time distribution
- Visualize memory usage
- Highlight optimization techniques

## Test Cases

### Basic Functionality
```python
def test_basic_operations():
    text = "babad"
    ma = ManachersAlgorithm(text)
    
    # Test longest palindrome
    longest = ma.get_longest_palindrome()
    assert len(longest) == 3  # "bab" or "aba"
    
    # Test palindrome count
    count = ma.count_palindromes()
    assert count > 0
```

### Advanced Scenarios
```python
def test_edge_cases():
    # Test empty string
    ma1 = ManachersAlgorithm("")
    assert ma1.get_longest_palindrome() == ""
    
    # Test single character
    ma2 = ManachersAlgorithm("a")
    assert ma2.get_longest_palindrome() == "a"
    
    # Test all same characters
    ma3 = ManachersAlgorithm("aaa")
    assert len(ma3.get_longest_palindrome()) == 3
```

### Performance Tests
```python
def test_performance():
    import time
    
    # Large string test
    text = "a" * 10000 + "b" * 10000
    start_time = time.time()
    ma = ManachersAlgorithm(text)
    build_time = time.time() - start_time
    
    # Query test
    start_time = time.time()
    longest = ma.get_longest_palindrome()
    query_time = time.time() - start_time
    
    assert build_time < 1.0  # Should be fast
    assert query_time < 0.1  # Should be very fast
```

## Dependencies

### Required Libraries
```python
import time
import numpy as np
from typing import List, Optional, Tuple, Dict, Any
import matplotlib.pyplot as plt
import networkx as nx
from collections import defaultdict
```

### Optional Dependencies
```python
# For advanced visualizations
import seaborn as sns
from matplotlib.animation import FuncAnimation
```

## File Structure

```
Algorithms_and_Data_Structures/Advanced Algorithms/Manachers_Algorithm/
├── README.md
├── manachers_algorithm.py
└── test_manachers_algorithm.py

## Complexity Summary

| Operation | Time Complexity | Space Complexity | Description |
|-----------|----------------|------------------|-------------|
| Construction | O(n) | O(n) | Build palindrome array |
| Longest Palindrome | O(1) | O(1) | Get longest palindromic substring |
| Count Palindromes | O(n) | O(1) | Count all palindromic substrings |
| Check Palindrome | O(1) | O(1) | Check if substring is palindrome |
| Get All Palindromes | O(n) | O(n) | Get all palindromic substrings |
| Palindrome Statistics | O(n) | O(1) | Get palindrome statistics |

## Applications in Real-World

1. **Text Processing**: Document analysis and processing
2. **Competitive Programming**: Fast palindrome solutions
3. **DNA Analysis**: Biological sequence analysis
4. **Pattern Recognition**: Symmetry detection
5. **Data Compression**: Palindrome-based compression
6. **Network Security**: Intrusion detection

## Advanced Topics

### 1. Generalized Manacher's
- Support for multiple strings
- Efficient comparison algorithms
- Pattern matching applications

### 2. Dynamic Updates
- Support for string modifications
- Efficient rebuilding
- Incremental updates

### 3. Compressed Representation
- Memory optimization
- Compression techniques
- Cache-friendly implementations

### 4. Specialized Variants
- Palindromic tree
- Eertree
- Palindrome automata

## Implementation Notes

1. **Preprocessing**: Proper special character insertion
2. **Mirror Property**: Efficient symmetry utilization
3. **Memory Management**: Efficient array allocation
4. **Error Handling**: Robust input validation
5. **Performance Optimization**: Cache-friendly implementation

## Future Enhancements

1. **GPU Acceleration**: CUDA/OpenCL implementations
2. **Compression**: Memory-efficient representations
3. **Distributed Processing**: Multi-node algorithms
4. **Specialized Variants**: Domain-specific optimizations
5. **Integration**: Text processing tool integration 