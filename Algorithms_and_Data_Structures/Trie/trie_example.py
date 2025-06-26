"""
Trie Data Structure Usage Examples

This module demonstrates various use cases and applications of the Trie data structure,
including autocomplete functionality, spell checking, and dictionary operations.
"""

from trie import Trie

def basic_operations_demo():
    """Demonstrate basic Trie operations."""
    print("=== Basic Trie Operations Demo ===")
    
    # Create a new Trie
    trie = Trie()
    
    # Insert words
    words = ["apple", "app", "application", "apply", "banana", "band", "bandana"]
    print(f"Inserting words: {words}")
    
    for word in words:
        trie.insert(word)
    
    print(f"Total words in Trie: {len(trie)}")
    print(f"All words: {trie.get_all_words()}")
    
    # Search operations
    search_words = ["app", "apple", "appl", "banana", "orange"]
    print("\n--- Search Results ---")
    for word in search_words:
        result = trie.search(word)
        print(f"Search '{word}': {'Found' if result else 'Not Found'}")
    
    # Prefix operations
    prefixes = ["app", "ban", "xyz"]
    print("\n--- Prefix Check Results ---")
    for prefix in prefixes:
        result = trie.starts_with(prefix)
        print(f"Prefix '{prefix}': {'Exists' if result else 'Does not exist'}")
    
    print()

def autocomplete_demo():
    """Demonstrate autocomplete functionality using Trie."""
    print("=== Autocomplete Demo ===")
    
    trie = Trie()
    
    # Dictionary of programming terms
    programming_terms = [
        "algorithm", "array", "binary", "class", "database", "debug", 
        "exception", "function", "hash", "inheritance", "java", "javascript",
        "keyword", "lambda", "method", "object", "parameter", "queue",
        "recursion", "stack", "thread", "variable", "while", "loop",
        "python", "programming", "program", "programmer"
    ]
    
    # Insert all terms
    for term in programming_terms:
        trie.insert(term)
    
    # Simulate autocomplete
    prefixes_to_complete = ["prog", "java", "alg", "py"]
    
    for prefix in prefixes_to_complete:
        suggestions = trie.get_all_words_with_prefix(prefix)
        print(f"Autocomplete for '{prefix}': {suggestions}")
    
    print()

def spell_checker_demo():
    """Demonstrate spell checking functionality."""
    print("=== Spell Checker Demo ===")
    
    trie = Trie()
    
    # Common English words dictionary (simplified)
    dictionary = [
        "hello", "world", "python", "programming", "computer", "science",
        "algorithm", "data", "structure", "implementation", "example",
        "function", "method", "class", "object", "variable", "string",
        "integer", "boolean", "array", "list", "dictionary", "tuple"
    ]
    
    # Build dictionary
    for word in dictionary:
        trie.insert(word)
    
    # Test words (some correct, some misspelled)
    test_words = ["hello", "wrold", "python", "programing", "computer", "algorithmm"]
    
    print("Spell Check Results:")
    for word in test_words:
        is_correct = trie.search(word)
        status = "✓ Correct" if is_correct else "✗ Misspelled"
        print(f"'{word}': {status}")
        
        if not is_correct:
            # Simple suggestion: find words with similar prefix
            suggestions = trie.get_all_words_with_prefix(word[:3])
            if suggestions:
                print(f"  Suggestions: {suggestions[:3]}")  # Show first 3 suggestions
    
    print()

def word_frequency_demo():
    """Demonstrate word frequency tracking."""
    print("=== Word Frequency Demo ===")
    
    trie = Trie()
    
    # Simulate text processing
    text = """
    Python is a programming language. Python is easy to learn.
    Programming with Python is fun. Python programming is powerful.
    """
    
    words = text.lower().replace('\n', ' ').replace('.', '').replace(',', '').split()
    
    # Insert words and track frequency
    word_count = {}
    for word in words:
        if word:  # Skip empty strings
            trie.insert(word)
            word_count[word] = word_count.get(word, 0) + 1
    
    print(f"Processed text: {' '.join(words)}")
    print(f"Unique words in Trie: {len(trie)}")
    print(f"All unique words: {sorted(trie.get_all_words())}")
    
    # Show word frequencies
    print("\nWord Frequencies:")
    for word in sorted(word_count.keys()):
        print(f"'{word}': {word_count[word]} times")
    
    print()

def performance_demo():
    """Demonstrate performance characteristics of Trie operations."""
    print("=== Performance Demo ===")
    
    import time
    import random
    import string
    
    trie = Trie()
    
    # Generate random words for testing
    def generate_random_word(length=5):
        return ''.join(random.choices(string.ascii_lowercase, k=length))
    
    # Insert performance test
    num_words = 1000
    words = [generate_random_word(random.randint(3, 10)) for _ in range(num_words)]
    
    start_time = time.time()
    for word in words:
        trie.insert(word)
    insert_time = time.time() - start_time
    
    print(f"Inserted {num_words} words in {insert_time:.4f} seconds")
    print(f"Average insertion time per word: {(insert_time/num_words)*1000:.4f} ms")
    
    # Search performance test
    search_words = random.sample(words, min(100, len(words)))
    
    start_time = time.time()
    found_count = 0
    for word in search_words:
        if trie.search(word):
            found_count += 1
    search_time = time.time() - start_time
    
    print(f"Searched {len(search_words)} words in {search_time:.4f} seconds")
    print(f"Found {found_count} words")
    print(f"Average search time per word: {(search_time/len(search_words))*1000:.4f} ms")
    
    print()

def advanced_features_demo():
    """Demonstrate advanced Trie features."""
    print("=== Advanced Features Demo ===")
    
    trie = Trie()
    
    # Insert words
    words = ["test", "testing", "tester", "tea", "teach", "teacher", "technology"]
    for word in words:
        trie.insert(word)
    
    print(f"Words in Trie: {trie.get_all_words()}")
    
    # Longest common prefix
    lcp = trie.longest_common_prefix()
    print(f"Longest common prefix: '{lcp}'")
    
    # Count operations
    print(f"Total words: {trie.count_words()}")
    print(f"Words with prefix 'te': {trie.count_words_with_prefix('te')}")
    print(f"Words with prefix 'tea': {trie.count_words_with_prefix('tea')}")
    
    # Delete operations
    print(f"\nBefore deletion: {trie.get_all_words()}")
    trie.delete("test")
    print(f"After deleting 'test': {trie.get_all_words()}")
    
    # Demonstrate __contains__ method
    print(f"\n'testing' in trie: {'testing' in trie}")
    print(f"'test' in trie: {'test' in trie}")
    
    print()

def main():
    """Run all demonstration examples."""
    print("Trie Data Structure - Complete Examples\n")
    
    basic_operations_demo()
    autocomplete_demo()
    spell_checker_demo()
    word_frequency_demo()
    performance_demo()
    advanced_features_demo()
    
    print("All demos completed successfully!")

if __name__ == "__main__":
    main()
