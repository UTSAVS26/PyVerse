"""
Comprehensive Test Suite for Trie Data Structure

This module contains extensive unit tests to verify the correctness and robustness
of the Trie implementation, covering all methods and edge cases.
"""

import unittest
import sys
import os

# Add the current directory to Python path to import trie module
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from trie import Trie, TrieNode

class TestTrieNode(unittest.TestCase):
    """Test cases for TrieNode class."""
    
    def test_node_initialization(self):
        """Test TrieNode initialization."""
        node = TrieNode()
        self.assertEqual(node.children, {})
        self.assertFalse(node.is_end_of_word)
        self.assertEqual(node.word_count, 0)

class TestTrieBasicOperations(unittest.TestCase):
    """Test cases for basic Trie operations."""
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.trie = Trie()
    
    def test_initialization(self):
        """Test Trie initialization."""
        self.assertIsNotNone(self.trie.root)
        self.assertEqual(len(self.trie), 0)
        self.assertEqual(self.trie.total_words, 0)
    
    def test_insert_single_word(self):
        """Test inserting a single word."""
        result = self.trie.insert("hello")
        self.assertTrue(result)
        self.assertEqual(len(self.trie), 1)
        self.assertTrue(self.trie.search("hello"))
    
    def test_insert_multiple_words(self):
        """Test inserting multiple words."""
        words = ["apple", "app", "application"]
        for word in words:
            self.trie.insert(word)
        
        self.assertEqual(len(self.trie), 3)
        for word in words:
            self.assertTrue(self.trie.search(word))
    
    def test_insert_empty_word(self):
        """Test inserting empty word."""
        result = self.trie.insert("")
        self.assertFalse(result)
        self.assertEqual(len(self.trie), 0)
    
    def test_insert_duplicate_words(self):
        """Test inserting duplicate words."""
        self.trie.insert("test")
        self.trie.insert("test")
        self.assertEqual(len(self.trie), 1)  # Should still be 1 unique word
    
    def test_case_insensitive_insert(self):
        """Test case-insensitive insertion."""
        self.trie.insert("Hello")
        self.trie.insert("HELLO")
        self.trie.insert("hello")
        self.assertEqual(len(self.trie), 1)  # Should be treated as same word

class TestTrieSearchOperations(unittest.TestCase):
    """Test cases for Trie search operations."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.trie = Trie()
        words = ["cat", "car", "card", "care", "careful", "cars", "carry"]
        for word in words:
            self.trie.insert(word)
    
    def test_search_existing_words(self):
        """Test searching for existing words."""
        test_words = ["cat", "car", "card", "care", "careful", "cars", "carry"]
        for word in test_words:
            self.assertTrue(self.trie.search(word))
    
    def test_search_non_existing_words(self):
        """Test searching for non-existing words."""
        test_words = ["ca", "cart", "caring", "dog", ""]
        for word in test_words:
            self.assertFalse(self.trie.search(word))
    
    def test_search_case_insensitive(self):
        """Test case-insensitive search."""
        self.assertTrue(self.trie.search("CAT"))
        self.assertTrue(self.trie.search("Car"))
        self.assertTrue(self.trie.search("CAREFUL"))
    
    def test_contains_operator(self):
        """Test __contains__ method (in operator)."""
        self.assertTrue("cat" in self.trie)
        self.assertTrue("CAR" in self.trie)
        self.assertFalse("dog" in self.trie)
        self.assertFalse("ca" in self.trie)

class TestTriePrefixOperations(unittest.TestCase):
    """Test cases for Trie prefix operations."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.trie = Trie()
        words = ["tea", "teach", "teacher", "teaching", "team", "technology"]
        for word in words:
            self.trie.insert(word)
    
    def test_starts_with_valid_prefixes(self):
        """Test starts_with for valid prefixes."""
        prefixes = ["te", "tea", "teac", "teach", "team", "tech"]
        for prefix in prefixes:
            self.assertTrue(self.trie.starts_with(prefix))
    
    def test_starts_with_invalid_prefixes(self):
        """Test starts_with for invalid prefixes."""
        prefixes = ["ta", "tem", "xyz", "teaching123"]
        for prefix in prefixes:
            self.assertFalse(self.trie.starts_with(prefix))
    
    def test_starts_with_empty_prefix(self):
        """Test starts_with with empty prefix."""
        self.assertTrue(self.trie.starts_with(""))
    
    def test_get_all_words_with_prefix(self):
        """Test getting all words with a specific prefix."""
        words_with_tea = self.trie.get_all_words_with_prefix("tea")
        expected = ["tea", "teach", "teacher", "teaching", "team"]
        self.assertEqual(sorted(words_with_tea), sorted(expected))
    
    def test_get_all_words_with_invalid_prefix(self):
        """Test getting words with invalid prefix."""
        words = self.trie.get_all_words_with_prefix("xyz")
        self.assertEqual(words, [])
    
    def test_count_words_with_prefix(self):
        """Test counting words with prefix."""
        count = self.trie.count_words_with_prefix("tea")
        self.assertEqual(count, 5)  # tea, teach, teacher, teaching, team
        
        count = self.trie.count_words_with_prefix("tech")
        self.assertEqual(count, 1)  # technology

class TestTrieDeleteOperations(unittest.TestCase):
    """Test cases for Trie delete operations."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.trie = Trie()
        words = ["cat", "cats", "car", "card", "care", "careful"]
        for word in words:
            self.trie.insert(word)
    
    def test_delete_existing_word(self):
        """Test deleting existing words."""
        initial_count = len(self.trie)
        result = self.trie.delete("cat")
        self.assertTrue(result)
        self.assertFalse(self.trie.search("cat"))
        self.assertEqual(len(self.trie), initial_count - 1)
    
    def test_delete_non_existing_word(self):
        """Test deleting non-existing words."""
        initial_count = len(self.trie)
        result = self.trie.delete("dog")
        self.assertFalse(result)
        self.assertEqual(len(self.trie), initial_count)
    
    def test_delete_prefix_not_word(self):
        """Test deleting a prefix that's not a complete word."""
        initial_count = len(self.trie)
        result = self.trie.delete("ca")  # prefix but not a word
        self.assertFalse(result)
        self.assertEqual(len(self.trie), initial_count)
    
    def test_delete_word_with_children(self):
        """Test deleting a word that has children."""
        result = self.trie.delete("car")
        self.assertTrue(result)
        self.assertFalse(self.trie.search("car"))
        # Children should still exist
        self.assertTrue(self.trie.search("card"))
        self.assertTrue(self.trie.search("care"))
    
    def test_delete_leaf_word(self):
        """Test deleting a leaf word."""
        result = self.trie.delete("careful")
        self.assertTrue(result)
        self.assertFalse(self.trie.search("careful"))
        # Parent should still exist
        self.assertTrue(self.trie.search("care"))

class TestTrieAdvancedFeatures(unittest.TestCase):
    """Test cases for advanced Trie features."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.trie = Trie()
    
    def test_get_all_words_empty_trie(self):
        """Test getting all words from empty Trie."""
        words = self.trie.get_all_words()
        self.assertEqual(words, [])
    
    def test_get_all_words_populated_trie(self):
        """Test getting all words from populated Trie."""
        test_words = ["apple", "app", "application", "banana"]
        for word in test_words:
            self.trie.insert(word)
        
        all_words = self.trie.get_all_words()
        self.assertEqual(sorted(all_words), sorted(test_words))
    
    def test_longest_common_prefix_empty_trie(self):
        """Test longest common prefix with empty Trie."""
        lcp = self.trie.longest_common_prefix()
        self.assertEqual(lcp, "")
    
    def test_longest_common_prefix_single_word(self):
        """Test longest common prefix with single word."""
        self.trie.insert("hello")
        lcp = self.trie.longest_common_prefix()
        self.assertEqual(lcp, "hello")
    
    def test_longest_common_prefix_multiple_words(self):
        """Test longest common prefix with multiple words."""
        words = ["testing", "test", "tester"]
        for word in words:
            self.trie.insert(word)
        lcp = self.trie.longest_common_prefix()
        self.assertEqual(lcp, "test")
    
    def test_longest_common_prefix_no_common(self):
        """Test longest common prefix with no common prefix."""
        words = ["apple", "banana", "cherry"]
        for word in words:
            self.trie.insert(word)
        lcp = self.trie.longest_common_prefix()
        self.assertEqual(lcp, "")
    
    def test_count_operations(self):
        """Test various count operations."""
        words = ["test", "testing", "tester", "tea", "team"]
        for word in words:
            self.trie.insert(word)
        
        self.assertEqual(self.trie.count_words(), 5)
        self.assertEqual(self.trie.count_words_with_prefix("test"), 3)
        self.assertEqual(self.trie.count_words_with_prefix("tea"), 2)
        self.assertEqual(self.trie.count_words_with_prefix("xyz"), 0)

class TestTrieEdgeCases(unittest.TestCase):
    """Test cases for edge cases and error handling."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.trie = Trie()
    
    def test_none_input(self):
        """Test operations with None input."""
        self.assertFalse(self.trie.insert(None))
        self.assertFalse(self.trie.search(None))
        self.assertFalse(self.trie.delete(None))
    
    def test_special_characters(self):
        """Test with special characters."""
        special_words = ["hello-world", "test_case", "file.txt", "user@domain.com"]
        for word in special_words:
            self.trie.insert(word)
            self.assertTrue(self.trie.search(word))
    
    def test_unicode_characters(self):
        """Test with Unicode characters."""
        unicode_words = ["café", "naïve", "résumé", "piñata"]
        for word in unicode_words:
            self.trie.insert(word)
            self.assertTrue(self.trie.search(word))
    
    def test_very_long_words(self):
        """Test with very long words."""
        long_word = "a" * 1000
        self.trie.insert(long_word)
        self.assertTrue(self.trie.search(long_word))
    
    def test_single_character_words(self):
        """Test with single character words."""
        single_chars = ["a", "b", "c", "x", "y", "z"]
        for char in single_chars:
            self.trie.insert(char)
            self.assertTrue(self.trie.search(char))

class TestTrieStringRepresentation(unittest.TestCase):
    """Test cases for string representation methods."""
    
    def test_str_method(self):
        """Test __str__ method."""
        trie = Trie()
        trie.insert("test")
        str_repr = str(trie)
        self.assertIn("Trie", str_repr)
        self.assertIn("test", str_repr)
    
    def test_repr_method(self):
        """Test __repr__ method."""
        trie = Trie()
        trie.insert("test")
        repr_str = repr(trie)
        self.assertIn("Trie", repr_str)
        self.assertIn("total_words=1", repr_str)

def run_performance_tests():
    """Run performance tests to ensure reasonable execution times."""
    import time
    
    print("\n=== Performance Tests ===")
    
    trie = Trie()
    
    # Test insertion performance
    start_time = time.time()
    for i in range(1000):
        trie.insert(f"word{i}")
    insert_time = time.time() - start_time
    print(f"Inserted 1000 words in {insert_time:.4f} seconds")
    
    # Test search performance
    start_time = time.time()
    for i in range(1000):
        trie.search(f"word{i}")
    search_time = time.time() - start_time
    print(f"Searched 1000 words in {search_time:.4f} seconds")
    
    # Performance assertions (reasonable thresholds)
    assert insert_time < 1.0, f"Insertion too slow: {insert_time} seconds"
    assert search_time < 1.0, f"Search too slow: {search_time} seconds"
    
    print("All performance tests passed!")

def main():
    """Run all tests."""
    print("Running Trie Data Structure Test Suite...\n")
    
    # Create test suite
    test_classes = [
        TestTrieNode,
        TestTrieBasicOperations,
        TestTrieSearchOperations,
        TestTriePrefixOperations,
        TestTrieDeleteOperations,
        TestTrieAdvancedFeatures,
        TestTrieEdgeCases,
        TestTrieStringRepresentation
    ]
    
    # Run unit tests
    suite = unittest.TestSuite()
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        suite.addTests(tests)
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Run performance tests
    if result.wasSuccessful():
        run_performance_tests()
        print(f"\n✅ All tests passed! ({result.testsRun} tests run)")
    else:
        print(f"\n❌ {len(result.failures)} test(s) failed, {len(result.errors)} error(s)")
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
