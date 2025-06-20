"""
Trie (Prefix Tree) Data Structure Implementation

A Trie is a tree-like data structure that stores a dynamic set of strings,
where the keys are usually strings. It's particularly useful for:
- Autocomplete features
- Spell checkers
- Dictionary implementations
- IP routing tables
- String prefix matching

Time Complexity:
- Insert: O(L) where L is the length of the word
- Search: O(L) where L is the length of the word
- StartsWith: O(P) where P is the length of the prefix
- Delete: O(L) where L is the length of the word

Space Complexity: O(ALPHABET_SIZE * N * L) where N is number of words and L is average length
"""

class TrieNode:
    """
    Node class for Trie data structure.
    Each node represents a character and contains references to child nodes.
    """
    
    def __init__(self):
        """
        Initialize a new TrieNode.
        
        Attributes:
            children (dict): Dictionary mapping characters to child TrieNodes
            is_end_of_word (bool): Flag indicating if this node marks the end of a word
            word_count (int): Number of words that end at this node (for duplicate handling)
        """
        self.children = {}
        self.is_end_of_word = False
        self.word_count = 0


class Trie:
    """
    Trie (Prefix Tree) implementation with comprehensive functionality.
    
    Supports insertion, search, prefix matching, deletion, and various utility operations.
    """
    
    def __init__(self):
        """Initialize an empty Trie with a root node."""
        self.root = TrieNode()
        self.total_words = 0
    
    def insert(self, word):
        """
        Insert a word into the Trie.
        
        Args:
            word (str): The word to insert into the Trie
            
        Returns:
            bool: True if word was inserted successfully
            
        Time Complexity: O(L) where L is the length of the word
        """
        if not word:
            return False
        
        word = word.lower()  # Convert to lowercase for case-insensitive operations
        current = self.root
        
        # Traverse through each character in the word
        for char in word:
            if char not in current.children:
                current.children[char] = TrieNode()
            current = current.children[char]
        
        # Mark the end of the word
        if not current.is_end_of_word:
            current.is_end_of_word = True
            self.total_words += 1
        
        current.word_count += 1
        return True
    
    def search(self, word):
        """
        Search for a complete word in the Trie.
        
        Args:
            word (str): The word to search for
            
        Returns:
            bool: True if the word exists in the Trie
            
        Time Complexity: O(L) where L is the length of the word
        """
        if not word:
            return False
        
        word = word.lower()
        node = self._find_node(word)
        return node is not None and node.is_end_of_word
    
    def starts_with(self, prefix):
        """
        Check if any word in the Trie starts with the given prefix.
        
        Args:
            prefix (str): The prefix to search for
            
        Returns:
            bool: True if any word starts with the prefix
            
        Time Complexity: O(P) where P is the length of the prefix
        """
        if not prefix:
            return True
        
        prefix = prefix.lower()
        node = self._find_node(prefix)
        return node is not None
    
    def delete(self, word):
        """
        Delete a word from the Trie.
        
        Args:
            word (str): The word to delete
            
        Returns:
            bool: True if the word was successfully deleted
            
        Time Complexity: O(L) where L is the length of the word
        """
        if not word or not self.search(word):
            return False
        
        word = word.lower()
        
        def _delete_helper(node, word, index):
            """
            Recursive helper function for deletion.
            
            Returns:
                bool: True if the current node should be deleted
            """
            if index == len(word):
                # We've reached the end of the word
                if not node.is_end_of_word:
                    return False
                
                node.word_count -= 1
                if node.word_count == 0:
                    node.is_end_of_word = False
                    self.total_words -= 1
                
                # Delete this node if it has no children and is not end of another word
                return len(node.children) == 0 and not node.is_end_of_word
            
            char = word[index]
            child_node = node.children.get(char)
            
            if not child_node:
                return False
            
            should_delete_child = _delete_helper(child_node, word, index + 1)
            
            if should_delete_child:
                del node.children[char]
                # Delete current node if it has no children and is not end of word
                return len(node.children) == 0 and not node.is_end_of_word
            
            return False
        
        _delete_helper(self.root, word, 0)
        return True
    
    def get_all_words_with_prefix(self, prefix):
        """
        Get all words in the Trie that start with the given prefix.
        
        Args:
            prefix (str): The prefix to search for
            
        Returns:
            list: List of all words starting with the prefix
            
        Time Complexity: O(P + N) where P is prefix length and N is number of nodes in subtree
        """
        if not prefix:
            return self.get_all_words()
        
        prefix = prefix.lower()
        words = []
        node = self._find_node(prefix)
        
        if node:
            self._collect_words(node, prefix, words)
        
        return words
    
    def get_all_words(self):
        """
        Get all words stored in the Trie.
        
        Returns:
            list: List of all words in the Trie
            
        Time Complexity: O(N) where N is the total number of nodes
        """
        words = []
        self._collect_words(self.root, "", words)
        return words
    
    def count_words(self):
        """
        Get the total number of words in the Trie.
        
        Returns:
            int: Total number of words
            
        Time Complexity: O(1)
        """
        return self.total_words
    
    def count_words_with_prefix(self, prefix):
        """
        Count the number of words that start with the given prefix.
        
        Args:
            prefix (str): The prefix to count words for
            
        Returns:
            int: Number of words starting with the prefix
            
        Time Complexity: O(P + N) where P is prefix length and N is number of nodes in subtree
        """
        return len(self.get_all_words_with_prefix(prefix))
    
    def longest_common_prefix(self):
        """
        Find the longest common prefix of all words in the Trie.
        
        Returns:
            str: The longest common prefix
            
        Time Complexity: O(L) where L is the length of the longest common prefix
        """
        if self.total_words == 0:
            return ""
        
        current = self.root
        prefix = ""
        
        while len(current.children) == 1 and not current.is_end_of_word:
            char = next(iter(current.children))
            prefix += char
            current = current.children[char]
        
        return prefix
    
    def _find_node(self, word):
        """
        Helper method to find the node corresponding to a word/prefix.
        
        Args:
            word (str): The word/prefix to find
            
        Returns:
            TrieNode or None: The node if found, None otherwise
        """
        current = self.root
        
        for char in word:
            if char not in current.children:
                return None
            current = current.children[char]
        
        return current
    
    def _collect_words(self, node, current_word, words):
        """
        Helper method to collect all words from a given node.
        
        Args:
            node (TrieNode): The starting node
            current_word (str): The word formed so far
            words (list): List to collect words in
        """
        if node.is_end_of_word:
            words.append(current_word)
        
        for char, child_node in node.children.items():
            self._collect_words(child_node, current_word + char, words)
    
    def __len__(self):
        """Return the number of words in the Trie."""
        return self.total_words
    
    def __contains__(self, word):
        """Check if a word is in the Trie using 'in' operator."""
        return self.search(word)
    
    def __str__(self):
        """String representation of the Trie."""
        return f"Trie(words={self.total_words}, all_words={self.get_all_words()})"
    
    def __repr__(self):
        """Developer-friendly string representation."""
        return f"Trie(total_words={self.total_words})"
