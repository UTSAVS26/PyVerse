# aho_corasick.py

class AhoCorasick:
    def __init__(self):
        self.num_nodes = 1
        self.edges = [{}]
        self.fail = [-1]
        self.output = [[]]

    def add_word(self, word, index):
        """
        Adds a word to the Trie structure.

        Parameters:
        word (str): The word to add.
        index (int): The index of the word for output.
        """
        current_node = 0
        for char in word:
            if char not in self.edges[current_node]:
                self.edges[current_node][char] = self.num_nodes
                self.edges.append({})
                self.fail.append(-1)
                self.output.append([])
                self.num_nodes += 1
            current_node = self.edges[current_node][char]
        self.output[current_node].append(index)

    def build(self):
        """
        Constructs the failure links for the Trie structure.
        """
        from collections import deque
        queue = deque()
        for char in self.edges[0]:
            child_node = self.edges[0][char]
            self.fail[child_node] = 0
            queue.append(child_node)

        while queue:
            current_node = queue.popleft()
            for char in self.edges[current_node]:
                child_node = self.edges[current_node][char]
                queue.append(child_node)
                fallback_node = self.fail[current_node]
                while fallback_node != -1 and char not in self.edges[fallback_node]:
                    fallback_node = self.fail[fallback_node]
                self.fail[child_node] = self.edges[fallback_node].get(char, 0)
                self.output[child_node].extend(self.output[self.fail[child_node]])

    def search(self, text):
        """
        Searches for patterns in the given text using the Aho-Corasick algorithm.

        Parameters:
        text (str): The text to search for patterns.

        Prints the starting index of each found pattern.
        """
        current_node = 0
        for i in range(len(text)):
            while current_node != -1 and text[i] not in self.edges[current_node]:
                current_node = self.fail[current_node]
            if current_node == -1:
                current_node = 0
                continue
            current_node = self.edges[current_node][text[i]]
            for pattern_index in self.output[current_node]:
                print(f"Pattern found at index {i}")

# Example usage
if __name__ == "__main__":
    ac = AhoCorasick()
    patterns = ["he", "she", "his", "hers"]
    for index, pattern in enumerate(patterns):
        ac.add_word(pattern, index)
    ac.build()
    ac.search("ushers")
