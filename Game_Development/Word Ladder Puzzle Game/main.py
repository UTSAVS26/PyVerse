import random

class WordLadderPuzzle:
    def __init__(self):
        self.words = set(self.load_words())

    def load_words(self):
        # In a real implementation, this would load from a file
        return ['cat', 'cot', 'dot', 'dog', 'log', 'lag', 'bag', 'big', 'pig', 'pin', 'pan']

    def get_neighbors(self, word):
        return [w for w in self.words if self.is_neighbor(word, w)]

    def is_neighbor(self, word1, word2):
        if len(word1) != len(word2):
            return False
        return sum(c1 != c2 for c1, c2 in zip(word1, word2)) == 1

    def find_ladder(self, start, end):
        if start not in self.words or end not in self.words:
            return None
        
        queue = [(start, [start])]
        visited = set()

        while queue:
            (word, path) = queue.pop(0)
            if word not in visited:
                visited.add(word)
                
                if word == end:
                    return path
                
                for neighbor in self.get_neighbors(word):
                    if neighbor not in visited:
                        queue.append((neighbor, path + [neighbor]))
        
        return None

    def play(self):
        print("Welcome to Word Ladder Puzzle!")
        start = input("Enter the starting word: ").lower()
        end = input("Enter the target word: ").lower()

        if len(start) != len(end):
            print("Words must be of the same length.")
            return

        ladder = self.find_ladder(start, end)
        if ladder:
            print(f"Found a ladder in {len(ladder) - 1} steps:")
            print(" -> ".join(ladder))
        else:
            print("No valid word ladder found.")

if __name__ == "__main__":
    game = WordLadderPuzzle()
    game.play()