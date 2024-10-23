from typing import List
import os

# Structure of the Trie Tree
class TrieNode:
  def __init__(self):
    self.children = {}
    self.is_end_of_word = False

# main class
class Trie:
  def __init__(self):
    self.root = TrieNode()
  
  def insert(self, word:str) -> None:
    # function to insert a single word in the trie
    if word == "":
      return

    current_node = self.root  
    
    for char in word:
    
      if char not in current_node.children:
        current_node.children[char] = TrieNode()
      current_node = current_node.children[char]
    
    current_node.is_end_of_word = True
  
  def search(self, word:str) -> bool:
    # function to search if a word is present in the trie or not
    # return type boolean value (True/False)    
    current_node = self.root

    for char in word:
      if char not in current_node.children:
        return False
      
      current_node = current_node.children[char]
    
    # return True if word is found in the trie and is the end of a word
    return current_node.is_end_of_word
      
  def starts_with(self, prefix:str) -> List[str]:
    # fucntion to search for a given pattern in the trie
    # return a boolean value
    def _collect_words(node:TrieNode, prefix: str) -> List[str]:
      # return a list of words that starts with given prefix
      words = []
      if node.is_end_of_word:
        words.append(prefix)
      for char, child_node in node.children.items():
        words.extend(_collect_words(child_node, prefix+char))
      return words
    
    current_node = self.root
    for char in prefix:
      if char not in current_node.children:
        return []

      current_node = current_node.children[char]
    return _collect_words(current_node, prefix)
  
  def delete(self, word:str) -> bool:
    # function to delete a word from trie

    def _delete(node: TrieNode, word:str, depth:int ) -> bool:
      # if trie is empty
      if not node:
        return False

      if depth == len(word):
        # for last character of the word , unmarks it as the end of word
        if not node.is_end_of_word:
          return False
        
        node.is_end_of_word = False

        # return if node has any child , if not it is safe to be deleted
        return len(node.children) == 0
      
      char = word[depth]
      if char not in node.children:
        return False
      
      # check if character can be deleted
      safe_to_delete = _delete(node.children[char], word, depth+1)

      # if true, delete the child node reference
      if safe_to_delete:
        del node.children[char]
        return len(node.children) == 0 and not node.is_end_of_word

      return False
    
    return _delete(self.root, word, 0)
      
def menu():
  trie = Trie()
  os.system('cls' if os.name == 'nt' else 'clear')
  while True:
    choice = int(input("\n\t-----| Prefix Tree |-----\n\n \t1. Insert word \n\t2. Search word \n\t3. Search prefix pattern \n\t4. Delete Word \n\t5.Exit \n\n\tEnter your choice : "))

    if choice == 1:
      word = input('\n\tEnter the word to insert : ')
      trie.insert(word)
      print("\nWord added successfuly ... ")
    
    elif choice == 2:
      word = input('\n\tEnter the word to search : ')
      is_present = trie.search(word)
      if is_present:
        print(f"\n{word} found in the Prefix Tree")
      else:
        print(f"\n{word} not present in the Prefix Tree")
    
    elif choice == 3:
      prefix = input("\n\tEnter the prefix pattern to search for : ")
      list_of_word = trie.starts_with(prefix)

      if len(list_of_word)>0:
        print(f"\nFollowing words matches the given prefix pattern '{prefix}'\n")
        for word in list_of_word:
          print(f"\t{word}")
      else:
        print(f"\nprefix '{prefix}' does not match with any word in the Data Structures")
    
    elif choice == 4:
      word = input("\n\tEnter the word to delete : ")
      is_deleted = trie.delete(word)

      if is_deleted:
        print("\nWord deleted from the Prefix Tree successfully ... ")
      else:
        print(f"\nUnable to delete word {word}")
    
    elif choice == 5:
      print("\nExiting......")
      break
    else:
      print("\nInvalid Input")
    

if __name__ == "__main__":
  menu()