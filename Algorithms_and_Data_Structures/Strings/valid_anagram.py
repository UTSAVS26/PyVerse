'''
we are going to use a hashmap approach to count the number of characters in both the strings and
finally compare both hashmaps.
'''
from collections import defaultdict

class valid_anagram:
    #method to determine if two strings are valid anagrams.
    def isAnagram(self, s: str, t: str) -> bool:
        
        #initialize two dictionaries to be used as hashmaps.
        dictis=defaultdict()
        dictit=defaultdict()
        
        #count the occurences of each character in both the strings.
        for i in s:
            dictis[i]=dictis.get(i,0)+1
        
        for i in t:
            dictit[i]=dictit.get(i,0)+1
        
        #if both the dictionaries are the same, return True 
        return True if dictis==dictit else False

#example usage
if __name__=='__main__':
    
    #create an instance of the valid_anagram class.
    anagram_checker=valid_anagram()
    
    #call the isAnagram method
    result = anagram_checker.isAnagram("dusty","study")
    print(result)