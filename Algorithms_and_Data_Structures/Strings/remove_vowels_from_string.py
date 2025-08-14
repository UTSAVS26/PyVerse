'''
given a string s, we are going to remove all vowels from
it and return a new, vowel-less string
'''

class remove_vowels:
    def remove_vowels_from_string(self, s: str) -> str:
        vowels = ['a', 'e', 'i', 'o', 'u', 'A', 'E', 'I', 'O', 'U']
        for i in vowels:
            s = s.replace(i, "")
        return s

#example usage
if __name__=='__main__':
    
    #create an instance of the remove_vowels class
    vowel_destroyer = remove_vowels()
    
    #call the remove_vowels_from_string method
    result = vowel_destroyer.remove_vowels_from_string("Abracadabra")
    print(result)