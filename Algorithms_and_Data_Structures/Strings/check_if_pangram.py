'''
a pangram is a sentence where every letter of
the english alphabet appears at least once.

given a string s, we check if that string is a
a pangram or not
'''

class pangram:
    def check_if_pangram(self, s: str) -> bool:
        s = set(s.lower().strip())
        alphabet = set('abcdefghijklmnopqrstuvwxyz')
        if (alphabet.issubset(s)):
            return True
        return False

#example usage
if __name__=='__main__':
    
    #create an instance of the pangram class
    pangram_check = pangram()
    
    #call the check_if_pangram method
    result = pangram.check_if_pangram("Sphinx of black quartz, judge my vow")
    print(result)