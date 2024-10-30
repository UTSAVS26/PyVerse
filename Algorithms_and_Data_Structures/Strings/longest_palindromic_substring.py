'''
longest palindromic substring without DP
We will be using a two pointer approach (l and r are the two pointers). We will be moving 
from the middle of the string to its extreme ends and check whether the characters
on either side of the string match.
'''
class longestPalindrome:
    def longestPalindromeFinder(self, s: str) -> str:
        #declare a variable to store the temporary length each time a palindrome is found        
        temp_len=0
        
        #declare a maximum length variable to store the overall maximum length
        max_len=0
        l=0
        r=0

        for i in range(len(s)):
            #odd palindrome
            #start from the middle of the string

            l=i
            r=i
            
            #As long as the character left pointer points to and right pointer points to, match
            #increment the right pointer and decrement the left one.
            while(l>=0 and r<=len(s)-1 and s[l]==s[r]):
                temp_len= r-l+1
                
                #if a new maximum length is found, store it in the max_len variable.
                if(temp_len>max_len):
                    max_len=temp_len
                    start=l

                l-=1
                r+=1
            
            #even palindrome
            #start from the two middle-most characters
            l=i
            r=i+1
            
            #the remaining procedure remains the same as that of odd palindrome.
            while(l>=0 and r<=len(s)-1 and s[l]==s[r]):
                temp_len= r-l+1
                if(temp_len>max_len):
                    max_len=temp_len
                    start=l

                l-=1
                r+=1
        #return a substring of the original string which contains the longest palindrome found so far.
        return s[start:start+max_len]
    
#example usage
if __name__=='__main__':
    
    #create an instance of the class longestPalindrome and print the result
    longest=longestPalindrome()
    result=longest.longestPalindromeFinder("abbabab")
    print(result)
            
