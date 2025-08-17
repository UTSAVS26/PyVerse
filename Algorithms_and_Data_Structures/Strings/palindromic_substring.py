'''
display the number of unique palindromic substrings present in a string; substrings are contiguous part of a string and palindromic are those that read the same, backward or forward.
'''
class PalindromicSubstringCounter:
    def count_and_list_unique_palindromic_substrings(self, s: str):
        unique_palindromes = set()
        
        def expand_around_center(left: int, right: int):
            while left >= 0 and right < len(s) and s[left] == s[right]:
                unique_palindromes.add(s[left:right+1])
                left -= 1
                right += 1
        
        for i in range(len(s)):
            # Odd length palindromes
            expand_around_center(i, i)
            # Even length palindromes
            expand_around_center(i, i + 1)
        
        return len(unique_palindromes), list(unique_palindromes)


if __name__ == '__main__':
    counter = PalindromicSubstringCounter()
    
    user_input = input("Enter a string: ").strip()
    count, unique_substrings = counter.count_and_list_unique_palindromic_substrings(user_input)
    
    print("Number of unique palindromic substrings:", count)
    print("Unique palindromic substrings array:", unique_substrings)

     
