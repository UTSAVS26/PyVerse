"""
Word Break II
Given a string s and a dictionary of words, return all possible sentences where s is segmented into a sequence of dictionary words.
"""

class Solution:
    def wordBreak(self, s, wordDict):
        word_set = set(wordDict)  # convert wordDict to a set for faster lookup
        memo = {}  # use a dictionary to store results for each start index
        
        # helper function to return all sentences starting from index start
        def backtrack(start):
            if start == len(s):  # if we've reached the end of the string
                return ['']  # return a list with an empty sentence
            if start in memo:  # if we've already solved this subproblem
                return memo[start]  # return the stored result
            sentences = []  # list to store all possible sentences
            for end in range(start + 1, len(s) + 1):  # try every possible end index
                word = s[start:end]  # get the current word
                if word in word_set:  # if the word is in the dictionary
                    # get all sentences for the rest of the string
                    for sub in backtrack(end):
                        if sub:
                            sentences.append(word + ' ' + sub)  # add the word and the rest
                        else:
                            sentences.append(word)  # just add the word
            memo[start] = sentences  # store the result for this start index
            return sentences  # return all possible sentences
        
        return backtrack(0)  # start from index 0 