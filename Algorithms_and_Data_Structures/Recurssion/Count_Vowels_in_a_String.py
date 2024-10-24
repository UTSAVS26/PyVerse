def count_vowels(s):
    if len(s) == 0:
        return 0
    vowels = "aeiouAEIOU"
    return (1 if s[0] in vowels else 0) + count_vowels(s[1:])
