
s = "  Hello World 123!   "

print("Original string:", repr(s)) # Original string: '  Hello World 123!   '

# 1. upper()
print("Uppercase:", s.upper()) # Uppercase:   HELLO WORLD 123!   

# 2. lower()
print("Lowercase:", s.lower()) # Lowercase:   hello world 123!   

# 3. capitalize()
print("Capitalize:", s.capitalize()) # Capitalize:   hello world 123!   

# 4. title()
print("Title case:", s.title()) # Title case:   Hello World 123!   

# 5. swapcase()
print("Swapcase:", s.swapcase()) # Swapcase:   hELLO wORLD 123!   

# 6. strip()
print("Strip (remove leading/trailing spaces):", repr(s.strip())) # Strip (remove leading/trailing spaces): 'Hello World 123!'

# 7. lstrip()
print("Lstrip (remove leading spaces):", repr(s.lstrip())) # Lstrip (remove leading spaces): 'Hello World 123!   '

# 8. rstrip()
print("Rstrip (remove trailing spaces):", repr(s.rstrip())) # Rstrip (remove trailing spaces): '  Hello World 123!'

# 9. replace()
print("Replace 'World' with 'Python':", s.replace("World", "Python")) # Replace 'World' with 'Python':   Hello Python 123!   

# 10. find()
print("Find 'World':", s.find("World")) # Find 'World': 7

# 11. rfind()
print("Find last occurrence of 'l':", s.rfind("l")) # Find last occurrence of 'l': 10

# 12. index()
print("Index of 'World':", s.index("World")) # Index of 'World': 7

# 13. rindex()
print("Index of last occurrence of 'l':", s.rindex("l")) # Index of last occurrence of 'l': 10

# 14. startswith()
print("Starts with 'Hello':", s.startswith("Hello")) # Starts with 'Hello': False

# 15. endswith()
print("Ends with '123!':", s.strip().endswith("123!")) # Ends with '123!': True

# 16. isalpha()
print("Is alphabetic (letters only):", s.isalpha()) # Is alphabetic (letters only): False

# 17. isdigit()
print("Is numeric (digits only):", s.isdigit()) # Is numeric (digits only): False

# 18. isalnum()
print("Is alphanumeric (letters and digits only):", s.isalnum()) # Is alphanumeric (letters and digits only): False

# 19. isspace()
whitespace_str = "   "
print("Is only whitespace:", whitespace_str.isspace()) # Is only whitespace: True

# 20. isupper()
print("Is uppercase:", s.isupper()) # Is uppercase: False

# 21. islower()
print("Is lowercase:", s.islower()) # Is lowercase: False
