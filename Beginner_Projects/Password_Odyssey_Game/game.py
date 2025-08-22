import string
import re 

def generate_conditions():
    # Precompute functions for conditions
    def no_run(regex):
        return lambda p: re.search(regex, p) is None

    def exactly_once(ch):
        return lambda p: p.count(ch) == 1

    def contains_exact_once(substr):
        return lambda p: p.count(substr) == 1

    def starts_with_consonant_lower():
        return lambda p: (
            len(p) > 0
            and p[0].isalpha()
            and p[0].islower()
            and p[0].lower() in "bcdfghjklmnpqrstvwxyz"
        )

    conditions = [
        # LENGTH / BASIC STRUCTURE (1–9)
        (lambda p: len(p) >= 32, "Password must be at least 32 characters long"),
        (lambda p: len(p) <= 64, "Password must be at most 64 characters long"),
        (lambda p: len(p) % 2 == 0, "Length must be divisible by 2"),
        (lambda p: len([c for c in p if c.isalpha()]) >= 15, "Must contain at least 15 letters"),
        (lambda p: len([c for c in p if c.isupper()]) >= 3, "Must contain at least 3 uppercase letters"),
        (lambda p: len([c for c in p if c.islower()]) >= 10, "Must contain at least 10 lowercase letters"),
        (lambda p: len(set(p)) >= 20, "Must contain at least 20 unique characters"),
        (lambda p: len([c for c in p if c.isdigit()]) >= 8, "Must contain at least 8 digits"),
        (lambda p: len([c for c in p if c in string.punctuation]) == 4, "Must contain exactly 4 special characters"),

        # POSITIONAL REQUIREMENTS (10–19)
        (starts_with_consonant_lower(), "Must start with a lowercase consonant"),
        (lambda p: p[1].islower() if len(p) >= 2 else False, "Second character must be lowercase"),
        (lambda p: p[2].isalpha() if len(p) >= 3 else False, "Third character must be a letter"),
        (lambda p: p[3].isdigit() if len(p) >= 4 else False, "Fourth character must be a digit"),
        (lambda p: p[4].islower() if len(p) >= 5 else False, "Fifth character must be lowercase"),
        (lambda p: p[5].isupper() if len(p) >= 6 else False, "Sixth character must be uppercase"),
        (lambda p: p[10].islower() if len(p) >= 11 else False, "11th character must be lowercase"),
        (lambda p: p[15].isalpha() if len(p) >= 16 else False, "16th character must be a letter"),
        (lambda p: p[-2].isalpha() if len(p) >= 2 else False, "Second-to-last character must be a letter"),
        (lambda p: p[-1].isdigit(), "Must end with a digit"),

        # DIGIT COMPOSITION (20–29)
        (lambda p: p.count('1') == 3, "Must contain exactly three '1' digits"),
        (lambda p: p.count('0') == 2, "Must contain exactly two '0' digits"),
        (lambda p: p.count('3') == 1, "Must contain exactly one '3' digit"),
        (lambda p: p.count('5') == 1, "Must contain exactly one '5' digit"),
        (lambda p: p.count('9') == 2, "Must contain exactly two '9' digits"),
        (lambda p: p.count('2') == 1, "Must contain exactly one '2' digit"),
        (lambda p: p.count('4') == 1, "Must contain exactly one '4' digit"),
        (lambda p: p.count('7') == 0, "Must not contain the digit '7'"),
        (lambda p: p.count('8') == 0, "Must not contain the digit '8'"),
        (lambda p: p.count('6') == 0, "Must not contain the digit '6'"),

        # SPECIAL CHARACTERS (30–39)
        (exactly_once('#'), "Must contain exactly one '#' symbol"),
        (exactly_once('!'), "Must contain exactly one '!' symbol"),
        (exactly_once('*'), "Must contain exactly one '*' symbol"),
        (exactly_once('$'), "Must contain exactly one '$' symbol"),
        (lambda p: all(c in {'#', '!', '*', '$'} for c in [c for c in p if c in string.punctuation]), "Only '#', '!', '*' and '$' are allowed as special characters"),
        (lambda p: p.count('%') == 0, "Must not contain the '%' symbol"),
        (lambda p: p.count('@') == 0, "Must not contain the '@' symbol"),
        (lambda p: p.count('&') == 0, "Must not contain the '&' symbol"),
        (lambda p: p.count('^') == 0, "Must not contain the '^' symbol"),
        (lambda p: p.count('~') == 0, "Must not contain the '~' symbol"),

        # REQUIRED LETTERS (40–59)
        (lambda p: 'a' in p.lower(), "Must contain the letter 'a'"),
        (lambda p: 'b' in p.lower(), "Must contain the letter 'b'"),
        (lambda p: 'c' in p.lower(), "Must contain the letter 'c'"),
        (lambda p: 'd' in p.lower(), "Must contain the letter 'd'"),
        (lambda p: 'f' in p.lower(), "Must contain the letter 'f'"),
        (lambda p: 'g' in p.lower(), "Must contain the letter 'g'"),
        (lambda p: 'h' in p.lower(), "Must contain the letter 'h'"),
        (lambda p: 'i' in p.lower(), "Must contain the letter 'i'"),
        (lambda p: 'j' in p.lower(), "Must contain the letter 'j'"),
        (lambda p: 'k' in p.lower(), "Must contain the letter 'k'"),
        (lambda p: 'l' in p.lower(), "Must contain the letter 'l'"),
        (lambda p: 'm' in p.lower(), "Must contain the letter 'm'"),
        (lambda p: 'n' in p.lower(), "Must contain the letter 'n'"),
        (lambda p: 'o' in p.lower(), "Must contain the letter 'o'"),
        (lambda p: 'p' in p.lower(), "Must contain the letter 'p'"),
        (lambda p: 'r' in p.lower(), "Must contain the letter 'r'"),
        (lambda p: 's' in p.lower(), "Must contain the letter 's'"),
        (lambda p: 't' in p.lower(), "Must contain the letter 't'"),
        (lambda p: 'u' in p.lower(), "Must contain the letter 'u'"),
        (lambda p: 'w' in p.lower(), "Must contain the letter 'w'"),

        # FORBIDDEN LETTERS (60–64)
        (lambda p: 'q' not in p.lower(), "Must not contain the letter 'q'"),
        (lambda p: 'v' not in p.lower(), "Must not contain the letter 'v'"),
        (lambda p: 'x' not in p.lower(), "Must not contain the letter 'x'"),
        (lambda p: 'y' not in p.lower(), "Must not contain the letter 'y'"),
        (lambda p: 'z' not in p.lower(), "Must not contain the letter 'z'"),

        # VOWEL RULES (65–69)
        (lambda p: len([c for c in p if c.lower() in 'aeiou']) >= 4, "Must contain at least 4 vowels total"),
        (lambda p: len([c for c in p if c.lower() in 'aeiou']) <= 7, "Must contain at most 7 vowels total"),
        (lambda p: p.count('e') >= 1, "Must contain at least one 'e'"),
        (lambda p: p.count('e') <= 2, "Must contain at most two 'e' letters"),
        (lambda p: len(set(c.lower() for c in p if c.lower() in 'aeiou')) >= 3, "Must contain at least 3 different vowel types"),

        # SUBSTRING REQUIREMENTS (70–82)
        (contains_exact_once("SUN"), "Must contain 'SUN' exactly once"),
        (contains_exact_once("101"), "Must contain '101' exactly once"),
        (lambda p: "love" not in p.lower(), "Must not contain the word 'love'"),
        (lambda p: "pass" not in p.lower(), "Must not contain the word 'pass'"),
        (lambda p: "word" not in p.lower(), "Must not contain the word 'word'"),
        (lambda p: "2024" not in p, "Must not contain '2024'"),
        (lambda p: "2025" not in p, "Must not contain '2025'"),
        (lambda p: "qwerty" not in p.lower(), "Must not contain 'qwerty' sequence"),
        (lambda p: "asdfgh" not in p.lower(), "Must not contain 'asdfgh' sequence"),
        (lambda p: "zxcvbn" not in p.lower(), "Must not contain 'zxcvbn' sequence"),
        (lambda p: "123" not in p, "Must not contain '123' sequence"),
        (lambda p: "098" not in p, "Must not contain '098' sequence"),
        (lambda p: p != p[::-1], "Must not be a palindrome"),

        # PATTERN RESTRICTIONS (83–91)
        (no_run(r"[0-9]{4}"), "No four consecutive digits"),
        (no_run(r"[A-Z]{7}"), "No four consecutive uppercase letters"),
        (no_run(r"[aeiouAEIOU]{6}"), "No three consecutive vowels"),
        (no_run(r"[!#*$]{2}"), "No two consecutive special characters"),
        (no_run(r"(.)\1{2}"), "No character can appear three times in a row"),
        (no_run(r"[!#*$][0-9]"), "No special character immediately followed by a digit"),
        (no_run(r"[0-9][!#*$]"), "No digit immediately followed by a special character"),
        (no_run(r"[A-Z][0-9][A-Z]"), "No uppercase-digit-uppercase sequence"),
        (no_run(r"\s"), "Must not contain spaces or whitespace"),

        # CHARACTER FREQUENCY LIMITS (92–95)
        (lambda p: p.count('a') <= 2, "Letter 'a' can appear at most twice"),
        (lambda p: p.count('o') <= 2, "Letter 'o' can appear at most twice"),
        (lambda p: p.count('i') <= 2, "Letter 'i' can appear at most twice"),
        (lambda p: p.count('u') <= 2, "Letter 'u' can appear at most twice"),

        # MATHEMATICAL CONDITIONS (96–100)
        (lambda p: sum(ord(c) for c in p if c.isalpha()) % 2 == 0, "Sum of letter ASCII values must be even"),
        (lambda p: len([c for c in p if c.isupper()]) % 2 == 1, "Number of uppercase letters must be odd"),
        (lambda p: len([c for c in p if c.islower()]) % 2 == 0, "Number of lowercase letters must be even"),
        (lambda p: sum(int(c) for c in p if c.isdigit()) % 5 == 0, "Sum of all digits must be divisible by 5"),
        (lambda p: 'SUN' in p and '101' in p and p.index('SUN') < p.index('101'), "Must contain both 'SUN' and '101', with 'SUN' appearing before '101'"),
    ]
    
    return conditions

def check_password(password, conditions):
    failed = [msg for cond, msg in conditions if not cond(password)]
    passed_count = 100 - len(failed)
    return len(failed) == 0, failed, passed_count

def password_game():
    print("Welcome to the Password Challenge!")
    print("Your password must satisfy 100 unique conditions.")
    print("Enter a password, and I'll tell you which conditions you failed.")
    print("Type 'quit' to exit.")
    
    conditions = generate_conditions()
    
    while True:
        print("\n" + "="*50)
        password = input("\nEnter your password: ")
        if password.lower() == 'quit':
            print("Game ended.")
            break
        
        is_valid, failed_conditions, passed_count = check_password(password, conditions)
        if is_valid:
            print()
            print(f"SUCCESS !")
            print(f"Your password meets all requirements! (All {passed_count}/100 checks passed)")
            print("\n" + "="*50 + "\n")
            break
        else:
            print()
            print(f"Progress: {passed_count}/100 checks passed.")
            for i, failure in enumerate(failed_conditions):
                print(f"NOTE: {failure}")
                break
            
# baf1gSUNhiejklmopr5stu$w9c#w!d101b*w934ww0w2

if __name__ == "__main__":
    print("\n" + "="*50 + "\n")
    password_game()