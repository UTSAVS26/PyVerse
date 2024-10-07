# rabin_karp.py

def rabin_karp(text, pattern):
    """
    Rabin-Karp algorithm for pattern searching.
    This function finds all occurrences of 'pattern' in 'text'
    using a hashing technique.

    Parameters:
    text (str): The text in which to search for the pattern.
    pattern (str): The pattern to search for.

    Prints the starting index of each occurrence of the pattern.
    """
    d = 256  # Number of characters in the input alphabet
    q = 101  # A prime number for hashing
    m = len(pattern)
    n = len(text)
    p = 0  # Hash value for pattern
    t = 0  # Hash value for text
    h = 1

    # Calculate the value of h
    for i in range(m - 1):
        h = (h * d) % q

    # Calculate the initial hash values for pattern and text
    for i in range(m):
        p = (d * p + ord(pattern[i])) % q
        t = (d * t + ord(text[i])) % q

    # Slide the pattern over text one by one
    for i in range(n - m + 1):
        if p == t:  # Check for a match
            if text[i:i + m] == pattern:
                print(f"Pattern found at index {i}")

        if i < n - m:
            t = (d * (t - ord(text[i]) * h) + ord(text[i + m])) % q
            # We might get negative value of t, converting it to positive
            if t < 0:
                t += q

# Example usage
if __name__ == "__main__":
    rabin_karp("ababcabcab", "abc")
