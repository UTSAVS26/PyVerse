# boyer_moore.py

def bad_character_heuristic(pattern):
    """
    Preprocesses the pattern to create the bad character table.
    
    Parameters:
    pattern (str): The pattern to preprocess.

    Returns:
    dict: A dictionary mapping characters to their last occurrence index.
    """
    bad_char = {}
    for i in range(len(pattern)):
        bad_char[pattern[i]] = i
    return bad_char

def boyer_moore(text, pattern):
    """
    Boyer-Moore algorithm for pattern searching.
    This function finds all occurrences of 'pattern' in 'text'
    using the Boyer-Moore algorithm, which skips sections of the text.

    Parameters:
    text (str): The text in which to search for the pattern.
    pattern (str): The pattern to search for.

    Prints the starting index of each occurrence of the pattern.
    """
    bad_char = bad_character_heuristic(pattern)
    m = len(pattern)
    n = len(text)
    s = 0  # Shift of the pattern with respect to text

    while s <= n - m:
        j = m - 1

        while j >= 0 and pattern[j] == text[s + j]:
            j -= 1

        if j < 0:
            print(f"Pattern found at index {s}")
            s += (m - bad_char.get(text[s + m], -1)) if s + m < n else 1
        else:
            s += max(1, j - bad_char.get(text[s + j], -1))

# Example usage
if __name__ == "__main__":
    boyer_moore("ababcabcab", "abc")
