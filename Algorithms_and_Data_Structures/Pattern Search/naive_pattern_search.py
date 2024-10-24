# naive_pattern_search.py

def naive_pattern_search(text, pattern):
    """
    Naive pattern search algorithm.
    This function searches for all occurrences of 'pattern' in 'text'
    by checking each position.

    Parameters:
    text (str): The text in which to search for the pattern.
    pattern (str): The pattern to search for.

    Prints the starting index of each occurrence of the pattern.
    """
    n = len(text)
    m = len(pattern)

    for i in range(n - m + 1):
        match = True
        for j in range(m):
            if text[i + j] != pattern[j]:
                match = False
                break
        if match:
            print(f"Pattern found at index {i}")

# Example usage
if __name__ == "__main__":
    naive_pattern_search("ababcabcab", "abc")
