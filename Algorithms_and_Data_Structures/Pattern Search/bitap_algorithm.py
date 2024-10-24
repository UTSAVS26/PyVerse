# bitap_algorithm.py

def bitap_search(text, pattern):
    """
    Bitap algorithm (also known as Shift-Or algorithm) for pattern searching.
    This function finds all occurrences of 'pattern' in 'text' using bitwise operations.

    Parameters:
    text (str): The text in which to search for the pattern.
    pattern (str): The pattern to search for.

    Prints the starting index of each occurrence of the pattern.
    """
    m = len(pattern)
    if m == 0:
        return
    all_ones = (1 << len(text)) - 1
    R = [0] * (m + 1)
    for i in range(m):
        R[i] = all_ones << i

    for i in range(len(text)):
        for j in range(m):
            if text[i] == pattern[m - 1 - j]:
                R[j] = R[j] | (1 << i)
            else:
                R[j] = R[j] & ~(1 << i)
        if R[m - 1] & (1 << i):
            print(f"Pattern found at index {i - m + 1}")

# Example usage
if __name__ == "__main__":
    bitap_search("abcabcabc", "abc")
