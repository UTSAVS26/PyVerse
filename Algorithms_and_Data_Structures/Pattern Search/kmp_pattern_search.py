# kmp_pattern_search.py

def kmp_pattern_search(text, pattern):
    """
    Knuth-Morris-Pratt (KMP) algorithm for pattern searching.
    This function finds all occurrences of 'pattern' in 'text'
    using the KMP algorithm, which preprocesses the pattern for efficient searching.

    Parameters:
    text (str): The text in which to search for the pattern.
    pattern (str): The pattern to search for.

    Prints the starting index of each occurrence of the pattern.
    """
    def compute_lps(pattern):
        """
        Computes the Longest Prefix Suffix (LPS) array for the pattern.
        
        Parameters:
        pattern (str): The pattern to preprocess.

        Returns:
        list: The LPS array.
        """
        lps = [0] * len(pattern)
        length = 0
        i = 1
        while i < len(pattern):
            if pattern[i] == pattern[length]:
                length += 1
                lps[i] = length
                i += 1
            else:
                if length != 0:
                    length = lps[length - 1]
                else:
                    lps[i] = 0
                    i += 1
        return lps

    lps = compute_lps(pattern)
    i = j = 0  # Index for text and pattern
    while i < len(text):
        if text[i] == pattern[j]:
            i += 1
            j += 1

        if j == len(pattern):
            print(f"Pattern found at index {i - j}")
            j = lps[j - 1]
        elif i < len(text) and text[i] != pattern[j]:
            if j != 0:
                j = lps[j - 1]
            else:
                i += 1

# Example usage
if __name__ == "__main__":
    kmp_pattern_search("ababcabcab", "abc")
