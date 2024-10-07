# suffix_array.py

def build_suffix_array(s):
    """
    Builds the suffix array for the given string.

    Parameters:
    s (str): The input string.

    Returns:
    list: The suffix array.
    """
    suffixes = sorted([(s[i:], i) for i in range(len(s))])
    return [suffix[1] for suffix in suffixes]

def kasai_lcp_array(s, suffix_array):
    """
    Constructs the LCP (Longest Common Prefix) array.

    Parameters:
    s (str): The input string.
    suffix_array (list): The suffix array.

    Returns:
    list: The LCP array.
    """
    n = len(s)
    rank = [0] * n
    lcp = [0] * n

    for i, suffix_index in enumerate(suffix_array):
        rank[suffix_index] = i

    h = 0
    for i in range(n):
        if rank[i] > 0:
            j = suffix_array[rank[i] - 1]
            while (i + h < n) and (j + h < n) and (s[i + h] == s[j + h]):
                h += 1
            lcp[rank[i]] = h
            if h > 0:
                h -= 1
    return lcp

# Example usage
if __name__ == "__main__":
    text = "banana"
    suffix_array = build_suffix_array(text)
    lcp = kasai_lcp_array(text, suffix_array)

    print("Suffix Array:", suffix_array)
    print("LCP Array:", lcp)
