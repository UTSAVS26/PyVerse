def sublist_search(main_list, sublist):
    n, m = len(main_list), len(sublist)

    for i in range(n - m + 1):
        if main_list[i:i + m] == sublist:
            return i
            
    return -1

# Example usage
main_list = [1, 2, 3, 4, 5, 6]
sublist = [3, 4]
print(sublist_search(main_list, sublist))
