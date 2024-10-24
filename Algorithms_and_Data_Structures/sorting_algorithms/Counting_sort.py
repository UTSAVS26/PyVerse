def counting_sort(arr):
    max_val = max(arr) + 1
    count = [0] * max_val

    for num in arr:
        count[num] += 1

    sorted_arr = []
    for i, c in enumerate(count):
        sorted_arr.extend([i] * c)
    
    return sorted_arr
