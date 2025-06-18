def insertion_sort(bucket):
    for i in range(1, len(bucket)):
        var = bucket[i]
        j = i - 1
        while j >= 0 and var < bucket[j]:
            bucket[j + 1] = bucket[j]
            j = j - 1
        bucket[j + 1] = var

def bucket_sort(input_list):
    # Create n empty buckets
    buckets = [[] for _ in range(len(input_list))]

    # Insert elements into their respective buckets
    for num in input_list:
        index = int(num * len(input_list))
        buckets[index].append(num)

    # Sort each bucket and concatenate the result
    sorted_list = []
    for bucket in buckets:
        insertion_sort(bucket)
        sorted_list.extend(bucket)
    return sorted_list

# Example usage
array = [0.42, 0.32, 0.33, 0.52, 0.37, 0.47, 0.51]
print("Sorted array is:", bucket_sort(array))
