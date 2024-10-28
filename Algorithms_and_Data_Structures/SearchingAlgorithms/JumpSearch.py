import math

def jumpSearch( arr , x , n ):
    # Finding block size 
    step = math.sqrt(n)
    
    # Finding the block where the element is present (if it is present)
    prev = 0
    while arr[int(min(step, n) - 1)] < x:
        prev = step
        step += math.sqrt(n)
        if prev >= n:  # If we go beyond array bounds
            return -1
    
    # Linear search within the identified block
    while arr[int(prev)] < x:
        prev += 1
        if prev == min(step, n):  # If we reach the next block or end of the array
            return -1
    
    # If element is found
    if arr[int(prev)] == x:
        return prev
    
    return -1

# Driver code to test the function
arr = [0, 1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 610]
x = 55
n = len(arr)

# Find the index of 'x' using Jump Search
index = jumpSearch(arr, x, n)

# Print the index where 'x' locate
print("Number", x, "is at index", "%.0f" % index)
