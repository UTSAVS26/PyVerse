def kadane_algorithm(arr):
    max_current = arr[0]
    max_global = arr[0]
    
    for i in range(1, len(arr)):
        max_current = max(arr[i], max_current + arr[i])
        
        if max_current > max_global:
            max_global = max_current
    
    return max_global

def run_test_cases():
    t = int(input("Enter number of test cases: "))
    
    for _ in range(t):
        n = int(input("Enter size of array: "))
        
        arr = list(map(int, input("Enter the elements of the array: ").split()))
        
        result = kadane_algorithm(arr)
        print("Maximum sum of contiguous subarray:", result)

run_test_cases()
