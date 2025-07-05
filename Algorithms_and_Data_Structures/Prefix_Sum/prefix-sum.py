
# Prefix sum array
def build_prefix_sum(arr):
    prefix_sum = [0] * (len(arr) + 1) # length is n+1 to accommodate the sum up to index 0
    for i in range(1, len(arr) + 1):
        prefix_sum[i] = prefix_sum[i - 1] + arr[i - 1]
    return prefix_sum

def range_sum(prefix_sum, left, right):
    if left < 0 or right >= len(prefix_sum) - 1 or left > right:
        raise ValueError("Invalid range")
    return prefix_sum[right + 1] - prefix_sum[left]

def main():
    # Example usage
    a = [10, 2, 3, 4, 14, 15, 2, 22]
    print("Input Array:", a)
    
    prefix_sum = build_prefix_sum(a)
    print("Prefix Sum Array:", prefix_sum)

    left = 2
    right = 5
    try:
        result = range_sum(prefix_sum, left, right)
        print(f"Sum from index {left} to {right} is: {result}")
    except ValueError as e:
        print(e)

if __name__ == "__main__":
    main()
