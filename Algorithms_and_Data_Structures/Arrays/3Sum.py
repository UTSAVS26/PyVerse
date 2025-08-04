def input_array(arr, n):
    for i in range(n):
        num = int(input(f"Enter element {i + 1}: "))
        arr.append(num)

def display_array(arr):
    print("\nInput array is:")
    for x in arr:
        print(f"{x}\t", end="")
    print()

def three_sum(arr, n, triplets):
    arr.sort()  # Sort the array for two-pointer technique
    k = 0  # Index for storing triplets

    for i in range(n):
        if i > 0 and arr[i] == arr[i - 1]:
            continue  # Skip duplicate elements for i

        left = i + 1
        right = n - 1

        while left < right:
            total = arr[i] + arr[left] + arr[right]

            if total < 0:
                left += 1
            elif total > 0:
                right -= 1
            else:
                # Found a valid triplet
                triplets[k] = [arr[i], arr[left], arr[right]]
                k += 1

                # Skip duplicates
                while left < right and arr[left] == triplets[k - 1][1]:
                    left += 1
                while left < right and arr[right] == triplets[k - 1][2]:
                    right -= 1

                left += 1
                right -= 1

    return k  # Number of valid triplets

def display_triplets(triplets, count):
    if count == 0:
        print("\nNo triplets found that sum to 0.")
    else:
        print(f"\nTriplets that sum to 0 (total {count}):")
        for i in range(count):
            print(f"{triplets[i][0]}\t{triplets[i][1]}\t{triplets[i][2]}")

if __name__ == "__main__":
    arr = []
    triplets = [[0] * 3 for _ in range(100)]  # You can increase size if needed

    n = int(input("Enter the number of elements in array: "))
    input_array(arr, n)
    display_array(arr)

    count = three_sum(arr, n, triplets)
    display_triplets(triplets, count)

