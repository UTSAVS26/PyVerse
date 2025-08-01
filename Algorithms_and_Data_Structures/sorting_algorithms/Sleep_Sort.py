import threading
import time
def sleep_sort_with_negatives(numbers):
    result = []

    # Shift values if there are negative numbers
    min_val = min(numbers)
    shift = -min_val if min_val < 0 else 0

    def sleeper(n):
        time.sleep((n + shift) * 0.01)  # Sleep after shifting
        result.append(n)

    threads = []
    for number in numbers:
        t = threading.Thread(target=sleeper, args=(number,))
        t.start()
        threads.append(t)

    for t in threads:
        t.join()

    return result

# Example usage
numbers = [3, -2, 1, 0, -1, 2]
sorted_numbers = sleep_sort_with_negatives(numbers)
print("Sorted:", sorted_numbers)
