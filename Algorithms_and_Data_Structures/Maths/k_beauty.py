def input_values():
    num = int(input("Enter integer num: "))
    k = int(input("Enter substring length k: "))
    return num, k

def k_beauty(num: int, k: int) -> int:
    """Count length-k substrings of |num| that evenly divide num (ignoring 0)."""
    if k <= 0:
        raise ValueError("k must be a positive integer")
    num_str = str(abs(num))
    if k > len(num_str):
        return 0
    count = 0
    for i in range(len(num_str) - k + 1):
        sub_num = int(num_str[i : i + k])
        if sub_num != 0 and num % sub_num == 0:
            count += 1
    return count

if __name__ == "__main__":
    num, k = input_values()
    result = k_beauty(num, k)
    print(f"The k-beauty of {num} with k = {k} is: {result}\n")
