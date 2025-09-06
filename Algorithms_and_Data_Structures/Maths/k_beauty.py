def input_values():
    num = int(input("Enter integer num: "))
    k = int(input("Enter substring length k: "))
    return num, k

def k_beauty(num, k):
    num_str = str(num)
    count = 0
    for i in range(len(num_str) - k + 1):
        sub_str = num_str[i:i+k]
        sub_num = int(sub_str)
        if sub_num != 0 and num % sub_num == 0:
            count += 1
    return count

if __name__ == "__main__":
    num, k = input_values()
    result = k_beauty(num, k)
    print(f"The k-beauty of {num} with k = {k} is: {result}\n")
