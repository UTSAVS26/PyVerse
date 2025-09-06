def input_value():
    n = int(input("Enter 3-digit integer n: "))
    return n

def is_fascinating(n):
    if n < 100 or n > 999:
        return False  # Ensure 3-digit input
    concat_str = str(n) + str(2*n) + str(3*n)
    if '0' in concat_str:
        return False
    digits_set = set(concat_str)
    return len(concat_str) == 9 and digits_set == set("123456789")

if __name__ == "__main__":
    n = input_value()
    result = is_fascinating(n)
    print(f"Is {n} fascinating? {result}\n")
