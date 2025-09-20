def is_perfect(n):
    if n < 2:
        return False
    return sum(i for i in range(1, n) if n % i == 0) == n

n = int(input("Enter a number: "))
print("Perfect Number" if is_perfect(n) else "Not a Perfect Number")
