def is_power_of(n, base):
    if n < 1 or base < 2:
        return False
    while n % base == 0:
        n //= base
    return n == 1

n = int(input("Enter number: "))
base = int(input("Enter base: "))
print(f"{n} is a power of {base}" if is_power_of(n, base) else f"{n} is not a power of {base}")
