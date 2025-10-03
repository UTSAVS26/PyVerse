def is_happy(n):
    seen = set()
    while n != 1 and n not in seen:
        seen.add(n)
        n = sum(int(d)**2 for d in str(n)) 
    return n == 1   # if the number eventually becomes 1 by repeatedly summing the squares of its digits return true.

n = int(input("Enter a number: "))
print("Happy Number" if is_happy(n) else "Not a Happy Number")
