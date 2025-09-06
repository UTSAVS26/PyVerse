def input_value():
    n = int(input("Enter number of teams n: "))
    return n

def tournament_matches(n):
    total_matches = 0
    while n > 1:
        if n % 2 == 0:
            matches = n // 2
            n = matches
        else:
            matches = (n - 1) // 2
            n = matches + 1
        total_matches += matches
    return total_matches

if __name__ == "__main__":
    n = input_value()
    result = tournament_matches(n)
    print(f"Total matches played until a winner is decided: {result}\n")
