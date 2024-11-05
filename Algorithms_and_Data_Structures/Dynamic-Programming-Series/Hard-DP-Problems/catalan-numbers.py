def catalan_numbers(n):
    # Initialize a list to store Catalan numbers
    catalan = [0] * (n + 1)
    
    # Base case
    catalan[0] = 1
    
    # Calculate the remaining catalan numbers up to nth
    for i in range(1, n + 1):
        catalan[i] = sum(catalan[j] * catalan[i - 1 - j] for j in range(i))
    
    return catalan

# Test the function
n = 10  # Calculate the first 10 Catalan numbers
print(f"The first {n} Catalan numbers are: {catalan_numbers(n)}")
