'''
a magic square is an n*n grid full of numbers
such that every row, column, and diagonals
add up to the same number
'''

def generate_magic_square(n):
    # generates magic square using siamese method
    magic_square = [[0] * n for _ in range(n)]
    num = 1
    i, j = 0, n // 2

    while num <= n * n:
        magic_square[i][j] = num
        num += 1
        new_i, new_j = (i - 1) % n, (j + 1) % n
        if magic_square[new_i][new_j]:
            i = (i + 1) % n
        else:
            i, j = new_i, new_j

    return magic_square

def scale_magic_square(square, target_sum):
    # scales magic square to reach the target sum
    n = len(square)
    current_sum = sum(square[0])
    scale = target_sum / current_sum

    scaled_square = [
        [round(cell * scale) for cell in row] for row in square
    ]
    return scaled_square

def print_square(square):
    # prints the magic square
    for row in square:
        print(" ".join(f"{num:4}" for num in row))

if __name__ == "__main__":
    n = int(input("Enter the size of magic square: "))
    if n < 3 or n % 2 == 0:
        print("Size must be an odd number ≥ 3.")

    target_sum = int(input("Enter the target sum: "))
    base_square = generate_magic_square(n)
    scaled_square = scale_magic_square(base_square, target_sum)

    print_square(scaled_square)
    print(f"\nTarget sum is: {target_sum}")
