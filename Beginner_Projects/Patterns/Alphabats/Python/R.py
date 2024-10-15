for row in range(6):
    for col in range(8):
        if ((row * col == 0 and col < 5 and row + col != 0) or (col == 5 and row > 0 and row < 3) or (row == 3 and col < 5) or (row > 3 and col + 2 == row)):
            print("*", end=" ")
        else:
            print(end="  ")
    print()
input()
