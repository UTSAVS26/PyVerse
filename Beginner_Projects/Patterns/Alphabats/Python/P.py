for row in range(7):
    for col in range(6):
        if ((row * col == 0 and col != 5) or (col == 5 and (row > 0 and row < 3) or (row == 3 and col < 5))):
            print("*", end=" ")
        else:
            print(end="  ")
    print()
input()
