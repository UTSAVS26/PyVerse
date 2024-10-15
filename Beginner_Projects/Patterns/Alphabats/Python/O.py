for row in range(5):
    for col in range(6):
        if (((row * col == 0 and (col + row != 0) and col != 5 and row != 4)) or (col == 5 and row != 0 and row != 4) or (row == 4 and col != 0 and col != 5)):

            print("*", end=" ")
        else:
            print(end="  ")
    print()
input()
