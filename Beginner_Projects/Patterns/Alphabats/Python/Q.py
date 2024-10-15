for row in range(6):
    for col in range(8):
        if (((row * col == 0 and (col + row != 0) and col < 6 and row < 4)) or (col == 6 and row != 0 and row <4) or (row == 4 and col != 0 and col!=7) or (row == 3 and col == 5)) or (row==5 and col==7):
            print("*", end=" ")
        else:
            print(end="  ")
    print()
input()
