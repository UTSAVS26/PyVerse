for row in range(5):
    for col in range(4):
        if col == 0 or (row+col == 3) or (col == row-1):
            print("*", end=" ")
        else:
            print(end="  ")
    print()
input()
