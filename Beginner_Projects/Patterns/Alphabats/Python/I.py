for rows in range(5):
    for cols in range(5):
        if rows == 0 or rows == 4 or cols == 2:
            print("*", end=" ")
        else:
            print(end="  ")
    print()
input()
