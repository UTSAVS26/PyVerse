for row in range(4):
    for col in range(7):
        if col-row==0 or col+row==6:
            print("*", end=" ")
        else:
            print(end="  ")
    print()
input()
