for row in range(4):
    for col in range(5):
        if row==0 or row+col==4 or row==3:
            print("*", end=" ")
        else:
            print(end="  ")
    print()
input()
