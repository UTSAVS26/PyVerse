for row in range(4):
    for col in range(5):
        if col==0 or col*row==2 or col==4 or (col==3 and row==2):
            print("*", end=" ")
        else:
            print(end="  ")
    print()
input()
