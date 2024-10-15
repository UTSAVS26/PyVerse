for row in range(4):
    for col in range(5):
        if row<3 and (row+col==4 or row-col==0) or col==2 and row>2:
            print("*", end=" ")
        else:
            print(end="  ")
    print()
input()
