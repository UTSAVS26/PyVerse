for row in range(4):
    for col in range(4):
        if(row*col==0 and col!=3) or (row==3 and col!=3) or (col==3 and (row>0 and row<3)):
            print("*",end=" ")
        else:
            print(end="  ")
    print()
input()