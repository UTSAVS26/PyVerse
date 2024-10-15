for row in range(5):
    for col in range(4):
        if(row*col==0) or ((row==2) and (col!=0 and col!=3)):
            print("*",end=" ")
    print()
input()