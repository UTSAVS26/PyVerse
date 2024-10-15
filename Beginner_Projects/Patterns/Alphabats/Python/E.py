for row in range(5):
    for col in range(4):
        if(row*col==0) or (col!=0 and (row==2 or row==4)):
            print("*",end=" ")
        print(end="  ")
    print()
input()