for row in range(5):
    for col in range(5):
        if(row*col==0 and col!=4) or ((row==2 or row==4) and col!=4) or ((row==1 or row==3) and col==4):
            print('*',end=' ')
        else:
            print(end="  ")
    print()
input()