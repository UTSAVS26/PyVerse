from turtle import end_fill


for row in range(4):
    for col in range(4):
        if(col==0 and (row!=0 and row!=3)) or (col!=0 and (row==0 or row==3)):
            print('*',end=' ')
        else:
            print(end="  ")
    print()
input()