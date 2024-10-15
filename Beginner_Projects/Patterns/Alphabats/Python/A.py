for rows in range(5):
    for cols in range(5):
        if((cols==0 or cols==4) and rows !=0 ) or ((rows==0 or rows==2) and (cols > 0 and cols < 4)):
            print("*",end=" ")
        else:
            print(end="  ")
    print()
input()