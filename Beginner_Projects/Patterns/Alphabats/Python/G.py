for row in range(5):
    for col in range(5):
        if(row == 4 and (col != 0 and col != 3)) or (row*col == 0) or (row == 3 and col > 2):
            print('*', end=" ")
        print(end=" ")
    print()
input()
