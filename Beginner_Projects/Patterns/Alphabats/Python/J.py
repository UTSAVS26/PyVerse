for row in range(7):
    for col in range(7):
        if(row == 0 or (col == 5 and (row < 5 and row != 0)) or (col == 4 and row == 5) or ((col >= 1 and col < 4) and row == 6) or ((row == 5 or row == 4) and col == 0)):
            print("*", end=" ")
        else:
            print(end="  ")
    print()
