count=0 # the variable that counts the number of zeroes present
def counting_number_of_zeroes(value):#function to count the number of  zeroes
    global count
    if len(value)==0:#checks if the string is empty or not
        return
    if value[0]=='0':#checks if the value is zero
        count+=1#increments the zero count
        return counting_number_of_zeroes(value[1:])#calls it self again
    else:
        return counting_number_of_zeroes(value[1:])
#asks the user to enter the number
number = input("Enter the number where the number of zeroes need to be counted:")

if number.isnumeric():#checks whether the number is an integer or not
    counting_number_of_zeroes(number)#calls the function that counts the number
    print(f"Number of zeroes:{count}")#prints the count
else:
    print("The number entered is not a number try again")#prints if the string is not numeric in nature

