def printMaxActivities(s, f):
    n = len(f)  # Get the number of activities
    print("Following activities are selected:")

    i = 0  # The first activity is always selected
    print(i, end=' ')

    # Loop through the remaining activities
    for j in range(1, n):
        # If the start time of the current activity is greater or equal to 
        # the finish time of the last selected activity
        if s[j] >= f[i]:
            print(j, end=' ')  # Select this activity
            i = j  # Update the last selected activity index

if __name__ == '__main__':
    n = int(input("Enter the number of activities: "))  # Input the number of activities
    s = []
    f = []

    # Input the start times of the activities
    print("Enter start times of activities separated by spaces:")
    s = list(map(int, input().split()))
    
    # Input the finish times of the activities
    print("Enter finish times of activities separated by spaces:")
    f = list(map(int, input().split()))

    # Call the function to print the maximum set of activities
    printMaxActivities(s, f)
