def printMaxActivities(s, f):
    n = len(f)
    print("Following activities are selected:")

    i = 0
    print(i, end=' ')

    for j in range(1, n):
        if s[j] >= f[i]:
            print(j, end=' ')
            i = j

if __name__ == '__main__':
    n = int(input("Enter the number of activities: "))
    s = []
    f = []

    print("Enter start times of activities separated by spaces:")
    s = list(map(int, input().split()))
    
    print("Enter finish times of activities separated by spaces:")
    f = list(map(int, input().split()))

    printMaxActivities(s, f)
