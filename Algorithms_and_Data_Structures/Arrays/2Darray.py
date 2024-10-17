def create(a, m, n):
    for i in range(m):
        for j in range(n):
            a[i][j] = int(input(f"\nEnter no. in row={i} column={j}: "))

def display(a, m, n):
    print("\nArray is:")
    for i in range(m):
        print("\n", end="")
        for j in range(n):
            print(f"{a[i][j]}\t", end="")
    print()

def add(a, b, m, n):
    c = [[0] * 10 for _ in range(10)]
    for i in range(m):
        for j in range(n):
            c[i][j] = a[i][j] + b[i][j]
    print("\nAddition of these:")
    display(c, m, n)

def sub(a, b, m, n):
    c = [[0] * 10 for _ in range(10)]
    for i in range(m):
        for j in range(n):
            c[i][j] = a[i][j] - b[i][j]
    print("\nSubtraction of these:")
    display(c, m, n)

def sparse_3tuple(a, c, m, n):
    k = 1
    for i in range(m):
        for j in range(n):
            if a[i][j] != 0:
                c[k][0] = i
                c[k][1] = j
                c[k][2] = a[i][j]
                k += 1
    c[0][0] = m
    c[0][1] = n
    c[0][2] = k - 1
    print("\nSparse to 3-tuple conversion:")
    print(f"{c[0][0]}\t{c[0][1]}\t{c[0][2]}")
    for t in range(1, k):
        print(f"{c[t][0]}\t{c[t][1]}\t{c[t][2]}")

def createsparse(b, m):
    print("\nEnter number of rows and columns:")
    b[0][0], b[0][1] = map(int, input().split())
    b[0][2] = m
    print("\nEnter row no., column no., and value at that place:")
    for x in range(1, m + 1):
        b[x] = list(map(int, input().split()))

def display_sparse(b, m):
    print("\nSparse matrix representation is:")
    for i in range(m + 1):
        print(f"{b[i][0]}\t{b[i][1]}\t{b[i][2]}")

def tuple_sparse(a, b, m):
    k = 1
    for i in range(b[0][0]):
        for j in range(b[0][1]):
            if k <= m and b[k][0] == i and b[k][1] == j:
                a[i][j] = b[k][2]
                k += 1
            else:
                a[i][j] = 0
    print("\nConversion is:")
    for i in range(b[0][0]):
        print("\n", end="")
        for j in range(b[0][1]):
            print(f"{a[i][j]}\t", end="")
    print()


if __name__ == "__main__":
    a = [[0] * 10 for _ in range(10)]
    b = [[0] * 3 for _ in range(10)]
    m, n = 0, 0
    
    # Uncomment the following lines to use creation, addition, and subtraction
    # m = int(input("Enter number of rows and columns: "))
    # create(a, m, n)
    # print("\nSparse matrix:")
    # display(a, m, n)
    # print("\nFor 2nd array:")
    # create(b, m, n)
    # display(b, m, n)
    # add(a, b, m, n)
    # sub(a, b, m, n)
    
    print("Creating 3-tuple or sparse matrix representation.")
    m = int(input("Enter number of non-zero elements: "))
    createsparse(b, m)
    display_sparse(b, m)
    tuple_sparse(a, b, m)
