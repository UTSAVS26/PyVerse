def input_matrix(mat, m, n):
    print(f"\nEnter elements for a {m} x {n} matrix (row-wise, sorted with each row's first element > previous row's last):")
    for i in range(m):
        row = []
        for j in range(n):
            val = int(input(f"Enter element at row {i}, column {j}: "))
            row.append(val)
        mat.append(row)

def display_matrix(mat, m, n):
    print("\nMatrix is:")
    for i in range(m):
        for j in range(n):
            print(f"{mat[i][j]}\t", end="")
        print()

def search_matrix(mat, m, n, target):
    left = 0
    right = m * n - 1

    while left <= right:
        mid = (left + right) // 2
        row = mid // n
        col = mid % n
        mid_val = mat[row][col]

        if mid_val == target:
            return True
        elif mid_val < target:
            left = mid + 1
        else:
            right = mid - 1

    return False

if __name__ == "__main__":
    matrix = []
    m = int(input("Enter number of rows: "))
    n = int(input("Enter number of columns: "))
    input_matrix(matrix, m, n)
    display_matrix(matrix, m, n)

    target = int(input("\nEnter the target element to search for: "))
    found = search_matrix(matrix, m, n, target)

    if found:
        print(f"\nTarget {target} found in the matrix.")
    else:
        print(f"\nTarget {target} not found in the matrix.")
