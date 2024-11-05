# Python code to perform 2x2 matrix multiplication using Strassen's method
import numpy as np

def main():
    x = np.zeros((2, 2), dtype=int)
    y = np.zeros((2, 2), dtype=int)
    z = np.zeros((2, 2), dtype=int)

    print("Enter the elements of the first matrix (2x2):")
    for i in range(2):
        for j in range(2):
            x[i][j] = int(input())

    print("Enter the elements of the second matrix (2x2):")
    for i in range(2):
        for j in range(2):
            y[i][j] = int(input())

    print("\nThe first matrix is:")
    for i in range(2):
        for j in range(2):
            print(f"{x[i][j]}\t", end="")
        print()

    print("\nThe second matrix is:")
    for i in range(2):
        for j in range(2):
            print(f"{y[i][j]}\t", end="")
        print()

    m1 = (x[0][0] + x[1][1]) * (y[0][0] + y[1][1])
    m2 = (x[1][0] + x[1][1]) * y[0][0]
    m3 = x[0][0] * (y[0][1] - y[1][1])
    m4 = x[1][1] * (y[1][0] - y[0][0])
    m5 = (x[0][0] + x[0][1]) * y[1][1]
    m6 = (x[1][0] - x[0][0]) * (y[0][0] + y[0][1])
    m7 = (x[0][1] - x[1][1]) * (y[1][0] + y[1][1])

    z[0][0] = m1 + m4 - m5 + m7
    z[0][1] = m3 + m5
    z[1][0] = m2 + m4
    z[1][1] = m1 - m2 + m3 + m6

    print("\nResultant matrix:")
    for i in range(2):
        for j in range(2):
            print(f"{z[i][j]}\t", end="")
        print()

if __name__ == "__main__":
    main()
