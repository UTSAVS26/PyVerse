def totalFruits(arr):
    f1, f2 = -1, -1
    c1, c2 = 0, 0
    maxC = 0
    n = len(arr)
    i = 0

    while i < n:
        if f1 == -1:
            f1 = arr[i]
        elif f2 == -1 and arr[i] != f1:
            f2 = arr[i]

        if arr[i] != f1 and arr[i] != f2:
            # Swap f1 with f2 and c1 with c2
            f1, f2 = f2, arr[i]
            c1, c2 = c2, 0

        if arr[i] == f1:
            c1 += 1
        elif arr[i] == f2:
            c2 += 1

        maxC = max(maxC, c1 + c2)
        i += 1

    return maxC
