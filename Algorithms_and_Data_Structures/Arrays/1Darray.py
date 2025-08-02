max_size = 20

def create(arr, n):
    for i in range(n):
        arr.append(int(input("Enter: ")))

def remove(arr):
    if len(arr) > 0:
        arr.pop(0)
        print("\nAfter removing the first element:")
        print(arr)
    else:
        print("Array is empty!")

def remove_at(arr):
    x = int(input("\nEnter the element you want to delete: "))
    if x in arr:
        i = arr.index(x)
        print(f"\nElement {x} found at index {i}, after removing it:")
        arr.pop(i)
        print(arr)
    else:
        print("\nElement not found!")

def replace(arr):
    x1 = int(input("\nEnter an element you want to replace: "))
    if x1 in arr:
        i = arr.index(x1)
        print("\nElement is found! Now enter the replacing element:")
        x2 = int(input())
        arr[i] = x2
        print(arr)
    else:
        print("Element not found!")

def merge(a, b):
    c = []
    i = j = 0
    while i < len(a) and j < len(b):
        if a[i] < b[j]:
            c.append(a[i])
            i += 1
        else:
            c.append(b[j])
            j += 1
    while i < len(a):
        c.append(a[i])
        i += 1
    while j < len(b):
        c.append(b[j])
        j += 1
    print(c)


if __name__ == "__main__":
    a = []
    b = []
    n = int(input("Enter number of elements in array 1: "))
    create(a, n)
    print(a)

    # Uncomment these to test removal and replacement
    # remove(a)
    # remove_at(a)
    # replace(a)

    m = int(input("Enter number of elements in array 2: "))
    create(b, m)
    print(b)
    merge(a, b)
    
