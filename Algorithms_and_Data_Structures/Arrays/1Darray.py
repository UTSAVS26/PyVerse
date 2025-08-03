max_size = 20

def create(arr, n):
    print("\nEnter elements:")
    for i in range(n):
        while True:
            try:
                arr.append(int(input(f"Element {i + 1}: ")))
                break
            except ValueError:
                print("Invalid input! Please enter an integer. Try again.")

def remove(arr):
    if arr:
        arr.pop(0)
        print("\nAfter removing the first element:", arr)
    else:
        print("Array is empty!")

def remove_at(arr):
    try:
        x = int(input("\nEnter the element you want to delete: "))
        if x in arr:
            i = arr.index(x)
            arr.pop(i)
            print(f"Removed {x}. New array:", arr)
        else:
            print("Element not found!")
    except ValueError:
        print("Invalid input!")

def replace(arr):
    try:
        x1 = int(input("\nEnter element to replace: "))
        if x1 in arr:
            i = arr.index(x1)
            x2 = int(input("Enter new element: "))
            arr[i] = x2
            print("Updated array:", arr)
        else:
            print("Element not found!")
    except ValueError:
        print("Invalid input!")

def merge(a, b):
    a.sort()
    b.sort()
    c = []
    i = j = 0
    while i < len(a) and j < len(b):
        if a[i] < b[j]:
            c.append(a[i])
            i += 1
        else:
            c.append(b[j])
            j += 1
    c.extend(a[i:])
    c.extend(b[j:])
    print("Merged array:", c)

if __name__ == "__main__":
    a, b = [], []

    while True:
        print("\n---- Menu ----")
        print("1. Create Array A")
        print("2. Remove first element from A")
        print("3. Remove specific element from A")
        print("4. Replace element in A")
        print("5. Create Array B")
        print("6. Merge A and B")
        print("7. Show Arrays")
        print("8. Exit")

        choice = input("Enter your choice: ")

        if choice == '1':
            try:
                n = int(input("Enter number of elements in Array A: "))
                if n < 0:
                    print("Array size cannot be negative!")
                    continue
            except ValueError:
                print("Invalid input! Please enter a valid number.")
                continue

            a.clear()
            create(a, n)
        elif choice == '2':
            remove(a)
        elif choice == '3':
            remove_at(a)
        elif choice == '4':
            replace(a)
        elif choice == '5':
            try:
                m = int(input("Enter number of elements in Array B: "))
                if m < 0:
                    print("Array size cannot be negative!")
                    continue
            except ValueError:
                print("Invalid input! Please enter a valid number.")
                continue
            b.clear()
            create(b, m)
        elif choice == '6':
            merge(a, b)
        elif choice == '7':
            print("Array A:", a)
            print("Array B:", b)
        elif choice == '8':
            print("Exiting...")
            break
        else:
            print("Invalid choice!")
