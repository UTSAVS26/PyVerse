from bst import BST

def main():
    bst = BST()

    while True:
        print("\n" + "="*30)
        print("Binary Search Tree Menu")
        print("="*30)
        print("1. Insert a value")
        print("2. Search for a value")
        print("3. Display in-order traversal")
        print("4. Get tree height")
        print("5. Exit")
        print("="*30)

        try:
            choice = int(input("Enter your choice (1-5): "))
        except ValueError:
            print("Invalid input. Please enter a number between 1 and 5.")
            continue

        if choice == 1:
            value = int(input("Enter the value to insert: "))
            bst.insert(value)
            print(f"Value {value} inserted into the BST.")

        elif choice == 2:
            value = int(input("Enter the value to search: "))
            found = bst.search(value)
            if found:
                print(f"Value {value} found in the BST.")
            else:
                print(f"Value {value} not found in the BST.")

        elif choice == 3:
            print("In-order traversal of the BST: ", end="")
            bst.inorder()
            print()  

        elif choice == 4:
            height = bst.get_height()
            print(f"Height of the BST: {height}")

        elif choice == 5:
            print("Exiting program.")
            break

        else:
            print("Invalid choice. Please select a number between 1 and 5.")

if __name__ == "__main__":
    main()
