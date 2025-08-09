from bst import BST
from tree_exceptions import EmptyTreeError, InvalidValueError

def print_menu():
    print("\n" + "="*40)
    print("Enhanced Binary Search Tree Menu")
    print("="*40)
    print("1. Insert a value")
    print("2. Delete a value")
    print("3. Search for a value")
    print("4. Display all traversals")
    print("5. Get tree height")
    print("6. Get total nodes")
    print("7. Get maximum value in the tree")
    print("8. Get minimum value in the tree")
    print("9. Clone the Binary Search Tree")
    print("10. Mirror the Binary Search Tree")
    print("11. Pretty print the Binary Search Tree")
    print("12. Exit")
    print("="*40)

def main():
    bst = BST()

    while True:
        print_menu()
        try:
            choice = int(input("Enter your choice (1-12): "))
        except ValueError:
            print("Invalid input. Please enter a number between 1 and 12.")
            continue

        try:
            if choice == 1:
                value = int(input("Enter the value to insert: "))
                bst.insert(value)
                print(f"Value {value} inserted into the BST.")

            elif choice == 2:
                if bst.root is None:
                    print("Tree is empty!")
                    continue
                value = int(input("Enter the value to delete: "))
                bst.delete(value)
                print(f"Value {value} deleted from the BST.")

            elif choice == 3:
                value = int(input("Enter the value to search: "))
                found, count = bst.search(value)
                if found:
                    print(f"Value {value} found in the BST. Count: {count}")
                else:
                    print(f"Value {value} not found in the BST.")

            elif choice == 4:
                if bst.root is None:
                    print("Tree is empty!")
                    continue
                traversals = bst.traversals()
                print("\nInorder:", traversals['inorder'])
                print("Preorder:", traversals['preorder'])
                print("Postorder:", traversals['postorder'])

            elif choice == 5:
                height = bst.get_height()
                print(f"Height of the BST: {height}")

            elif choice == 6:
                print(f"Total nodes in the BST: {bst.num_nodes}")

            elif choice == 7:
                if bst.root is None:
                    print("Tree is empty!")
                    continue
                node = bst.find_max()
                print(f"Value of the maximum node : {node.value}")
                
            elif choice == 8:
                if bst.root is None:
                    print("Tree is empty!")
                    continue
                node = bst.find_min()
                print(f"Value of the minimum node : {node.value}")
            elif choice == 9:
                cloned_bst = bst.clone_bst()
                print("\nCloned Binary Search Tree :")
                cloned_bst.print_tree()
                
            elif choice == 10:
                mirrored_tree = bst.mirror_bst()
                print("\n Mirrored Tree :")
                mirrored_tree.print_tree()
                
            elif choice == 11:
                print("\nBinary Search Tree :")
                bst.print_tree()
                
            elif choice == 12:
                print("Thank you for using the BST program!")
                break

            else:
                print("Invalid choice. Please select a number between 1 and 12.")

        except (EmptyTreeError, InvalidValueError) as e:
            print(f"Error: {e}")
        except Exception as e:
            print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    main()