# Simple Grocery Inventory Management

inventory = {}

def add_item():
    item_name = input("Enter the name of the item: ").strip().capitalize()
    if item_name in inventory:
        print(f"{item_name} already exists. Updating quantity.")
        inventory[item_name] += int(input("Enter the quantity to add: "))
    else:
        quantity = int(input("Enter the quantity: "))
        price = float(input("Enter the price per unit: "))
        inventory[item_name] = {"quantity": quantity, "price": price}
    print(f"{item_name} has been added/updated.")

def view_inventory():
    if not inventory:
        print("The inventory is empty.")
        return
    print("\nCurrent Inventory:")
    print(f"{'Item':<15}{'Quantity':<10}{'Price/Unit':<10}")
    print("-" * 35)
    for item, details in inventory.items():
        print(f"{item:<15}{details['quantity']:<10}{details['price']:<10.2f}")
    print()

def update_item():
    item_name = input("Enter the name of the item to update: ").strip().capitalize()
    if item_name in inventory:
        quantity = int(input("Enter the new quantity: "))
        price = float(input("Enter the new price per unit: "))
        inventory[item_name] = {"quantity": quantity, "price": price}
        print(f"{item_name} has been updated.")
    else:
        print(f"{item_name} does not exist in the inventory.")

def delete_item():
    item_name = input("Enter the name of the item to delete: ").strip().capitalize()
    if item_name in inventory:
        del inventory[item_name]
        print(f"{item_name} has been deleted.")
    else:
        print(f"{item_name} does not exist in the inventory.")

def main():
    while True:
        print("\nGrocery Inventory Management")
        print("1. Add Item")
        print("2. View Inventory")
        print("3. Update Item")
        print("4. Delete Item")
        print("5. Exit")
        choice = input("Enter your choice: ").strip()

        if choice == '1':
            add_item()
        elif choice == '2':
            view_inventory()
        elif choice == '3':
            update_item()
        elif choice == '4':
            delete_item()
        elif choice == '5':
            print("Exiting the program. Goodbye!")
            break
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main()
