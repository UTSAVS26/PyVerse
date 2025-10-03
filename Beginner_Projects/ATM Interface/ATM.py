# Basic ATM Interface in Python

def display_menu():
    print("\nATM Menu")
    print("1. Check Balance")
    print("2. Deposit Money")
    print("3. Withdraw Money")
    print("4. Exit")

def account_number():
    account_number=input("enter your account number:")
    return account_number

def check_balance(balance):
    print(f"\nYour current balance is: ${balance}")


def deposit_money(balance):
    try:
        amount = float(input("Enter amount to deposit: $"))
        if amount > 0:
            balance += amount
            print(f"${amount} deposited successfully.")
        else:
            print("Invalid deposit amount!")
    except ValueError:
        print("Please enter a valid number.")
    return balance


def withdraw_money(balance):
    try:
        amount = float(input("Enter amount to withdraw: $"))
        if amount > 0 and amount <= balance:
            balance -= amount
            print(f"${amount} withdrawn successfully.")
        elif amount > balance:
            print("Insufficient funds!")
        else:
            print("Invalid withdrawal amount!")
    except ValueError:
        print("Please enter a valid number.")
    return balance


def atm_interface():
    balance = 5000.00  # Initial balance
    while True:
        display_menu()
        choice = int(input("\nEnter your choice (1-4): "))

        if choice == '1':
            check_balance(balance)
        elif choice == '2':
            balance = deposit_money(balance)
        elif choice == '3':
            balance = withdraw_money(balance)
        elif choice == '4':
            print("Thank you for using the ATM. Goodbye!")
            break
        else:
            print("Invalid choice! Please try again.")


# Run the ATM interface
print(account_number())
atm_interface()
