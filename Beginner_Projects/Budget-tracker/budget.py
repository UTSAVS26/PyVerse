import csv
import os

DATA_FILE = 'budget_data.csv'

def initialize_data_file():
    if not os.path.exists(DATA_FILE):
        with open(DATA_FILE, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Type', 'Category', 'Amount', 'Description'])

def add_transaction(transaction_type, category, amount, description):
    with open(DATA_FILE, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([transaction_type, category, amount, description])

def update_transaction(index, transaction_type, category, amount, description):
    temp_file = DATA_FILE + '.tmp'
    with open(DATA_FILE, mode='r') as file, open(temp_file, mode='w', newline='') as temp:
        reader = csv.reader(file)
        writer = csv.writer(temp)
        for i, row in enumerate(reader):
            if i == index + 1:  # Skip header
                writer.writerow([transaction_type, category, amount, description])
            else:
                writer.writerow(row)
    os.replace(temp_file, DATA_FILE)

def delete_transaction(index):
    temp_file = DATA_FILE + '.tmp'
    with open(DATA_FILE, mode='r') as file, open(temp_file, mode='w', newline='') as temp:
        reader = csv.reader(file)
        writer = csv.writer(temp)
        for i, row in enumerate(reader):
            if i != index + 1:  # Skip header
                writer.writerow(row)
    os.replace(temp_file, DATA_FILE)

def read_transactions():
    with open(DATA_FILE, mode='r') as file:
        reader = csv.reader(file)
        return list(reader)[1:]  # Skip header

def generate_report():
    income = 0.0
    expenses = 0.0
    with open(DATA_FILE, mode='r') as file:
        reader = csv.reader(file)
        next(reader)  # Skip header
        for row in reader:
            amount = float(row[2])
            if row[0] == 'income':
                income += amount
            elif row[0] == 'expense':
                expenses += amount
    savings = income - expenses

    print(f"Total Income: ${income:.2f}")
    print(f"Total Expenses: ${expenses:.2f}")
    print(f"Savings: ${savings:.2f}")

def main():
    initialize_data_file()
    while True:
        print("\nBudget Tracker")
        print("1. Add Transaction")
        print("2. Update Transaction")
        print("3. Delete Transaction")
        print("4. Generate Report")
        print("5. Exit")
        choice = input("Choose an option: ")

        if choice == '1':
            transaction_type = input("Enter transaction type (income/expense): ")
            category = input("Enter category: ")
            amount = input("Enter amount: ")
            description = input("Enter description: ")
            add_transaction(transaction_type, category, amount, description)
        elif choice == '2':
            index = int(input("Enter transaction index to update: "))
            transaction_type = input("Enter transaction type (income/expense): ")
            category = input("Enter category: ")
            amount = input("Enter amount: ")
            description = input("Enter description: ")
            update_transaction(index, transaction_type, category, amount, description)
        elif choice == '3':
            index = int(input("Enter transaction index to delete: "))
            delete_transaction(index)
        elif choice == '4':
            generate_report()
        elif choice == '5':
            break
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main()