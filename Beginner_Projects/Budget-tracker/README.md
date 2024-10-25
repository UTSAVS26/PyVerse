# Budget Tracker

## Overview
This Python-based budget tracker helps users manage their financial transactions with ease. Users can add, update, delete, and view transactions, as well as generate a financial summary that includes total income, expenses, and savings.

## Features
- **Initialize Data File**: Automatically creates a CSV file to store transaction data if the file doesn't already exist.
- **Add Transaction**: Allows users to record a new income or expense with a category, amount, and description.
- **Update Transaction**: Allows users to modify an existing transaction based on its index (position) in the list.
- **Delete Transaction**: Allows users to remove a transaction by specifying its index.
- **Read Transactions**: Displays all transactions stored in the CSV file (excluding the header)
- **Generate Report**: Summarizes total income, total expenses, and calculates the savings.

## Functions
1. **``initialize_data_file()``**: Creates a CSV file with the headers 'Type', 'Category', 'Amount', 'Description' if it does not already exist.

2. **``add_transaction(transaction_type, category, amount, description)``**: Adds a new transaction to the CSV file:
    - **``transaction_type``**: 'income' or 'expense'
    - **``category``**: The category of the transaction (e.g., 'Salary', 'Groceries').
    - **``amount``**: The transaction amount (positive numbers only).
    - **``description``**: A brief description of the transaction.

3. **``update_transaction(index, transaction_type, category, amount, description)``**: Modifies an existing transaction at the specified index. The index is 0-based (e.g., the first transaction is at index 0).

4. **``delete_transaction(index)``**: Deletes a transaction from the CSV file based on its index.

5. **``read_transactions()``**: Reads and returns all transactions from the CSV file, excluding the header.

6. **``generate_report()``**: Calculates total income, total expenses, and savings, and prints a financial report. Savings are calculated as the difference between income and expenses.

7. **``main()``**: The main function that provides a command-line interface for the user to interact with the budget tracker.

## Requirements
- Python 3.x

## Usage
- Run the script from the command line or terminal: ``python budget_tracker.py``
- Follow the on-screen prompts to add, update, delete transactions, or generate a financial report.

### Example:
```sh
Budget Tracker
1. Add Transaction
2. Update Transaction
3. Delete Transaction
4. Generate Report
5. Exit

Choose an option: 1
Enter transaction type (income/expense): income
Enter category: Salary
Enter amount: 5000
Enter description: Monthly salary
```

This will add a new income transaction to the budget data.

## Author
`Akash Choudhury`
[GitHub Profile](https://github.com/ezDecode)
