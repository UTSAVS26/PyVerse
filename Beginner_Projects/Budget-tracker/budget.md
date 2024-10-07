# Budget Tracker

## Overview

This Python script is a simple budget tracker that allows users to manage their financial transactions. It provides functionalities to add, update, delete, and read transactions, as well as generate a financial report summarizing income, expenses, and savings.

## Features

- **Initialize Data File**: Automatically creates a CSV file to store transaction data if it doesn't already exist.
- **Add Transaction**: Allows users to add a new transaction to the CSV file.
- **Update Transaction**: Enables users to update an existing transaction in the CSV file based on the index provided.
- **Delete Transaction**: Permits users to delete a transaction from the CSV file based on the index provided.
- **Read Transactions**: Reads all transactions from the CSV file, excluding the header.
- **Generate Report**: Generates a report summarizing total income, total expenses, and savings.

## Functions

### `initialize_data_file()`
Initializes the data file by creating a CSV file with headers if it does not already exist.

### `add_transaction(transaction_type, category, amount, description)`
Adds a new transaction to the CSV file.

### `update_transaction(index, transaction_type, category, amount, description)`
Updates an existing transaction in the CSV file at the specified index.

### `delete_transaction(index)`
Deletes a transaction from the CSV file at the specified index.

### `read_transactions()`
Reads all transactions from the CSV file, excluding the header.

### `generate_report()`
Generates and prints a report of total income, total expenses, and savings.

### `main()`
The main function that provides a command-line interface for the user to interact with the budget tracker.

## Usage

Run the script and follow the on-screen prompts to add, update, delete transactions, or generate a financial report.

```sh
python budget_tracker.py
```

## Example

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