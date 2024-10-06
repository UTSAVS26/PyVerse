# Task Manager Application

This is a simple GUI-based Task Manager application built using Python's `tkinter` library and `sqlite3` for database management. The application allows users to add, update, delete, and mark tasks as completed. Tasks can also be sorted based on priority and deadline.

## Key Features

- **Add Task**: Add a new task with a description, priority, and deadline.
- **Update Task**: Update the details of an existing task.
- **Delete Task**: Remove a task from the list.
- **Mark as Completed**: Mark a task as completed.
- **Sort Tasks**: Sort tasks based on priority and deadline.
- **Persistent Storage**: Tasks are stored in a SQLite database, ensuring data persistence.

## Description

The application consists of a main window with input fields for task details, buttons for various actions, and a listbox to display the tasks. The tasks are stored in a SQLite database (`tasks.db`) with the following schema:

```sql
CREATE TABLE IF NOT EXISTS tasks (
    id INTEGER PRIMARY KEY,
    task TEXT,
    priority INTEGER,
    deadline TEXT,
    completed INTEGER
);
```

## How to Use

1. **Setup**: Ensure you have Python installed on your system. Install the required libraries if not already available:
    ```bash
    pip install tk
    ```

2. **Run the Application**: Save the provided code in a file (e.g., `task_manager.py`) and run it using Python:
    ```bash
    python task_manager.py
    ```

3. **Add a Task**: Enter the task details (description, priority, deadline) in the respective fields and click "Add Task".

4. **Update a Task**: Select a task from the list, modify the details in the input fields, and click "Update Task".

5. **Delete a Task**: Select a task from the list and click "Delete Task".

6. **Mark as Completed**: Select a task from the list and click "Mark as Completed".

7. **Sort Tasks**: Click "Sort Tasks" to sort the tasks based on priority and deadline.

## Code Overview

The main components of the application are:

- **Database Setup**: Initializes the SQLite database and creates the `tasks` table if it doesn't exist.
- **TaskManager Class**: Manages the GUI and interactions with the database.
  - `__init__`: Initializes the main window and loads tasks.
  - `create_widgets`: Creates the input fields, buttons, and task listbox.
  - `add_task`, `update_task`, `delete_task`, `complete_task`, `sort_tasks`: Methods to handle respective actions.
  - `load_tasks`: Loads tasks from the database and displays them in the listbox.
  - `get_task_details`: Retrieves task details from input fields and validates them.
  - `get_selected_task`: Gets the currently selected task from the listbox.
  - `update_task_listbox`: Updates the listbox with the provided tasks.

## Example

Here is an example of how the tasks are displayed in the listbox:

```
1 Buy groceries (Priority: 1, Deadline: 2023-10-01, Completed: No)
2 Finish project report (Priority: 2, Deadline: 2023-10-05, Completed: Yes)
```

## Closing the Application

When you close the application, the database connection is also closed to ensure data integrity.

```python
if __name__ == "__main__":
    root = tk.Tk()
    app = TaskManager(root)
    root.mainloop()
    conn.close()
```

This ensures that all changes are saved and the application exits gracefully.


## Author
`Akash Choudhury`
[GitHub Profile](https://github.com/ezDecode)