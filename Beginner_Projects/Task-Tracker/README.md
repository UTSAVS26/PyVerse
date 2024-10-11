# Task Tracker CLI Project

The **Task Tracker** is a command-line interface (CLI) project designed to help you track and manage tasks. It allows you to add, update, and delete tasks, as well as mark tasks as "in-progress" or "done." This project will help you practice essential programming skills, including working with the filesystem, handling user inputs, and building a basic CLI application.

## Requirements

The application should run from the command line, accept user actions and inputs as arguments, and store the tasks in a JSON file. The user should be able to:

- Add, Update, and Delete tasks
- Mark a task as "in-progress" or "done"
- List all tasks
- List all tasks that are marked as "done"
- List all tasks that are marked as "todo"
- List all tasks that are marked as "in-progress"

### Constraints

- You can use any programming language to build this project.
- Use **positional arguments** in the command line to accept user inputs.
- Use a **JSON file** to store the tasks in the current directory.
- The JSON file should be created if it does not exist.
- Use the **native filesystem module** of your programming language to interact with the JSON file.
- Do not use any external libraries or frameworks to build this project.
- Ensure to handle errors and edge cases gracefully.

## Example Commands

Here are some example commands and their usage:

```bash
# Adding a new task
taskr add "Buy groceries"
# Output: Task added successfully (ID: 1)

# Updating and deleting tasks
taskr update 1 "Buy groceries and cook dinner"
taskr delete 1

# Marking a task as in-progress or done
taskr mark-in-progress 1
taskr mark-done 1

# Listing all tasks
taskr list

# Listing tasks by status
taskr list done
taskr list todo
taskr list in-progress
```

## Task Properties

Each task should have the following properties:

- `id`: A unique identifier for the task
- `description`: A short description of the task
- `status`: The status of the task (`todo`, `in-progress`, `done`)
- `createdAt`: The date and time when the task was created
- `updatedAt`: The date and time when the task was last updated

These properties should be stored in the JSON file when adding a new task and updated when modifying a task.

## Conclusion

This project is an opportunity to improve your programming and CLI development skills. It allows you to practice interacting with the filesystem, handling JSON data, and managing user input via the command line—all while building a useful task-tracking tool.

Original Project Link: [Task Tracker CLI](https://roadmap.sh/projects/task-tracker)

---

## Getting Started

### Prerequisites

Make sure you have Python installed on your system. You can download it from [python.org](https://www.python.org/).

### Installation

1. Clone the repository:
    ```sh
    git clone https://github.com/yashksaini-coder/Task-Tracker.git
    cd Task-Tracker
    ```

2. Create a virtual environment:
    ```sh
    python -m venv venv
    ```

3. Activate the virtual environment:
    - On Windows:
        ```sh
        venv\Scripts\activate
        ```
    - On macOS/Linux:
        ```sh
        source venv/bin/activate
        ```

4. Install the required dependencies:
    ```sh
    pip install -r requirements.txt
    ```

### Usage

1. Run the Task Tracker CLI:
    ```sh
    pip install -e .
    task-tracker    
    ```

2. Follow the on-screen instructions to add, view, update, or delete tasks.

### Basic Commands

- **Add a Task**: 
    ```sh
    taskr add "Task Description"
    ```

- **View All Tasks**: 
    ```sh
    taskr list [status]
    ```

- **Update a Task**: 
    ```sh
    taskr update <task_id> "New Task Description"
    ```

- **Delete a Task**: 
    ```sh
    taskr delete <task_id>
    ```

### License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---
