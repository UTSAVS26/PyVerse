# Define the tasks dictionary
tasks = {}

# Function to add a task
def add_task(title, content):
    if title.strip() == "":
        return "Task title cannot be empty."
    if title in tasks:
        return "Task title already exists. Please choose a different title."
    tasks[title] = content
    return "Task added successfully!"

# Function to view all tasks
def view_tasks():
    if tasks:
        output = []
        for title, content in tasks.items():
            output.append(f"Title: {title}\nContent: {content}\n" + "-" * 20)
        return "\n".join(output)
    else:
        return "No tasks found."

# Function to update a task
def update_task(title, new_content):
    if title in tasks:
        tasks[title] = new_content
        return f"Task '{title}' updated successfully!"
    else:
        return f"Task '{title}' not found."

# Function to delete a task
def delete_task(title):
    if title in tasks:
        del tasks[title]
        return f"Task '{title}' deleted successfully!"
    else:
        return f"Task '{title}' not found."

# Function to display the menu
def task_menu():
    print("\nTASKER MENU")
    print("1. Add a Task")
    print("2. View All Tasks")
    print("3. Update a Task")
    print("4. Delete a Task")
    print("5. Exit")

# Main interactive loop
def main():
    while True:
        task_menu()
        choice = input("Enter your choice (1-5): ")
        
        if choice == '1':
            title = input("Enter task title: ")
            content = input("Enter task content: ")
            print(add_task(title, content))
        elif choice == '2':
            print(view_tasks())
        elif choice == '3':
            title = input("Enter title of task to update: ")
            new_content = input("Enter new content for the task: ")
            print(update_task(title, new_content))
        elif choice == '4':
            title = input("Enter title of task to delete: ")
            print(delete_task(title))
        elif choice == '5':
            print("Thank you for using the TASKER. Goodbye!")
            break
        else:
            print("Invalid choice. Please enter a number from 1 to 5.")

# Run the main function in the notebook
main()
# Define the tasks dictionary
tasks = {}

# Function to add a task
def add_task(title, content):
    if title.strip() == "":
        return "Task title cannot be empty."
    if title in tasks:
        return "Task title already exists. Please choose a different title."
    tasks[title] = content
    return "Task added successfully!"

# Function to view all tasks
def view_tasks():
    if tasks:
        output = []
        for title, content in tasks.items():
            output.append(f"Title: {title}\nContent: {content}\n" + "-" * 20)
        return "\n".join(output)
    else:
        return "No tasks found."

# Function to update a task
def update_task(title, new_content):
    if title in tasks:
        tasks[title] = new_content
        return f"Task '{title}' updated successfully!"
    else:
        return f"Task '{title}' not found."

# Function to delete a task
def delete_task(title):
    if title in tasks:
        del tasks[title]
        return f"Task '{title}' deleted successfully!"
    else:
        return f"Task '{title}' not found."

# Function to display the menu
def task_menu():
    print("\nTASKER MENU")
    print("1. Add a Task")
    print("2. View All Tasks")
    print("3. Update a Task")
    print("4. Delete a Task")
    print("5. Exit")

# Main interactive loop
def main():
    while True:
        task_menu()
        choice = input("Enter your choice (1-5): ")
        
        if choice == '1':
            title = input("Enter task title: ")
            content = input("Enter task content: ")
            print(add_task(title, content))
        elif choice == '2':
            print(view_tasks())
        elif choice == '3':
            title = input("Enter title of task to update: ")
            new_content = input("Enter new content for the task: ")
            print(update_task(title, new_content))
        elif choice == '4':
            title = input("Enter title of task to delete: ")
            print(delete_task(title))
        elif choice == '5':
            print("Thank you for using the TASKER. Goodbye!")
            break
        else:
            print("Invalid choice. Please enter a number from 1 to 5.")

# Run the main function in the notebook
main()
