tasks = {}

def add_task():
    title = input("Enter task title: ")
    if title.strip() == "":
        print("Task title cannot be empty.")
        return
    content = input("Enter task content: ")
    tasks[title] = content
    print("Task added successfully!")

def view_tasks():
    if tasks:
        print("Your tasks:")
        for title, content in tasks.items():
            print(f"Title: {title}")
            print(f"Content: {content}")
            print("-" * 20)
    else:
        print("No tasks found.")

def update_task():
    title = input("Enter title of task to update: ")
    if title in tasks:
        new_content = input("Enter new content for the task: ")
        tasks[title] = new_content
        print(f"Task '{title}' updated successfully!")
    else:
        print(f"Task '{title}' not found.")

def delete_task():
    title = input("Enter title of task to delete: ")
    if title in tasks:
        del tasks[title]
        print(f"Task '{title}' deleted successfully!")
    else:
        print(f"Task '{title}' not found.")

def main_menu():
    while True:
        print("\nTASKER")
        print("1. Add a Task")
        print("2. View All Tasks")
        print("3. Update a Task")
        print("4. Delete a Task")
        print("5. Exit")

        choice = input("Enter your choice (1-5): ")

        if choice == '1':
            add_task()
        elif choice == '2':
            view_tasks()
        elif choice == '3':
            update_task()
        elif choice == '4':
            delete_task()
        elif choice == '5':
            print("Thank you for using the TASKER. Goodbye!")
            break
        else:
            print("Invalid choice. Please enter a number from 1 to 5.")

if __name__ == "__main__":
    main_menu()
