#!/usr/bin/env python3
import json
import os
from datetime import datetime
from rich.console import Console
from rich.table import Table
import argparse
from rich.prompt import Prompt

console = Console()

TASK_TRACKER_ASCII = """
╔╤╤╤╤╤╤╤╤╤╤╤╤╤╤╤╤╤╤╤╤╤╤╤╤╤╤╤╤╤╤╤╤╤╤╤╤╤╤╤╤╤╤╤╤╤╤╤╤╤╤╤╤╤╤╤╤╤╤╤╗
╟┼┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┼╢
╟┤████████╗ █████╗ ███████╗██╗  ██╗                        ├╢
╟┤╚══██╔══╝██╔══██╗██╔════╝██║ ██╔╝                        ├╢
╟┤   ██║   ███████║███████╗█████╔╝                         ├╢
╟┤   ██║   ██╔══██║╚════██║██╔═██╗                         ├╢
╟┤   ██║   ██║  ██║███████║██║  ██╗                        ├╢
╟┤   ╚═╝   ╚═╝  ╚═╝╚══════╝╚═╝  ╚═╝                        ├╢
╟┤████████╗██████╗  █████╗  ██████╗██╗  ██╗███████╗██████╗ ├╢
╟┤╚══██╔══╝██╔══██╗██╔══██╗██╔════╝██║ ██╔╝██╔════╝██╔══██╗├╢
╟┤   ██║   ██████╔╝███████║██║     █████╔╝ █████╗  ██████╔╝├╢
╟┤   ██║   ██╔══██╗██╔══██║██║     ██╔═██╗ ██╔══╝  ██╔══██╗├╢
╟┤   ██║   ██║  ██║██║  ██║╚██████╗██║  ██╗███████╗██║  ██║├╢
╟┤   ╚═╝   ╚═╝  ╚═╝╚═╝  ╚═╝ ╚═════╝╚═╝  ╚═╝╚══════╝╚═╝  ╚═╝├╢
╟┼┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┼╢
╚╧╧╧╧╧╧╧╧╧╧╧╧╧╧╧╧╧╧╧╧╧╧╧╧╧╧╧╧╧╧╧╧╧╧╧╧╧╧╧╧╧╧╧╧╧╧╧╧╧╧╧╧╧╧╧╧╧╧╧╝
"""

def load_tasks():
    if not os.path.exists('tasks.json'):
        with open('tasks.json', 'w') as file:
            json.dump([], file)
        return []
    
    with open('tasks.json', 'r') as file:
        try:
            return json.load(file)
        except json.JSONDecodeError:
            return []

def save_tasks(tasks):
    with open('tasks.json', 'w') as file:
        json.dump(tasks, file, indent=4)


def get_next_id(tasks):
    return max([task['id'] for task in tasks], default=0) + 1

def add_task(description):
    tasks = load_tasks()
    task_id = get_next_id(tasks)
    current_time = datetime.now()
    formatted_time = current_time.strftime("%Y-%m-%d %H:%M:%S")
    
    new_task = {
        'id': task_id,
        'description': description,
        'status': 'Pending',
        'createdAt': formatted_time,
        'updatedAt': formatted_time,
    }
    tasks.append(new_task)
    save_tasks(tasks)
    console.print(f"[green]Task added successfully (ID: {task_id})[/green]")


def list_tasks(status=None):
    """List tasks filtered by status, or list all tasks if no status is provided."""
    tasks = load_tasks()
    if status:
        tasks = [task for task in tasks if task['status'] == status.lower().capitalize()]
    if not tasks:
        console.print("[red]No tasks found.[/red]")
        return
    
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("#")
    table.add_column("Description")
    table.add_column("Status")
    table.add_column("Created At")
    table.add_column("Updated At")
    for task in tasks:
        if task['status'] == 'Pending':
            status_color = "green"
        elif task['status'] == 'In-progress':
            status_color = "purple"
        elif task['status'] == 'Completed':
            status_color = "red"
        else:
            status_color = "white"
        table.add_row(
            str(task['id']),
            task['description'],
            f"[{status_color}]{task['status']}[/{status_color}]",
            task['createdAt'],
            task['updatedAt']
        )
        
    console.print(table)


def update_task_status(task_id, new_status):
    """Update the status of an existing task."""
    tasks = load_tasks()
    for task in tasks:
        if task['id'] == task_id:
            task['status'] = new_status
            task['updatedAt'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            save_tasks(tasks)
            console.print(f"[green]Task ID {task_id} marked as {new_status}.[/green]")
            return
    console.print(f"[red]Task ID {task_id} not found.[/red]")

def mark_in_progress(task_id):
    update_task_status(task_id, 'In-progress')

def mark_done(task_id):
    update_task_status(task_id, 'Completed')

def delete_task(task_id):
    """Delete a task by its ID."""
    tasks = load_tasks()
    task_exists = False
    for task in tasks:
        if task['id'] == task_id:
            tasks.remove(task)
            task_exists = True
            break
    if task_exists:
        save_tasks(tasks)
        console.print(f"[yellow]Task ID {task_id} deleted successfully.[/yellow]")
    else:
        console.print(f"[red]Task ID {task_id} does not exist.[/red]")


def update_task(task_id, new_description):
    tasks = load_tasks()
    for task in tasks:
        if task['id'] == task_id:
            task['description'] = new_description
            task['updatedAt'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            save_tasks(tasks)
            console.print(f"[green]Task ID {task_id} updated successfully.[/green]")
            return
    console.print(f"[red]Task ID {task_id} not found.[/red]")


def main():
    parser = argparse.ArgumentParser(description="Task Tracker CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Add command
    parser_add = subparsers.add_parser("add", help="Add a new task")
    parser_add.add_argument("description", type=str, help="Description of the task")

    # List command
    parser_list = subparsers.add_parser("list", help="List all tasks")
    parser_list.add_argument("status", type=str, nargs='?', default=None, help="Status of the tasks to list (optional)")

    # Update command
    parser_update = subparsers.add_parser("update", help="Update an existing task")
    parser_update.add_argument("id", type=int, help="ID of the task to update")
    parser_update.add_argument("description", type=str, help="New description of the task")

    # Mark command
    parser_mark = subparsers.add_parser("mark", help="Mark a task as completed or pending")
    parser_mark.add_argument("id", type=int, help="ID of the task to mark")
    parser_mark.add_argument("status", type=str, choices=["completed", "pending"], help="New status of the task")

    # Delete command
    parser_delete = subparsers.add_parser("delete", help="Delete a task")
    parser_delete.add_argument("id", type=int, help="ID of the task to delete")

    args = parser.parse_args()

    console.print(TASK_TRACKER_ASCII, style="bold violet")
    console.print("Welcome to the Task Tracker v1.0", style="green")
    console.print(
        """
        - A CLI-based Task tracker tool that can easily track your small todo tasks.
        - Store them in JSON format.
        - Keep a log of them.
        - Categorize them by using specific labels.
        """,
        style="green"
    )

    if args.command == "add":
        add_task(args.description)
    elif args.command == "list":
        list_tasks(args.status)
    elif args.command == "update":
        update_task(args.id, args.description)
    elif args.command == "mark":
        update_task_status(args.id, args.status.capitalize())
    elif args.command == "delete":
        if Prompt.ask(f"Are you sure you want to delete task {args.id}?", choices=["y", "n"]) == "y":
            delete_task(args.id)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
