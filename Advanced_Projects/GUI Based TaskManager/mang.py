import tkinter as tk
from tkinter import messagebox
import sqlite3
from datetime import datetime

# Database setup
conn = sqlite3.connect('tasks.db')
c = conn.cursor()
c.execute('''CREATE TABLE IF NOT EXISTS tasks
             (id INTEGER PRIMARY KEY, task TEXT, priority INTEGER, deadline TEXT, completed INTEGER)''')
conn.commit()

class TaskManager:
    def __init__(self, root):
        self.root = root
        self.root.title("Task Manager")
        
        self.create_widgets()
        self.load_tasks()
    
    def create_widgets(self):
        # Task input
        self.create_label_entry("Task", 0)
        self.create_label_entry("Priority", 1)
        self.create_label_entry("Deadline (YYYY-MM-DD)", 2)
        
        # Buttons
        self.create_button("Add Task", self.add_task, 3)
        self.create_button("Update Task", self.update_task, 4)
        self.create_button("Delete Task", self.delete_task, 5)
        self.create_button("Mark as Completed", self.complete_task, 6)
        self.create_button("Sort Tasks", self.sort_tasks, 7)
        
        # Task list
        self.task_listbox = tk.Listbox(self.root)
        self.task_listbox.grid(row=8, column=0, columnspan=2)
    
    def create_label_entry(self, text, row):
        label = tk.Label(self.root, text=text)
        label.grid(row=row, column=0)
        entry = tk.Entry(self.root)
        entry.grid(row=row, column=1)
        setattr(self, f"{text.lower().split()[0]}_entry", entry)
    
    def create_button(self, text, command, row):
        button = tk.Button(self.root, text=text, command=command)
        button.grid(row=row, column=0, columnspan=2)
    
    def add_task(self):
        task, priority, deadline = self.get_task_details()
        if not task or not priority or not deadline:
            return
        
        c.execute("INSERT INTO tasks (task, priority, deadline, completed) VALUES (?, ?, ?, ?)",
                  (task, priority, deadline, 0))
        conn.commit()
        self.load_tasks()
    
    def update_task(self):
        selected_task = self.get_selected_task()
        if not selected_task:
            return
        
        task_id = selected_task.split()[0]
        task, priority, deadline = self.get_task_details()
        if not task or not priority or not deadline:
            return
        
        c.execute("UPDATE tasks SET task = ?, priority = ?, deadline = ? WHERE id = ?",
                  (task, priority, deadline, task_id))
        conn.commit()
        self.load_tasks()
    
    def delete_task(self):
        selected_task = self.get_selected_task()
        if not selected_task:
            return
        
        task_id = selected_task.split()[0]
        c.execute("DELETE FROM tasks WHERE id = ?", (task_id,))
        conn.commit()
        self.load_tasks()
    
    def complete_task(self):
        selected_task = self.get_selected_task()
        if not selected_task:
            return
        
        task_id = selected_task.split()[0]
        c.execute("UPDATE tasks SET completed = 1 WHERE id = ?", (task_id,))
        conn.commit()
        self.load_tasks()
    
    def sort_tasks(self):
        c.execute("SELECT * FROM tasks ORDER BY priority, deadline")
        tasks = c.fetchall()
        self.update_task_listbox(tasks)
    
    def load_tasks(self):
        c.execute("SELECT * FROM tasks")
        tasks = c.fetchall()
        self.update_task_listbox(tasks)
    
    def get_task_details(self):
        task = self.task_entry.get()
        priority = self.priority_entry.get()
        deadline = self.deadline_entry.get()
        
        if not task or not priority or not deadline:
            messagebox.showwarning("Input Error", "All fields are required")
            return None, None, None
        
        try:
            priority = int(priority)
            datetime.strptime(deadline, '%Y-%m-%d')
        except ValueError:
            messagebox.showwarning("Input Error", "Invalid priority or deadline format")
            return None, None, None
        
        return task, priority, deadline
    
    def get_selected_task(self):
        selected_task = self.task_listbox.curselection()
        if not selected_task:
            messagebox.showwarning("Selection Error", "No task selected")
            return None
        return self.task_listbox.get(selected_task)
    
    def update_task_listbox(self, tasks):
        self.task_listbox.delete(0, tk.END)
        for task in tasks:
            self.task_listbox.insert(tk.END, f"{task[0]} {task[1]} (Priority: {task[2]}, Deadline: {task[3]}, Completed: {'Yes' if task[4] else 'No'})")

if __name__ == "__main__":
    root = tk.Tk()
    app = TaskManager(root)
    root.mainloop()
    conn.close()