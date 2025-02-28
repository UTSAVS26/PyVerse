users = {}
workouts = [
    {"workout_type": "Push Up", "duration": 30, "calories": 100, "level": "Beginner"},
    {"workout_type": "Squat", "duration": 30, "calories": 120, "level": "Beginner"},
    {"workout_type": "Lunges", "duration": 30, "calories": 130, "level": "Beginner"},
    {"workout_type": "Jumping Jack", "duration": 30, "calories": 150, "level": "Beginner"},
    {"workout_type": "Sit Up", "duration": 30, "calories": 100, "level": "Beginner"},
    {"workout_type": "Plank", "duration": 30, "calories": 80, "level": "Beginner"},
    {"workout_type": "Glute Bridge", "duration": 30, "calories": 90, "level": "Beginner"},
    {"workout_type": "Bicycle Crunch", "duration": 30, "calories": 110, "level": "Beginner"},
    {"workout_type": "Wall Sit", "duration": 30, "calories": 70, "level": "Beginner"},
    {"workout_type": "Tricep Dips", "duration": 30, "calories": 120, "level": "Beginner"},
    
    {"workout_type": "Burpee", "duration": 30, "calories": 150, "level": "Intermediate"},
    {"workout_type": "Mountain Climber", "duration": 30, "calories": 160, "level": "Intermediate"},
    {"workout_type": "Kettlebell Swing", "duration": 30, "calories": 200, "level": "Intermediate"},
    {"workout_type": "Tricep Dips", "duration": 30, "calories": 120, "level": "Intermediate"},
    {"workout_type": "Bicycle Crunches", "duration": 30, "calories": 110, "level": "Intermediate"},
    {"workout_type": "High Knees", "duration": 30, "calories": 150, "level": "Intermediate"},
    {"workout_type": "Push Press", "duration": 30, "calories": 180, "level": "Intermediate"},
    {"workout_type": "Box Jump", "duration": 30, "calories": 230, "level": "Intermediate"},
    {"workout_type": "Jump Rope", "duration": 30, "calories": 200, "level": "Intermediate"},
    {"workout_type": "Plank Jacks", "duration": 30, "calories": 140, "level": "Intermediate"},
    
    {"workout_type": "Running", "duration": 30, "calories": 300, "level": "Advanced"},
    {"workout_type": "Deadlift", "duration": 30, "calories": 250, "level": "Advanced"},
    {"workout_type": "Bench Press", "duration": 30, "calories": 200, "level": "Advanced"},
    {"workout_type": "Pull Up", "duration": 30, "calories": 200, "level": "Advanced"},
    {"workout_type": "Thruster", "duration": 30, "calories": 250, "level": "Advanced"},
    {"workout_type": "Barbell Squat", "duration": 30, "calories": 220, "level": "Advanced"},
    {"workout_type": "Plyometric Push Up", "duration": 30, "calories": 210, "level": "Advanced"},
    {"workout_type": "Tire Flip", "duration": 30, "calories": 300, "level": "Advanced"},
    {"workout_type": "Sled Push", "duration": 30, "calories": 280, "level": "Advanced"},
    {"workout_type": "Battle Ropes", "duration": 30, "calories": 250, "level": "Advanced"},
]

def register_user(username, level):
    if username in users:
        print("Username already exists.")
    else:
        users[username] = {"level": level, "progress": 0}
        print("User registered successfully!")

def log_workout(username, workout_type):
    if username not in users:
        print("User not found.")
        return
    
    workout = next((w for w in workouts if w["workout_type"].lower() == workout_type.lower()), None)
    
    if workout:
        users[username]["progress"] += workout["calories"]
        print(f"Workout '{workout['workout_type']}' logged! Calories burned: {workout['calories']}")
    else:
        print("Workout type not found.")

def view_user_progress(username):
    if username in users:
        user = users[username]
        print(f"\nUser: {username}\nLevel: {user['level']}\nTotal Calories Burned: {user['progress']}\n")
    else:
        print("User not found.")

def list_workouts():
    print("\nAvailable Workouts:")
    for workout in workouts:
        print(f"- {workout['workout_type']} (Calories: {workout['calories']}, Level: {workout['level']})")
    print()

def main():
    while True:
        print("\nFitness Tracker")
        print("1. Register User")
        print("2. Log Workout")
        print("3. View Progress")
        print("4. List Available Workouts")
        print("5. Exit")
        
        choice = input("Choose an option: ")
        
        if choice == '1':
            username = input("Enter username: ")
            level = input("Enter fitness level (Beginner/Intermediate/Advanced): ")
            register_user(username, level)
            
        elif choice == '2':
            username = input("Enter your username: ")
            list_workouts()
            workout_type = input("Enter workout type: ")
            log_workout(username, workout_type)
            
        elif choice == '3':
            username = input("Enter your username: ")
            view_user_progress(username)
            
        elif choice == '4':
            list_workouts()
            
        elif choice == '5':
            print("Thank you for using the Fitness Tracker! Goodbye!")
            break
            
        else:
            print("Invalid choice, please try again.")

if __name__ == "__main__":
    main()