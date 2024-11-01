import random
import time

# Function to generate a random math question
def generate_question():
    operations = ['+', '-', '*', '/']
    num1 = random.randint(1, 100)
    num2 = random.randint(1, 100)
    operation = random.choice(operations)

    # Ensure no division by zero
    if operation == '/':
        num1 = num1 * num2  # Ensure division gives a whole number

    # Create the question and calculate the correct answer
    if operation == '+':
        answer = num1 + num2
    elif operation == '-':
        answer = num1 - num2
    elif operation == '*':
        answer = num1 * num2
    elif operation == '/':
        answer = num1 // num2  # Perform integer division
    
    return f"{num1} {operation} {num2}", answer

# Function to start the game
def start_game():
    score = 0
    total_questions = 0
    time_limit = 60  # 60 seconds for the game
    start_time = time.time()  # Get the current time

    print("Welcome to Math Wizard!")
    print("Solve as many math equations as you can in 60 seconds.\n")

    while time.time() - start_time < time_limit:
        # Generate a random question
        question, correct_answer = generate_question()
        print(f"Question: {question}")
        
        try:
            # Get player's answer
            player_answer = int(input("Your answer: "))

            # Check if the answer is correct
            if player_answer == correct_answer:
                print("Correct!\n")
                score += 1
            else:
                print(f"Incorrect! The correct answer was {correct_answer}\n")

            total_questions += 1

        except ValueError:
            print("Please enter a valid number!\n")

    # End of game, print the result
    print(f"\nTime's up! You solved {score} out of {total_questions} questions correctly.")
    print(f"Your final score is: {score}")

if __name__ == "__main__":
    start_game()
