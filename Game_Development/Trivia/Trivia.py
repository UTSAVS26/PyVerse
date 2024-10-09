import random

def get_questions():
    return {
        "Geography": {
            "What is the capital of France?": "Paris",
            "Which river is the longest in the world?": "Nile",
            "What is the largest desert in the world?": "Sahara",
            "Which country has the most natural lakes?": "Canada",
            "What mountain range separates Europe and Asia?": "Ural",
            "What is the smallest country in the world?": "Vatican City",
            "Which city is known as the Big Apple?": "New York",
            "What continent is Egypt located in?": "Africa",
        },
        "Mathematics": {
            "What is 2 + 2?": "4",
            "What is the square root of 16?": "4",
            "What is the value of pi (up to two decimal points)?": "3.14",
            "What is 5 x 6?": "30",
            "What is the derivative of x^2?": "2x",
            "What is the Fibonacci sequence's first number?": "0",
            "What is 10% of 200?": "20",
            "What is the sum of the interior angles of a triangle?": "180",
        },
        "Literature": {
            "Who wrote 'Romeo and Juliet'?": "Shakespeare",
            "What is the title of the first Harry Potter book?": "Harry Potter and the Philosopher's Stone",
            "Who wrote '1984'?": "George Orwell",
            "Which novel begins with 'Call me Ishmael'?": "Moby Dick",
            "Who wrote 'Pride and Prejudice'?": "Jane Austen",
            "What is the pen name of Samuel Clemens?": "Mark Twain",
            "Who wrote 'The Great Gatsby'?": "F. Scott Fitzgerald",
            "What is the first book in the 'Lord of the Rings' trilogy?": "The Fellowship of the Ring",
        },
        "Science": {
            "What is the chemical symbol for gold?": "Au",
            "What planet is known as the Red Planet?": "Mars",
            "What gas do plants absorb from the atmosphere?": "Carbon dioxide",
            "What is the speed of light?": "299792458 m/s",
            "What is H2O commonly known as?": "Water",
            "What part of the cell contains the genetic material?": "Nucleus",
            "What is the powerhouse of the cell?": "Mitochondria",
            "What element does 'O' represent on the periodic table?": "Oxygen",
        },
    }

def trivia_game():
    questions = get_questions()
    total_score = 0
    rounds = 3

    print("Welcome to the Trivia Game!")
    print("Categories: Geography, Mathematics, Literature, Science")
    
    for _ in range(rounds):
        category = input("\nChoose a category: ").strip().title()
        
        if category not in questions:
            print("Invalid category. Please try again.")
            continue
        
        question_items = list(questions[category].items())
        random.shuffle(question_items)

        score = 0
        total_questions = len(question_items)

        print(f"\nYou will be asked {total_questions} questions from the {category} category. Let's get started!")

        for question, answer in question_items:
            user_answer = input(f"{question} ")
            if user_answer.strip().lower() == answer.lower():
                print("Correct!")
                score += 1
            else:
                print(f"Wrong! The correct answer is {answer}.")
        
        print(f"Round over! Your score for this round: {score}/{total_questions}")
        total_score += score

    print(f"\nGame over! Your total score: {total_score}/{rounds * len(questions[category])}")

if __name__ == "__main__":
    trivia_game()
