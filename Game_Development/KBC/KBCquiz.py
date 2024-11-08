# Importing necessary modules
import random  # For random selection of questions and options
import time    # For adding delays to simulate a dynamic quiz experience

# Print 120 asterisks for visual effect
for i in range(120):
    print("*", end="")
    time.sleep(0)

# Print the welcome message
print()
print("\t\t\t                               Welcome to")
print("\t\t\t                           Kaun Banega Crorepati")

# Print another line of asterisks
for i in range(120):
    print("*", end="")
    time.sleep(0)
print()

# Ask the user to input their name
a = input("\t Enter Your Name - ")

# Print another line of asterisks
for i in range(120):
    print("*", end="")
    time.sleep(0)
print()

# Welcome message with the player's name
print("\n\tOK ", a, " Let's Start The Game")
time.sleep(1)

# List of questions and their corresponding answers
questions = [
    "Who is The Prime Minister of India",
    "In Which Country Area 51 is Located",
    "Which one is the largest Continent in the world",
    "What is the Latest Version of Windows Since 2019",
    "Which One of these is not a Software Company",
    "How Many MB Makes 1 GB",
    "Facebook Was Firstly Developed By",
    "Founder of Apple is",
    "_________ is one of The Founder of Google",
    "BIGG BOSS season 13 Starts in ____ & ends in _____",
    "Apple's Laptop is Also Known as",
    "First Apple Computer is Known as",
    "Joystick is used For",
    "____________ is used to Encrypt Drives in Computer"
]
answer = [
    "Narendra Modi", "United States", "Asia", "Windows 11", "Honda", "1024",
    "Mark Zuckerberg", "Steve Jobs", "Larry Page", "2019 - 2020", "Macbook",
    "Mactonish", "Playing Games", "Bitlocker"
]

# List of wrong answers for each question (to generate multiple choices)
wronganswers = [
    ["Amit Shah", "Aditya Nath Yogi", "Azhar Ansari"],
    ["India", "Africa", "Iraq"],
    ["South Africa", "North America", "Europe"],
    ["Windows 7", "Windows 8", "Windows 10"],
    ["Oracle", "Microsoft", "Google"],
    ["10024", "1004", "2024"],
    ["Bill Gates", "Larry Page", "Azhar Ansari"],
    ["Azhar Ansari", "Charles Babbage", "Sundar Pichai"],
    ["Larry Hensberg", "Sunder Pichai", "Bill Gates"],
    ["2020 - 2021", "Not Starts Now", "2018 - 2019"],
    ["ThinBook", "Notebook", "ChromeBook"],
    ["Apple v.1", "Apple Computer", "Appbook"],
    ["Giving output command", "Shutting down Computer", "Log off Computer"],
    ["KeyGuard", "Windows Secure", "No Software like this"]
]

# Initialize variables for attempted questions, question count, and prize amount
attempquestion = []
count = 1
amount = 0

# Start the game loop
while True:
    # Select a question that hasn't been asked yet
    while True:
        selectquestion = random.choice(questions)
        if selectquestion in attempquestion:
            pass  # Skip if the question was already asked
        else:
            attempquestion.append(selectquestion)  # Add the question to the attempted list
            questionindex = questions.index(selectquestion)  # Find the index of the selected question
            correctanswer = answer[questionindex]  # Get the correct answer for the question
            break

    # Generate multiple choice options
    optionslist = []
    inoptionlist = []
    optioncount = 1
    while optioncount < 4:  # Pick three wrong answers
        optionselection = random.choice(wronganswers[questionindex])
        if optionselection not in inoptionlist:
            optionslist.append(optionselection)
            inoptionlist.append(optionselection)
            optioncount += 1
    optionslist.append(correctanswer)  # Add the correct answer to the options list

    # Shuffle and display the options in random order
    alreadydisplay = []
    optiontodisplay = []
    for _ in range(4):
        while True:
            a = random.choice(optionslist)
            if a not in alreadydisplay:
                alreadydisplay.append(a)
                optiontodisplay.append(a)
                break

    # Identify the correct option label (a, b, c, d)
    right_answer = ""
    if correctanswer == optiontodisplay[0]:
        right_answer = "a"
    elif correctanswer == optiontodisplay[1]:
        right_answer = "b"
    elif correctanswer == optiontodisplay[2]:
        right_answer = "c"
    elif correctanswer == optiontodisplay[3]:
        right_answer = "d"

    # Display the question and options
    print("-"*120)
    print("\t\t\tAmount Won - ", amount)
    print("-"*120)
    time.sleep(1)
    print("\n\t\tQuestion ", count, " on your Screen")
    print("-"*120)
    time.sleep(1)
    print("  |  Question - ", selectquestion)
    print("\t"+("-"*80))
    time.sleep(1)
    print("\t|  A. ", optiontodisplay[0])
    print("\t"+("-"*80))
    time.sleep(1)
    print("\t|  B. ", optiontodisplay[1])
    print("\t"+("-"*80))
    time.sleep(1)
    print("\t|  C. ", optiontodisplay[2])
    print("\t"+("-"*80))
    time.sleep(1)
    print("\t|  D. ", optiontodisplay[3])
    print("\t"+("-"*80))

    # Ask the player for their answer
    useranswer = input("\t\tEnter Correct Option\t   or \t press Q to quit.\n\t\t\t...").lower()

    # Check if the answer is correct
    if useranswer == right_answer:
        # Update the prize amount based on the number of correct answers
        if count == 1:
            amount = 10000
        elif count == 2:
            amount = 20000
        elif count == 3:
            amount = 50000
        elif count == 4:
            amount = 100000
        elif count == 5:
            amount = 400000
        elif count == 6:
            amount = 800000
        elif count == 7:
            amount = 1600000
        elif count == 8:
            amount = 3200000
        elif count == 9:
            amount = 6400000
        elif count == 10:
            amount = 10000000  # Maximum prize for answering all questions correctly
            print("\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n")
            print("*"*120)
            print("\t\t\t !!!!!!!!!! Congratulations! !!!!!!!!!!")
            print("\t\t\t||||||||||| You Won The Game |||||||||||")
            print("*"*120)
            print("\n\n\t\t You Won Rs. ", amount)
            print()
            break

        # Display message for correct answer and proceed to the next question
        print("\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n")
        print("*"*120)
        print("\t\t\t !!!!!!!!!! Congratulations! !!!!!!!!!!")
        print("\t\t\t||||||||||||| Right Answer ||||||||||||||")
        print("*"*120)
        count += 1

    # If the player chooses to quit
    elif useranswer == "q":
        print("\t\tCorrect answer was "+right_answer.upper())
        print("\n\n\t\t You Won Rs. ", amount)
        break

    # If the player provides a wrong answer
    else:
        print("*"*120)
        print("\t\t\t\t\t\t\t\t Wrong Answer")
        print("\t\t\t\t\t\t\t\t Correct answer is "+right_answer.upper())
        print("*"*120)
        print("\n\n\t\t \t\t\t\t\t\t You Won Rs. ", amount)
        print("*"*120)
        break
