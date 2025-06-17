import time
import random
import sys
import os

sentences=[
    "Typing speed tests are fun and useful.",
    "Did you know that the shortest complete sentence in the English language is 'I am'?",
    "Another interesting fact is that 'rhythm' is the longest English word without a vowel.",
    "The word 'set' has more definitions than any other word in the English dictionary, boasting hundreds of meanings.",
    "These linguistic quirks highlight the rich and often surprising nature of the English vocabulary." 
]

def countDown(seconds=3):
    print("\n Get Ready!!")
    for i in range (seconds,0,-1):
        print(f"{i}...")
        time.sleep(1)
    print("GO!!")

def clear_console():
    os.system('cls' if os.name == 'nt' else 'clear')

def typing_speed_test():
    sentence=random.choice(sentences)
    print("\n Type the following sentence:\n")
    print(f" {sentence}\n")
    input("Press Enter to start...")

    clear_console()
    countDown()

    start_time=end_time-start_time
    time_in_minutes=elapsed_time/60
    words_typed=len(user_input.split())
    wpm=words_typed/time_in_minutes

    correct_chars=0
    total_chars=len(sentence)
    for i in range (min(len(user_input),total_chars)):
        if user_input[i]==sentence[i]:
            correct_chars+=1
    accuracy=(correct_chars/total_chars)*100
    print("\n Typing Report")
    print("------------------------")
    print("Original Sentence: ", sentence)
    print("Your input : ", user_input)
    print("Time Taken : {:.2f} seconds".format(elapsed_time))
    print("📊 Typing Speed      : {:.2f} WPM".format(wpm))
    print("🎯 Accuracy          : {:.2f}%".format(accuracy))
    print("🔢 Correct Characters: {}/{}".format(correct_chars, total_chars))

if __name__=="__main__":
    typing_speed_test()

