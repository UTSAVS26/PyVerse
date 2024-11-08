import pygame
import sys

# Initialize Pygame
pygame.init()

# Screen dimensions
WIDTH, HEIGHT = 800, 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Space Explorer")

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
BLUE = (0, 0, 255)
GREEN = (0, 255, 0)  # Color for correct answer
RED = (255, 0, 0)    # Color for incorrect answer

# Load background image
background = pygame.image.load("jump3.jpg")  # Change to your background image file
background = pygame.transform.scale(background, (WIDTH, HEIGHT))  # Scale it to fit the screen

# Font
font = pygame.font.Font(None, 36)

# Load sound effects
correct_sound = pygame.mixer.Sound("correct.wav")
incorrect_sound = pygame.mixer.Sound("incorrect.wav")

# Questions and answers
questions = [
    {
        "question": "What is the largest planet in our solar system?",
        "options": ["Earth", "Jupiter", "Mars", "Saturn"],
        "answer": "Jupiter",
    },
    {
        "question": "What planet is known as the Red Planet?",
        "options": ["Venus", "Mars", "Mercury", "Jupiter"],
        "answer": "Mars",
    },
    {
        "question": "What is the name of our galaxy?",
        "options": ["Andromeda", "Milky Way", "Whirlpool", "Sombrero"],
        "answer": "Milky Way",
    },
    {
        "question": "How many planets are in our solar system?",
        "options": ["7", "8", "9", "10"],
        "answer": "8",
    },
]

# Game state
current_question = 0
score = 0
time_left = 30  # 30 seconds for each question
leaderboard = []  # List to store scores

def display_question(question_data):
    """Displays the current question and answer options centered on the screen."""
    screen.blit(background, (0, 0))  # Draw the background image
    question_text = font.render(question_data["question"], True, WHITE)
    
    # Center the question text
    question_rect = question_text.get_rect(center=(WIDTH // 2, HEIGHT // 2 - 50))
    screen.blit(question_text, question_rect)
    
    # Display answer options centered
    for i, option in enumerate(question_data["options"]):
        option_text = font.render(f"{i + 1}. {option}", True, WHITE)  # Set text color to white
        option_rect = option_text.get_rect(center=(WIDTH // 2, HEIGHT // 2 + i * 40))
        screen.blit(option_text, option_rect)

    # Display timer
    timer_text = font.render(f"Time left: {time_left}", True, WHITE)  # Set timer text color to white
    timer_rect = timer_text.get_rect(topright=(WIDTH - 10, 10))
    screen.blit(timer_text, timer_rect)

    pygame.display.flip()

def show_feedback(correct):
    """Displays feedback on the selection."""
    feedback_text = "Correct!" if correct else "Incorrect!"
    feedback_color = GREEN if correct else RED  # Set color based on correctness
    feedback_surface = font.render(feedback_text, True, feedback_color)  # Set feedback text color
    feedback_rect = feedback_surface.get_rect(center=(WIDTH // 2, HEIGHT // 2 + 100))
    screen.blit(feedback_surface, feedback_rect)
    pygame.display.flip()
    pygame.time.wait(1000)  # Show feedback for 1 second

def show_score():
    """Displays the final score centered on the screen."""
    screen.blit(background, (0, 0))
    score_text = font.render(f"Your Score: {score} / {len(questions)}", True, WHITE)  # Set score text color to white
    score_rect = score_text.get_rect(center=(WIDTH // 2, HEIGHT // 2))
    screen.blit(score_text, score_rect)
    pygame.display.flip()
    pygame.time.wait(3000)  # Wait for 3 seconds

def show_leaderboard():
    """Displays the leaderboard."""
    screen.blit(background, (0, 0))
    leaderboard_text = font.render("Leaderboard", True, WHITE)  # Set leaderboard title color to white
    leaderboard_rect = leaderboard_text.get_rect(center=(WIDTH // 2, 50))
    screen.blit(leaderboard_text, leaderboard_rect)

    for i, score in enumerate(leaderboard):
        score_text = font.render(f"{i + 1}. Score: {score}", True, WHITE)  # Set score text color to white
        score_rect = score_text.get_rect(center=(WIDTH // 2, 100 + i * 30))
        screen.blit(score_text, score_rect)

    pygame.display.flip()
    pygame.time.wait(5000)  # Wait for 5 seconds before returning to main menu

def show_final_screen():
    """Displays the final screen with options to retake the quiz or quit."""
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:  # Restart quiz
                    main()
                elif event.key == pygame.K_l:  # Show leaderboard
                    show_leaderboard()
                elif event.key == pygame.K_q:  # Quit
                    pygame.quit()
                    sys.exit()

        # Display final options
        screen.blit(background, (0, 0))
        final_text = font.render("Press 'R' to retake, 'L' for leaderboard or 'Q' to quit.", True, WHITE)  # Set final options text color to white
        final_rect = final_text.get_rect(center=(WIDTH // 2, HEIGHT // 2))
        screen.blit(final_text, final_rect)
        pygame.display.flip()

def main():
    global current_question, score, time_left
    current_question = 0
    score = 0
    time_left = 30  # Reset timer for the new game

    # Main game loop
    pygame.time.set_timer(pygame.USEREVENT + 1, 1000)  # Timer event every second
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

            if event.type == pygame.USEREVENT + 1:  # Timer event
                time_left -= 1
                if time_left <= 0:
                    current_question += 1
                    time_left = 30  # Reset timer for the next question

            if event.type == pygame.KEYDOWN:
                if event.key in [pygame.K_1, pygame.K_2, pygame.K_3, pygame.K_4]:
                    selected_option = event.unicode  
                    selected_index = int(selected_option) - 1

                    if selected_index >= 0 and selected_index < len(questions[current_question]["options"]):
                        correct = questions[current_question]["options"][selected_index] == questions[current_question]["answer"]
                        if correct:
                            score += 1  
                            correct_sound.play()  # Play correct sound
                        else:
                            incorrect_sound.play()  # Play incorrect sound

                        show_feedback(correct)
                        current_question += 1  

                        if current_question >= len(questions):
                            leaderboard.append(score)  # Add score to leaderboard
                            show_score()
                            show_final_screen()  # Show the final screen

        if current_question < len(questions):
            display_question(questions[current_question])
        else:
            show_score()

if __name__ == "__main__":
    main()

