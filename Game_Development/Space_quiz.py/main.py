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

# Load background image
background = pygame.image.load("jump3.jpg")  # Change to your background image file
background = pygame.transform.scale(background, (WIDTH, HEIGHT))  # Scale it to fit the screen

# Font
font = pygame.font.Font(None, 36)

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

def display_question(question_data):
    """Displays the current question and answer options centered on the screen."""
    screen.blit(background, (0, 0))  # Draw the background image
    question_text = font.render(question_data["question"], True, WHITE)
    
    # Center the question text
    question_rect = question_text.get_rect(center=(WIDTH // 2, HEIGHT // 2 - 50))
    screen.blit(question_text, question_rect)
    
    # Display answer options centered
    for i, option in enumerate(question_data["options"]):
        option_text = font.render(f"{i + 1}. {option}", True, WHITE)
        option_rect = option_text.get_rect(center=(WIDTH // 2, HEIGHT // 2 + i * 40))
        screen.blit(option_text, option_rect)

    pygame.display.flip()

def show_score():
    """Displays the final score centered on the screen."""
    screen.blit(background, (0, 0))
    score_text = font.render(f"Your Score: {score} / {len(questions)}", True, BLUE)
    score_rect = score_text.get_rect(center=(WIDTH // 2, HEIGHT // 2))
    screen.blit(score_text, score_rect)
    pygame.display.flip()
    pygame.time.wait(3000)  # Wait for 3 seconds

# Main game loop
while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()

        if event.type == pygame.KEYDOWN:
            if event.key in [pygame.K_1, pygame.K_2, pygame.K_3, pygame.K_4]:
                selected_option = event.unicode  
                selected_index = int(selected_option) - 1

                if selected_index >= 0 and selected_index < len(questions[current_question]["options"]):
                    if questions[current_question]["options"][selected_index] == questions[current_question]["answer"]:
                        score += 1  
                    current_question += 1  

                    if current_question >= len(questions):
                        show_score()
                        pygame.quit()
                        sys.exit()

    if current_question < len(questions):
        display_question(questions[current_question])
    else:
        show_score()
