import pygame
from board import Board

def load_piece_images():
    """Load all piece images and return a dictionary."""
    return {
        'white_pawn': pygame.image.load('images/white_pawn.png'),
        'black_pawn': pygame.image.load('images/black_pawn.png'),
        'white_rook': pygame.image.load('images/white_rook.png'),
        'black_rook': pygame.image.load('images/black_rook.png'),
        'white_knight': pygame.image.load('images/white_knight.png'),
        'black_knight': pygame.image.load('images/black_knight.png'),
        'white_bishop': pygame.image.load('images/white_bishop.png'),
        'black_bishop': pygame.image.load('images/black_bishop.png'),
        'white_queen': pygame.image.load('images/white_queen.png'),
        'black_queen': pygame.image.load('images/black_queen.png'),
        'white_king': pygame.image.load('images/white_king.png'),
        'black_king': pygame.image.load('images/black_king.png'),
    }

def get_piece_image(piece, images):
    if piece is None:
        return None
    key = f"{piece.color}_{piece.__class__.__name__.lower()}"
    return images.get(key)

def draw_board(screen, board, images):
    """Draw the chess board and pieces on the screen."""
    colors = [(240, 217, 181), (181, 136, 99)]  # Light and dark squares
    square_size = 60

    for row in range(8):
        for col in range(8):
            color = colors[(row + col) % 2]
            pygame.draw.rect(screen, color, (col * square_size, row * square_size, square_size, square_size))
            piece = board[row][col]
            img = get_piece_image(piece, images)
            if img:
                screen.blit(img, (col * square_size, row * square_size))
    pygame.display.flip()

def initialize_game():
    pygame.init()
    screen = pygame.display.set_mode((480, 480))
    pygame.display.set_caption("Chess Game")
    images = load_piece_images()
    board_obj = Board()
    return screen, board_obj, images

def main():
    screen, board_obj, images = initialize_game()
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        draw_board(screen, board_obj.board, images)
    pygame.quit()