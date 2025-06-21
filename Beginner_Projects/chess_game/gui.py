import pygame
from board import Board
from game import Game

def load_piece_images():
    """Load and scale all piece images to 60x60 and return a dictionary."""
    size = (60, 60)
    return {
        'white_pawn': pygame.transform.scale(pygame.image.load('pieces/wP.png'), size),
        'black_pawn': pygame.transform.scale(pygame.image.load('pieces/bP.png'), size),
        'white_rook': pygame.transform.scale(pygame.image.load('pieces/wR.png'), size),
        'black_rook': pygame.transform.scale(pygame.image.load('pieces/bR.png'), size),
        'white_knight': pygame.transform.scale(pygame.image.load('pieces/wN.png'), size),
        'black_knight': pygame.transform.scale(pygame.image.load('pieces/bN.png'), size),
        'white_bishop': pygame.transform.scale(pygame.image.load('pieces/wB.png'), size),
        'black_bishop': pygame.transform.scale(pygame.image.load('pieces/bB.png'), size),
        'white_queen': pygame.transform.scale(pygame.image.load('pieces/wQ.png'), size),
        'black_queen': pygame.transform.scale(pygame.image.load('pieces/bQ.png'), size),
        'white_king': pygame.transform.scale(pygame.image.load('pieces/wK.png'), size),
        'black_king': pygame.transform.scale(pygame.image.load('pieces/bK.png'), size),
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
    screen, _, images = initialize_game()
    game = Game(color='white')
    selected = None
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.MOUSEBUTTONDOWN:
                x, y = event.pos
                row, col = y // 60, x // 60
                if selected is None:
                    piece = game.board.board[row][col]
                    if piece is not None and piece.color == game.current_color:
                        selected = (row, col)
                else:
                    moved = game.board.move_pieces(selected, (row, col))
                    if moved:
                        # Check for check, checkmate, or stalemate after a move
                        if game.is_checkmate(game.current_color):
                            print(f"Checkmate! {game.current_color} loses.")
                            running = False
                        elif game.is_in_check(game.current_color):
                            print(f"{game.current_color} is in check!")
                        game.switch_turns()
                    selected = None

        draw_board(screen, game.board.board, images)
    pygame.quit()

if __name__ == "__main__":
    main()