import pygame
from colorama import init, Fore, Style

# Initialize Pygame and Colorama
pygame.init()
init(autoreset=True)

# Screen settings
WIDTH, HEIGHT = 800, 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Tower of Hanoi")

# Colors
BACKGROUND_COLOR = (30, 30, 60)
ROD_COLOR = (220, 220, 220)
DISK_COLORS = [(240, 128, 128), (250, 200, 200), (100, 149, 237), (152, 251, 152), (255, 105, 180)]

# Font
font = pygame.font.Font(None, 36)

# Rod positions and disk settings
rod_x_positions = [WIDTH // 4, WIDTH // 2, 3 * WIDTH // 4]
rod_y_position = HEIGHT - 50
rod_width, rod_height = 10, 250
disk_height = 20
disks = []

# Game settings
num_disks = 5
rods = [[], [], []]
for i in range(num_disks, 0, -1):
    disks.append({
        'width': i * 30,
        'color': DISK_COLORS[i % len(DISK_COLORS)],
        'position': (rod_x_positions[0], rod_y_position - disk_height * len(rods[0]))
    })
    rods[0].append(disks[-1])

# Draw rods and disks
def draw_rods_and_disks():
    screen.fill(BACKGROUND_COLOR)
    for i, x in enumerate(rod_x_positions):
        pygame.draw.rect(screen, ROD_COLOR, (x - rod_width // 2, rod_y_position - rod_height, rod_width, rod_height))
        for j, disk in enumerate(rods[i]):
            disk_x = x - disk['width'] // 2
            disk_y = rod_y_position - disk_height * (j + 1)
            pygame.draw.rect(screen, disk['color'], (disk_x, disk_y, disk['width'], disk_height))

# Console instructions
print(Fore.CYAN + "Welcome to the Tower of Hanoi Game!" + Style.RESET_ALL)
print(Fore.YELLOW + "Move all disks from the first rod to the third rod following the rules:")
print("- Move one disk at a time.")
print("- Only smaller disks can be placed on larger disks.\n" + Style.RESET_ALL)

# Main game loop
running = True
selected_disk = None
selected_rod = None

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.MOUSEBUTTONDOWN:
            mouse_x, mouse_y = event.pos
            for i, rod in enumerate(rods):
                if rod and (rod_x_positions[i] - rod[0]['width'] // 2 < mouse_x < rod_x_positions[i] + rod[0]['width'] // 2):
                    selected_disk = rod.pop()
                    selected_rod = i
                    break
        elif event.type == pygame.MOUSEBUTTONUP and selected_disk:
            mouse_x, mouse_y = event.pos
            for i, x in enumerate(rod_x_positions):
                if abs(mouse_x - x) < 50:  # Drop zone for rods
                    if not rods[i] or selected_disk['width'] < rods[i][-1]['width']:
                        rods[i].append(selected_disk)
                        selected_disk = None
                        break
                    else:
                        print(Fore.RED + "Invalid move! You can't place a larger disk on a smaller one." + Style.RESET_ALL)
                        rods[selected_rod].append(selected_disk)
                        selected_disk = None
                        break
            if selected_disk:
                rods[selected_rod].append(selected_disk)
                selected_disk = None

    draw_rods_and_disks()
    pygame.display.flip()

# Exit
pygame.quit()
