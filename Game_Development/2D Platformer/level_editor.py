import pygame
import pickle
from os import path


pygame.init()

clock = pygame.time.Clock()
fps = 60

# game window
tile_size = 35
cols = 20
margin = 100
screen_width = tile_size * cols
screen_height = (tile_size * cols) + margin
level = 0

screen = pygame.display.set_mode((screen_width, screen_height))
pygame.display.set_caption('Level Editor')


# load images
#sun_img = pygame.image.load('sun.png')
#sun_img = pygame.transform.scale(sun_img, (tile_size, tile_size))
bg_img = pygame.image.load(f'bg{level}.png').convert_alpha()
bg_img = pygame.transform.scale(bg_img, (screen_width, screen_height - margin))
# level 0 ,1,2
dirt_img = pygame.image.load('grassCenter.png')
grass_img = pygame.image.load('grassMid.png')
platform_x_img = pygame.image.load('grassHalfMid.png')
platform_y_img = pygame.image.load('grassHalfMidcopy.png')
# level 3
sand_img = pygame.image.load('sandCenter.png')
sand2_img = pygame.image.load('sandMid.png')
sand_x_img = pygame.image.load('sandHalfMid.png')
sand_y_img = pygame.image.load('sandHalfMidcopy.png')
# level 4
cake_img = pygame.image.load('cakeCenter.png')
cake2_img = pygame.image.load('cakeMid.png')
cake_x_img = pygame.image.load('cakeHalfAltMid.png')
cake_y_img = pygame.image.load('cakeHalfAltMidcopy.png')
# level 5
choco_img = pygame.image.load('chocoCenter.png')
choco2_img = pygame.image.load('chocoMid.png')
choco_x_img = pygame.image.load('chocoHalfAltMid.png')
choco_y_img = pygame.image.load('chocoHalfAltMidcopy.png')
# level 6
stone_img = pygame.image.load('stoneCenter.png')
stone2_img = pygame.image.load('stoneMid.png')
stone_x_img = pygame.image.load('stoneHalfMid.png')
stone_y_img = pygame.image.load('stoneHalfMidcopy.png')
# level 7
tundra_img = pygame.image.load('tundraCenter.png')
tundra2_img = pygame.image.load('tundraMid.png')
tundra_x_img = pygame.image.load('tundraHalfMid.png')
tundra_y_img = pygame.image.load('tundraHalfMidcopy.png')


blob_img = pygame.image.load('blockerMad.png')
lava_img = pygame.image.load('liquidLavaTop_mid.png')
coin_img = pygame.image.load('coin.png')
exit_img = pygame.image.load('gate.png')
save_img = pygame.image.load('save_btn.png')
load_img = pygame.image.load('load_btn.png')


# define game variables
clicked = False
level = 0

# define colours
white = (255, 255, 255)
green = (144, 201, 120)

font = pygame.font.SysFont('Futura', 24)

# create empty tile list
world_data = []
for row in range(20):
    r = [0] * 20
    world_data.append(r)

# create boundary
for tile in range(0, 20):
    world_data[19][tile] = 2
    world_data[0][tile] = 1
    world_data[tile][0] = 1
    world_data[tile][19] = 1

# function for outputting text onto the screen


def draw_text(text, font, text_col, x, y):
    img = font.render(text, True, text_col)
    screen.blit(img, (x, y))


def draw_grid():
    for c in range(21):
        # vertical lines
        pygame.draw.line(screen, white, (c * tile_size, 0),
                         (c * tile_size, screen_height - margin))
        # horizontal lines
        pygame.draw.line(screen, white, (0, c * tile_size),
                         (screen_width, c * tile_size))


def draw_world():
    for row in range(20):
        for col in range(20):
            if world_data[row][col] > 0:
                if world_data[row][col] == 1:
                    # dirt blocks
                    img = pygame.transform.scale(
                        dirt_img, (tile_size, tile_size))
                    screen.blit(img, (col * tile_size, row * tile_size))
                if world_data[row][col] == 2:
                    # grass blocks
                    img = pygame.transform.scale(
                        grass_img, (tile_size, tile_size))
                    screen.blit(img, (col * tile_size, row * tile_size))
                if world_data[row][col] == 3:
                    # enemy blocks
                    img = pygame.transform.scale(
                        blob_img, (tile_size, int(tile_size)))
                    screen.blit(img, (col * tile_size, row *
                                      tile_size + (tile_size*0.01)))
                if world_data[row][col] == 4:
                    # horizontally moving platform
                    img = pygame.transform.scale(
                        platform_x_img, (tile_size, tile_size))
                    screen.blit(img, (col * tile_size, row * tile_size))
                if world_data[row][col] == 5:
                    # vertically moving platform
                    img = pygame.transform.scale(
                        platform_y_img, (tile_size, tile_size))
                    screen.blit(img, (col * tile_size, row * tile_size))
                if world_data[row][col] == 6:
                    # lava
                    img = pygame.transform.scale(
                        lava_img, (tile_size, tile_size - tile_size//4))
                    screen.blit(img, (col * tile_size, row *
                                      tile_size + (tile_size // 4)))
                if world_data[row][col] == 7:
                    # coin
                    img = pygame.transform.scale(
                        coin_img, (tile_size-tile_size//4, tile_size-tile_size//4))
                    screen.blit(img, (col * tile_size + (tile_size//8),
                                      row * tile_size + (tile_size//8)))
                if world_data[row][col] == 8:
                    # exit
                    img = pygame.transform.scale(
                        exit_img, (tile_size, int(tile_size*1.5 + (tile_size//2))))
                    screen.blit(img, (col * tile_size, row *
                                      tile_size - (int(tile_size//1.05))))
                if world_data[row][col] == 9:
                    # sand blocks
                    img = pygame.transform.scale(
                        sand_img, (tile_size, tile_size))
                    screen.blit(img, (col * tile_size, row * tile_size))
                if world_data[row][col] == 10:
                    # sand2 blocks
                    img = pygame.transform.scale(
                        sand2_img, (tile_size, tile_size))
                    screen.blit(img, (col * tile_size, row * tile_size))
                if world_data[row][col] == 11:
                    # horizontally moving platform sand
                    img = pygame.transform.scale(
                        sand_x_img, (tile_size, tile_size))
                    screen.blit(img, (col * tile_size, row * tile_size))
                if world_data[row][col] == 12:
                    # vertically moving platform sand
                    img = pygame.transform.scale(
                        sand_y_img, (tile_size, tile_size))
                    screen.blit(img, (col * tile_size, row * tile_size))
                if world_data[row][col] == 13:
                    # cake blocks
                    img = pygame.transform.scale(
                        cake_img, (tile_size, tile_size))
                    screen.blit(img, (col * tile_size, row * tile_size))
                if world_data[row][col] == 14:
                    # cake2 blocks
                    img = pygame.transform.scale(
                        cake2_img, (tile_size, tile_size))
                    screen.blit(img, (col * tile_size, row * tile_size))
                if world_data[row][col] == 15:
                    # horizontally moving platform cake
                    img = pygame.transform.scale(
                        cake_x_img, (tile_size, tile_size))
                    screen.blit(img, (col * tile_size, row * tile_size))
                if world_data[row][col] == 16:
                    # vertically moving platform cake
                    img = pygame.transform.scale(
                        cake_y_img, (tile_size, tile_size))
                    screen.blit(img, (col * tile_size, row * tile_size))
                if world_data[row][col] == 17:
                    # choco blocks
                    img = pygame.transform.scale(
                        choco_img, (tile_size, tile_size))
                    screen.blit(img, (col * tile_size, row * tile_size))
                if world_data[row][col] == 18:
                    # choco2 blocks
                    img = pygame.transform.scale(
                        choco2_img, (tile_size, tile_size))
                    screen.blit(img, (col * tile_size, row * tile_size))
                if world_data[row][col] == 19:
                    # horizontally moving platform choco
                    img = pygame.transform.scale(
                        choco_x_img, (tile_size, tile_size))
                    screen.blit(img, (col * tile_size, row * tile_size))
                if world_data[row][col] == 20:
                    # vertically moving platform choco
                    img = pygame.transform.scale(
                        choco_y_img, (tile_size, tile_size))
                    screen.blit(img, (col * tile_size, row * tile_size))
                if world_data[row][col] == 21:
                    # stone blocks
                    img = pygame.transform.scale(
                        stone_img, (tile_size, tile_size))
                    screen.blit(img, (col * tile_size, row * tile_size))
                if world_data[row][col] == 22:
                    # stone2 blocks
                    img = pygame.transform.scale(
                        stone2_img, (tile_size, tile_size))
                    screen.blit(img, (col * tile_size, row * tile_size))
                if world_data[row][col] == 23:
                    # horizontally moving platform stone
                    img = pygame.transform.scale(
                        stone_x_img, (tile_size, tile_size))
                    screen.blit(img, (col * tile_size, row * tile_size))
                if world_data[row][col] == 24:
                    # vertically moving platform stone
                    img = pygame.transform.scale(
                        stone_y_img, (tile_size, tile_size))
                    screen.blit(img, (col * tile_size, row * tile_size))
                if world_data[row][col] == 25:
                    # tundra blocks
                    img = pygame.transform.scale(
                        tundra_img, (tile_size, tile_size))
                    screen.blit(img, (col * tile_size, row * tile_size))
                if world_data[row][col] == 26:
                    # tundra2 blocks
                    img = pygame.transform.scale(
                        tundra2_img, (tile_size, tile_size))
                    screen.blit(img, (col * tile_size, row * tile_size))
                if world_data[row][col] == 27:
                    # horizontally moving platform tundra
                    img = pygame.transform.scale(
                        tundra_x_img, (tile_size, tile_size))
                    screen.blit(img, (col * tile_size, row * tile_size))
                if world_data[row][col] == 28:
                    # vertically moving platform tundra
                    img = pygame.transform.scale(
                        tundra_y_img, (tile_size, tile_size))
                    screen.blit(img, (col * tile_size, row * tile_size))


class Button():
    def __init__(self, x, y, image):
        self.image = image
        self.rect = self.image.get_rect()
        self.rect.topleft = (x, y)
        self.clicked = False

    def draw(self):
        action = False

        # get mouse position
        pos = pygame.mouse.get_pos()

        # check mouseover and clicked conditions
        if self.rect.collidepoint(pos):
            if pygame.mouse.get_pressed()[0] == 1 and self.clicked == False:
                action = True
                self.clicked = True

        if pygame.mouse.get_pressed()[0] == 0:
            self.clicked = False

        # draw button
        screen.blit(self.image, (self.rect.x, self.rect.y))

        return action


# create load and save buttons
save_button = Button(screen_width // 2 - 150, screen_height - 80, save_img)
load_button = Button(screen_width // 2 + 50, screen_height - 80, load_img)

# main game loop
run = True
while run:

    clock.tick(fps)

    # draw background
    screen.fill(green)
    screen.blit(bg_img, (0, 0))
    #screen.blit(sun_img, (tile_size * 2, tile_size * 2))

    # load and save level
    if save_button.draw():
        # save level data
        pickle_out = open(f'level{level}_data', 'wb')
        pickle.dump(world_data, pickle_out)
        pickle_out.close()
    if load_button.draw():
        # load in level data
        if path.exists(f'level{level}_data'):
            pickle_in = open(f'level{level}_data', 'rb')
            world_data = pickle.load(pickle_in)

    # show the grid and draw the level tiles
    draw_grid()
    draw_world()

    # text showing current level
    draw_text(f'Level: {level}', font, white, tile_size, screen_height - 60)
    draw_text('Press UP or DOWN to change level', font,
              white, tile_size, screen_height - 40)

    # event handler
    for event in pygame.event.get():
        # quit game
        if event.type == pygame.QUIT:
            run = False
        # mouseclicks to change tiles
        if event.type == pygame.MOUSEBUTTONDOWN and clicked == False:
            clicked = True
            pos = pygame.mouse.get_pos()
            x = pos[0] // tile_size
            y = pos[1] // tile_size
            # check that the coordinates are within the tile area
            if x < 20 and y < 20:
                # update tile value
                if pygame.mouse.get_pressed()[0] == 1:
                    world_data[y][x] += 1
                    if world_data[y][x] > 28:
                        world_data[y][x] = 0
                elif pygame.mouse.get_pressed()[2] == 1:
                    world_data[y][x] -= 1
                    if world_data[y][x] < 0:
                        world_data[y][x] = 28
        if event.type == pygame.MOUSEBUTTONUP:
            clicked = False
        # up and down key presses to change level number
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_UP:
                level += 1
                bg_img = pygame.image.load(f'bg{level}.png').convert_alpha()
            elif event.key == pygame.K_DOWN and level > 0:
                level -= 1
                bg_img = pygame.image.load(f'bg{level}.png').convert_alpha()

    # update game display window
    pygame.display.update()

pygame.quit()
