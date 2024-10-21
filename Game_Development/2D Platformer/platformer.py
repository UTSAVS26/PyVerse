import pygame
from pygame.locals import *
from pygame import mixer
import pickle
from os import path

# initializing pygame so dat it can be put to use/pygame starts working
pygame.mixer.pre_init(44100, -16, 2, 512)
mixer.init()
pygame.init()

clock = pygame.time.Clock()
fps = 62

screen_width = 700
screen_height = 700

screen = pygame.display.set_mode((screen_width, screen_height))
pygame.display.set_caption('Platformer')


# define font
font = pygame.font.SysFont('Bauhaus 93', 70)
font_score = pygame.font.SysFont('Bauhaus 93', 30)

# define game variables
tile_size = 35
game_over = 0
main_menu = True
level = 0
max_levels = 7
score = 0

# define colours
white = (255, 255, 255)
blue = (0, 0, 255)


# load images
#cloud_img = pygame.image.load("cloud1.png").convert_alpha()
bg_img = pygame.image.load("./images/bg0.png").convert_alpha()
restart_img = pygame.image.load("./images/restart_btn.png").convert_alpha()
#restart_img = pygame.transform.scale(restartimg, (tile_size*6, tile_size*3))
playimg = pygame.image.load("./images/start_btn.png").convert_alpha()
play_img = pygame.transform.scale(playimg, (tile_size*6, tile_size*3))
exitimg = pygame.image.load("./images/exit_btn.png").convert_alpha()
exit_img = pygame.transform.scale(exitimg, (tile_size*6, tile_size*3))


# load sounds[making the volume 50% of original by using 0.5]
pygame.mixer.music.load('./audio/music.wav')
pygame.mixer.music.play(-1, 0.0, 5000)
coin_fx = pygame.mixer.Sound('./audio/coin.wav')
coin_fx.set_volume(0.5)
jump_fx = pygame.mixer.Sound('./audio/jump.wav')
jump_fx.set_volume(0.5)
game_over_fx = pygame.mixer.Sound('./audio/game_over.wav')
game_over_fx.set_volume(0.5)
game_won = pygame.mixer.Sound('./audio/tada.wav')


# pygame first converts the text into img (using render) then blit it onto screen
def draw_text(text, font, text_col, x, y):
    img = font.render(text, True, text_col)
    screen.blit(img, (x, y))


def reset_level(level):
    player.reset(40, screen_height-110)
    blob_group.empty()
    platform_group.empty()
    lava_group.empty()
    coin_group.empty()
    exit_group.empty()

    # dummy coin
    score_coin = Coin(tile_size//2, tile_size//2)
    coin_group.add(score_coin)

    # load in level data and create world
    if path.exists(f"C:\Downloads\level{level}_data"):
        pickle_in = open(f'C:\Downloads\level{level}_data', 'rb')
        world_data = pickle.load(pickle_in)
    world = World(world_data)

    return world


class Button():
    def __init__(self, x, y, image):
        self.image = image
        self.rect = self.image.get_rect()
        self.rect.x = x
        self.rect.y = y
        self.clicked = False

    def draw(self):
        action = False

        # get mouse position (as a point)
        pos = pygame.mouse.get_pos()

        # check is mouse is over the button and clicked conditions
        if self.rect.collidepoint(pos):
            # 0 indicates left mouse key and 1 indicates its been clicked once
            if pygame.mouse.get_pressed()[0] == 1 and self.clicked == False:
                action = True
                self.clicked = True

        # to unclick
        if pygame.mouse.get_pressed()[0] == 0:
            self.clicked = False

        # draw button
        screen.blit(self.image, self.rect)
        return action


class Player():
    def __init__(self, x, y):
        self.reset(x, y)

    def update(self, game_over):
        dx = 0
        dy = 0
        # to slow down the animation [5 iterations pass before the index gets +1]
        walk_cooldown = 3
        col_thresh = 20

        if game_over == 0:
            # get keypresses
            key = pygame.key.get_pressed()
            if key[pygame.K_SPACE] == True and self.jumped == False and self.in_air == False:
                jump_fx.play()
                self.vel_y = -14
                self.jumped = True
            if key[pygame.K_SPACE] == False:
                self.jumped = False
            if key[pygame.K_LEFT]:
                dx -= 4
                self.counter += 1
                self.direction = -1
            if key[pygame.K_RIGHT]:
                dx += 4
                self.counter += 1
                self.direction = 1
            if key[pygame.K_LEFT] == False and key[pygame.K_RIGHT] == False:
                self.counter = 0
                self.index = 0
                if self.direction == 1:
                    self.image = self.images_right[self.index]
                if self.direction == -1:
                    self.image = self.images_left[self.index]

            # handle animation
            if self.counter > walk_cooldown:
                self.counter = 0
                self.index += 1
                # to reset animation
                if self.index >= len(self.images_right):
                    self.index = 0
                if self.direction == 1:
                    self.image = self.images_right[self.index]
                if self.direction == -1:
                    self.image = self.images_left[self.index]

            # add gravity
            self.vel_y += 1
            if self.vel_y > 10:
                self.vel_y = 10

            dy += self.vel_y

            # check for collision (the player hasnt moved with updates dx and dy so we r checking
            # collision before dat happens so he doesnt run into tile and we have made our own rect in collisionrect
            # so the overlapping condition preventss it from colliding)
            self.in_air = True
            for tile in world.tile_list:
                # check for collision in x direction
                if tile[1].colliderect(self.rect.x+dx, self.rect.y, self.width, self.height):
                    dx = 0
                # check for collision in y direction
                if tile[1].colliderect(self.rect.x, self.rect.y+dy, self.width, self.height):
                    # check if below ground/tile i.e. jumping
                    if self.vel_y < 0:
                        dy = tile[1].bottom-self.rect.top
                        self.vel_y = 0
                    # check if above ground/tile i.e. falling
                    elif self.vel_y >= 0:
                        dy = tile[1].top-self.rect.bottom
                        self.vel_y = 0
                        self.in_air = False

            # check for collision with enemies {if set to True then it will delete our enemy sprite which we dont want}
            if pygame.sprite.spritecollide(self, blob_group, False):
                game_over = -1
                game_over_fx.play()

            # check for collision with lava
            if pygame.sprite.spritecollide(self, lava_group, False):
                game_over = -1
                game_over_fx.play()

            # check for collision with exit
            if pygame.sprite.spritecollide(self, exit_group, False):
                game_over = 1

            # check for collision with platforms
            for platform in platform_group:
                # collision in x direction
                if platform.rect.colliderect(self.rect.x+dx, self.rect.y, self.width, self.height):
                    dx = 0
                # collision in x direction
                if platform.rect.colliderect(self.rect.x, self.rect.y+dy, self.width, self.height):
                    # check if below platform {first line of if checks if theres a collision b/w players head and platforms bottoms}
                    if abs((self.rect.top+dy)-platform.rect.bottom) < col_thresh:
                        self.vel_y = 0
                        dy = platform.rect.bottom-self.rect.top

                    # check if above platform {-1 is added so dat he can move freely on platform withut getting unnecessary collisions}
                    if abs((self.rect.bottom+dy)-platform.rect.top) < col_thresh:
                        self.rect.bottom = platform.rect.top-1
                        self.in_air = False
                        dy = 0
                    # move sideways with platform
                    if platform.move_x != 0:
                        self.rect.x += platform.move_direction

            # update player coordinates
            self.rect.x += dx
            self.rect.y += dy

        elif game_over == -1:
            self.image = self.dead_image
            draw_text('GAME OVER!', font, blue,
                      (screen_width//2)-140, (screen_height//2)-55)
            # if self.rect.y > 200:
            self.rect.y -= 5

        # draw player onto screen
        screen.blit(self.image, self.rect)

        return game_over

    def reset(self, x, y):
        self.images_right = []
        self.images_left = []
        self.index = 0
        self.counter = 0
        for num in range(1, 12):
            img_right = pygame.image.load(f"./images/p1_walk{num}.png").convert_alpha()
            img_right = pygame.transform.scale(img_right, (32, 68))
            img_left = pygame.transform.flip(img_right, True, False)
            self.images_right.append(img_right)
            self.images_left.append(img_left)
        self.dead_image = pygame.image.load('./images/ghost_dead.png')
        self.dead_image = pygame.transform.scale(
            self.dead_image, (tile_size, tile_size*2))
        self.image = self.images_right[self.index]
        self.rect = self.image.get_rect()
        self.rect.x = x
        self.rect.y = y
        self.width = self.image.get_width()
        self.height = self.image.get_height()
        self.vel_y = 0
        self.jumped = False
        self.direction = 0
        self.in_air = True


class World():
    def __init__(self, data):
        self.tile_list = []

        # load images {rect will be basically storing the coordinates of each tile}
        dirt_img = pygame.image.load("./images/grassCenter.png")
        grass_img = pygame.image.load("./images/grassMid.png")
        sand_img = pygame.image.load('./images/sandCenter.png')
        sand2_img = pygame.image.load('./images/sandMid.png')
        cake_img = pygame.image.load('./images/cakeCenter.png')
        cake2_img = pygame.image.load('./images/cakeMid.png')
        choco_img = pygame.image.load('./images/chocoCenter.png')
        choco2_img = pygame.image.load('./images/chocoMid.png')
        stone_img = pygame.image.load('./images/stoneCenter.png')
        stone2_img = pygame.image.load('./images/stoneMid.png')
        tundra_img = pygame.image.load('./images/tundraCenter.png')
        tundra2_img = pygame.image.load('./images/tundraMid.png')

        row_count = 0
        for row in data:
            col_count = 0
            for tile in row:
                if tile == 1:
                    img = pygame.transform.scale(
                        dirt_img, (tile_size, tile_size))
                    img_rect = img.get_rect()
                    img_rect.x = col_count*tile_size
                    img_rect.y = row_count*tile_size
                    tile = (img, img_rect)
                    self.tile_list.append(tile)
                if tile == 2:
                    img = pygame.transform.scale(
                        grass_img, (tile_size, tile_size))
                    img_rect = img.get_rect()
                    img_rect.x = col_count*tile_size
                    img_rect.y = row_count*tile_size
                    tile = (img, img_rect)
                    self.tile_list.append(tile)
                if tile == 3:
                    blob = Enemy(col_count*tile_size, row_count *
                                 tile_size)
                    blob_group.add(blob)
                if tile == 4:
                    platform = Platform('grassHalfMid', col_count*tile_size, row_count *
                                        tile_size, 1, 0)
                    platform_group.add(platform)
                if tile == 5:
                    platform = Platform('grassHalfMidcopy', col_count*tile_size, row_count *
                                        tile_size, 0, 1)
                    platform_group.add(platform)
                if tile == 6:
                    lava = Lava(col_count*tile_size, row_count *
                                tile_size + (tile_size // 4))
                    lava_group.add(lava)
                if tile == 7:
                    coin = Coin(col_count*tile_size + (tile_size // 2), row_count *
                                tile_size + (tile_size // 2))
                    coin_group.add(coin)
                if tile == 8:
                    exit = Exit(col_count*tile_size, row_count *
                                tile_size-(int(tile_size//1.05)))
                    exit_group.add(exit)
                if tile == 9:
                    img = pygame.transform.scale(
                        sand_img, (tile_size, tile_size))
                    img_rect = img.get_rect()
                    img_rect.x = col_count*tile_size
                    img_rect.y = row_count*tile_size
                    tile = (img, img_rect)
                    self.tile_list.append(tile)
                if tile == 10:
                    img = pygame.transform.scale(
                        sand2_img, (tile_size, tile_size))
                    img_rect = img.get_rect()
                    img_rect.x = col_count*tile_size
                    img_rect.y = row_count*tile_size
                    tile = (img, img_rect)
                    self.tile_list.append(tile)
                if tile == 13:
                    img = pygame.transform.scale(
                        cake_img, (tile_size, tile_size))
                    img_rect = img.get_rect()
                    img_rect.x = col_count*tile_size
                    img_rect.y = row_count*tile_size
                    tile = (img, img_rect)
                    self.tile_list.append(tile)
                if tile == 14:
                    img = pygame.transform.scale(
                        cake2_img, (tile_size, tile_size))
                    img_rect = img.get_rect()
                    img_rect.x = col_count*tile_size
                    img_rect.y = row_count*tile_size
                    tile = (img, img_rect)
                    self.tile_list.append(tile)
                if tile == 17:
                    img = pygame.transform.scale(
                        choco_img, (tile_size, tile_size))
                    img_rect = img.get_rect()
                    img_rect.x = col_count*tile_size
                    img_rect.y = row_count*tile_size
                    tile = (img, img_rect)
                    self.tile_list.append(tile)
                if tile == 18:
                    img = pygame.transform.scale(
                        choco2_img, (tile_size, tile_size))
                    img_rect = img.get_rect()
                    img_rect.x = col_count*tile_size
                    img_rect.y = row_count*tile_size
                    tile = (img, img_rect)
                    self.tile_list.append(tile)
                if tile == 21:
                    img = pygame.transform.scale(
                        stone_img, (tile_size, tile_size))
                    img_rect = img.get_rect()
                    img_rect.x = col_count*tile_size
                    img_rect.y = row_count*tile_size
                    tile = (img, img_rect)
                    self.tile_list.append(tile)
                if tile == 22:
                    img = pygame.transform.scale(
                        stone2_img, (tile_size, tile_size))
                    img_rect = img.get_rect()
                    img_rect.x = col_count*tile_size
                    img_rect.y = row_count*tile_size
                    tile = (img, img_rect)
                    self.tile_list.append(tile)
                if tile == 25:
                    img = pygame.transform.scale(
                        tundra_img, (tile_size, tile_size))
                    img_rect = img.get_rect()
                    img_rect.x = col_count*tile_size
                    img_rect.y = row_count*tile_size
                    tile = (img, img_rect)
                    self.tile_list.append(tile)
                if tile == 26:
                    img = pygame.transform.scale(
                        tundra2_img, (tile_size, tile_size))
                    img_rect = img.get_rect()
                    img_rect.x = col_count*tile_size
                    img_rect.y = row_count*tile_size
                    tile = (img, img_rect)
                    self.tile_list.append(tile)
                if tile == 11:
                    platform = Platform('sandHalfMid', col_count*tile_size, row_count *
                                        tile_size, 1, 0)
                    platform_group.add(platform)
                if tile == 12:
                    platform = Platform('sandHalfMidcopy', col_count*tile_size, row_count *
                                        tile_size, 0, 1)
                    platform_group.add(platform)
                if tile == 15:
                    platform = Platform('cakeHalfAltMid', col_count*tile_size, row_count *
                                        tile_size, 1, 0)
                    platform_group.add(platform)
                if tile == 16:
                    platform = Platform('cakeHalfAltMidcopy', col_count*tile_size, row_count *
                                        tile_size, 0, 1)
                    platform_group.add(platform)
                if tile == 19:
                    platform = Platform('chocoHalfAltMid', col_count*tile_size, row_count *
                                        tile_size, 1, 0)
                    platform_group.add(platform)
                if tile == 20:
                    platform = Platform('chocoHalfAltMidcopy', col_count*tile_size, row_count *
                                        tile_size, 0, 1)
                    platform_group.add(platform)
                if tile == 23:
                    platform = Platform('stoneHalfMid', col_count*tile_size, row_count *
                                        tile_size, 1, 0)
                    platform_group.add(platform)
                if tile == 24:
                    platform = Platform('stoneHalfMidcopy', col_count*tile_size, row_count *
                                        tile_size, 0, 1)
                    platform_group.add(platform)
                if tile == 27:
                    platform = Platform('tundraHalfMid', col_count*tile_size, row_count *
                                        tile_size, 1, 0)
                    platform_group.add(platform)
                if tile == 28:
                    platform = Platform('tundraHalfMidcopy', col_count*tile_size, row_count *
                                        tile_size, 0, 1)
                    platform_group.add(platform)

                col_count += 1
            row_count += 1

    def draw(self):
        for tile in self.tile_list:
            screen.blit(tile[0], tile[1])
            # tile[0] shows img and tile[1]shows coordinates of rect or tile
            #pygame.draw.rect(screen, (255, 255, 255), tile[1], 2)


# sprite is a class of pygame and will help us with func of update,draw...
class Enemy(pygame.sprite.Sprite):
    def __init__(self, x, y):
        pygame.sprite.Sprite.__init__(self)
        self.image = pygame.image.load('./images/blockermad.png')
        self.image = pygame.transform.scale(
            self.image, (tile_size, tile_size))
        self.rect = self.image.get_rect()
        self.rect.x = x
        self.rect.y = y
        self.move_direction = 1
        self.move_counter = 0

    def update(self):
        self.rect.x += self.move_direction
        self.move_counter += 1
        if abs(self.move_counter) > 40:
            self.move_direction *= -1
            self.move_counter *= -1


class Platform(pygame.sprite.Sprite):
    def __init__(self, image, x, y, move_x, move_y):
        pygame.sprite.Sprite.__init__(self)
        img = pygame.image.load(f'./images/{image}.png')
        self.image = pygame.transform.scale(img, (tile_size, tile_size))
        self.rect = self.image.get_rect()
        self.rect.x = x
        self.rect.y = y
        self.move_direction = 1
        self.move_counter = 0
        self.move_x = move_x
        self.move_y = move_y

    def update(self):
        self.rect.x += self.move_direction*self.move_x
        self.rect.y += self.move_direction*self.move_y
        self.move_counter += 1
        if abs(self.move_counter) > 40:
            self.move_direction *= -1
            self.move_counter *= -1


class Lava(pygame.sprite.Sprite):
    def __init__(self, x, y):
        pygame.sprite.Sprite.__init__(self)
        img = pygame.image.load('./images/liquidLavaTop_mid.png')
        self.image = pygame.transform.scale(
            img, (tile_size, tile_size - tile_size//4))
        self.rect = self.image.get_rect()
        self.rect.x = x
        self.rect.y = y


class Coin(pygame.sprite.Sprite):
    def __init__(self, x, y):
        pygame.sprite.Sprite.__init__(self)
        img = pygame.image.load('./images/coin.png')
        self.image = pygame.transform.scale(
            img, (tile_size-tile_size//4, tile_size-tile_size//4))
        self.rect = self.image.get_rect()
        self.rect.center = (x, y)
        # central coordinates of coin are taken since they r smoller than a tile
        # taking x and y like before will pick up the top left corner of tile which wont have our coin


class Exit(pygame.sprite.Sprite):
    def __init__(self, x, y):
        pygame.sprite.Sprite.__init__(self)
        img = pygame.image.load('./images/gate.png')
        self.image = pygame.transform.scale(
            img, (tile_size, int(tile_size*1.5 + (tile_size//2))))
        self.rect = self.image.get_rect()
        self.rect.x = x
        self.rect.y = y


player = Player(40, screen_height-110)


# group is kinda like a list that will be storing our enemies.
blob_group = pygame.sprite.Group()
platform_group = pygame.sprite.Group()
lava_group = pygame.sprite.Group()
coin_group = pygame.sprite.Group()
exit_group = pygame.sprite.Group()

# create dummy coin for showing the score
score_coin = Coin(tile_size//2, tile_size//2)
coin_group.add(score_coin)

# load in level data and create world
if path.exists(f'C:\Downloads\level{level}_data'):
    pickle_in = open(f'C:\Downloads\level{level}_data', 'rb')
    world_data = pickle.load(pickle_in)
world = World(world_data)

# create buttons
restart_button = Button(290, 350, restart_img)
play_button = Button(100, 300, play_img)
exit_button = Button(380, 300, exit_img)

run = True
while run:

    clock.tick(fps)

    # blit puts images in a chosen screen
    screen.blit(bg_img, (0, 0))
    #screen.blit(cloud_img, (50, 70))

    if main_menu == True:
        if exit_button.draw() == True:
            run = False
        if play_button.draw():
            main_menu = False

    else:
        world.draw()

        if game_over == 0:
            blob_group.update()
            platform_group.update()
            # update score
            # check if a coin has been collected
            if pygame.sprite.spritecollide(player, coin_group, True):
                score += 1
                coin_fx.play()
            draw_text('X '+str(score), font_score, white, tile_size, 10)
            draw_text('Level ' + str(level), font_score,
                      white, tile_size + 580, 10)

        blob_group.draw(screen)
        platform_group.draw(screen)
        lava_group.draw(screen)
        coin_group.draw(screen)
        exit_group.draw(screen)

        game_over = player.update(game_over)

        # if player has died
        if game_over == -1:
            if restart_button.draw():
                world_data = []
                world = reset_level(level)
                game_over = 0
                score = 0

        # if player has completed the level
        if game_over == 1:
            # reset game and go to next level
            level += 1
            if level <= max_levels:
                # reset level
                bg_img = pygame.image.load(f'./images/bg{level}.png').convert_alpha()
                world_data = []
                world = reset_level(level)
                game_over = 0
            else:
                game_won.play()
                draw_text('YOU WIN!', font, blue,
                          (screen_width//2)-120, screen_height//2-55)
                # reset game
                if restart_button.draw():
                    bg_img = pygame.image.load('./images/bg0.png').convert_alpha()
                    level = 0
                    # reset level
                    world_data = []
                    world = reset_level(level)
                    game_over = 0
                    score = 0

    # event handles all the buttons present in game and keys mapping and by using.get we get access to em(events)
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            run = False

    pygame.display.update()

pygame.quit()
# stopping the initialization done at first
