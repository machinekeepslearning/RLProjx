import math
import random

import pygame

pygame.init()
bounds = (1300, 700)
screen = pygame.display.set_mode(bounds)
clock = pygame.time.Clock()
running = True
font = pygame.font.SysFont('Arial', 30)
fps = 600

action_table = {}
reward_table = {}
q_table = {}

death_coordinates = []
used_coordinates = []
coordinates = []
#hyper params
learning_rate = 0.1
gamma = 0.9

# Initialize transition table
# 0: left, 1: right, 2: up, 3: down
# id: 0 is obstacle, 1 is goal
for i in range(13):
    for j in range(7):
        x = i * 100
        y = j * 100
        coordinates.append((x, y))


        pos_actions = []
        if 0 <= x + 100 <= 1200:
            pos_actions.append((x + 100, y))
        if 0 <= x - 100 <= 1200:
            pos_actions.append((x - 100, y))
        if 0 <= y + 100 <= 600:
            pos_actions.append((x, y + 100))
        if 0 <= y - 100 <= 600:
            pos_actions.append((x, y - 100))

        action_table.update({(x, y): pos_actions})
        reward_table.update({(x, y): 0.0})
        #reward_table.update({(x, y): 1.0/math.dist((x, y), (1200, 600))})
        q_table.update({(x, y): 0.0})

print(action_table)

class Player(pygame.sprite.Sprite):
    def __init__(self, color, width, height, speed):
        pygame.sprite.Sprite.__init__(self)
        self.speed = speed
        self.image = pygame.Surface([width, height])
        self.image.fill(color)
        self.rect = self.image.get_rect()
        self.rect.x = 0
        self.rect.y = 0

    def update(self):
        keys = pygame.key.get_pressed()

        if keys[pygame.K_w]:
            self.rect = self.rect.move(0, -self.speed)
        if keys[pygame.K_s]:
            self.rect = self.rect.move(0, self.speed)
        if keys[pygame.K_a]:
            self.rect = self.rect.move(-self.speed, 0)
        if keys[pygame.K_d]:
            self.rect = self.rect.move(self.speed, 0)


class Bot(pygame.sprite.Sprite):
    def __init__(self, color, width, height):
        pygame.sprite.Sprite.__init__(self)
        self.image = pygame.Surface([width, height])
        self.image.fill(color)
        self.rect = self.image.get_rect()
        self.rect.x = 0
        self.rect.y = 0
        self.win_counter = 0

    def update(self, *args, **kwargs):
        global fps

        actions = action_table.get((self.rect.x, self.rect.y))
        #best_action = actions[0]
        best_q = q_table.get(actions[0])

        my_actions = []
        for action in actions:
            if q_table.get(action) > best_q:
                best_q = q_table.get(action)
        for action in actions:
            if q_table.get(action) == best_q:
                my_actions.append(action)
        idx = random.randint(0, len(my_actions) - 1)
        best_action = my_actions[idx]
        new_q = (
                learning_rate * q_table.get((self.rect.x, self.rect.y)) +
                (1.0 - learning_rate) * (reward_table.get((self.rect.x, self.rect.y)) + gamma * best_q)

        )

        q_table.update({(self.rect.x, self.rect.y): new_q})

        if not ((self.rect.x, self.rect.y) in death_coordinates):
            self.rect.x = best_action[0]
            self.rect.y = best_action[1]
        else:
            if self.rect.x == 1200 and self.rect.y == 600:
                self.win_counter += 1
            self.rect.x = 0
            self.rect.y = 0

        if self.win_counter == 100:
            fps = 1


class Object(pygame.sprite.Sprite):
    def __init__(self, width, height, x, y, id):
        pygame.sprite.Sprite.__init__(self)
        self.id = id
        self.image = pygame.Surface([width, height])
        if id == 0:
            self.image.fill("red")
        else:
            self.image.fill("green")
        self.rect = self.image.get_rect()
        self.rect.x = x
        self.rect.y = y

    def get_id(self):
        return self.id


all_sprites = pygame.sprite.Group()

bot = Bot("black", 100, 100)

all_sprites.add(Object(100, 100, 1200, 600, 1))

all_sprites.add(bot)

used_coordinates.append((1200, 600))
reward_table.update({(1200, 600): 10})
used_coordinates.append((0, 0))

for i in range(7):
    x = random.randint(0, 12) * 100
    y = random.randint(0, 6) * 100
    dupe = (x, y) in used_coordinates
    while dupe:
        x = random.randint(0, 12) * 100
        y = random.randint(0, 6) * 100
        dupe = (x, y) in used_coordinates

    all_sprites.add(Object(100, 100, x, y, 0))
    used_coordinates.append((x, y))
    reward_table.update({(x, y): -10})
    death_coordinates.append((x, y))
death_coordinates.append((1200, 600))

def render_vals():
    for i in range(13):
        for j in range(7):
            text = font.render(str(int(q_table.get((i * 100, j * 100)))), True, "black")
            screen.blit(text, (i * 100, j * 100))

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    all_sprites.update()

    screen.fill("gray")
    all_sprites.draw(screen)

    render_vals()

    pygame.display.flip()
    clock.tick(fps)

pygame.quit()
