import math
import random
import threading

import numpy
import pygame

pygame.init()
bounds = (700, 700)
screen = pygame.display.set_mode(bounds)
clock = pygame.time.Clock()
running = True
fps = 60
gameOver = True

#pygame.time.wait(12000)

font = pygame.font.SysFont("Arial", 12)
gmfont = pygame.font.SysFont("Arial", 40)
gameOverSurface = gmfont.render("GAME OVER", False, "black")

bullets = pygame.sprite.Group()
all_sprites = pygame.sprite.Group()
asteroids = pygame.sprite.Group()
agents = pygame.sprite.Group()


def cooldown(debounce):
    pygame.time.wait(200)
    debounce[0] = True


def spawnAsteroids():
    while running and not gameOver:
        asteroids.add(Asteroid(400))
        pygame.time.wait(800)


class Bullet(pygame.sprite.Sprite):
    def __init__(self, rect, angle):
        super().__init__()
        self.angle = angle
        self.image = pygame.image.load("bullet.png")
        self.rect = self.image.get_rect(center=rect.center)
        self.image = pygame.transform.rotate(self.image, angle)
        self.rect = self.image.get_rect(center=self.rect.center)

    def update(self):
        if self.rect.x > 700 or self.rect.x < 0 or self.rect.y > 700 or self.rect.y < 0:
            self.kill()
        vec_x = -10 * math.sin(self.angle * math.pi / 180.0)
        vec_y = -10 * math.cos(self.angle * math.pi / 180.0)
        self.rect.move_ip(round(vec_x), round(vec_y))


class Bot(pygame.sprite.Sprite):
    def __init__(self, width, height, accel):
        super().__init__()
        self.accel = accel
        self.speed = 0
        #self.ogimage = pygame.Surface([width, height])
        self.ogimage = pygame.image.load("ship.png")
        self.ogimage = pygame.transform.scale_by(self.ogimage, 0.2)
        self.image = self.ogimage
        self.rect = self.image.get_rect(center=(350, 350))
        self.angle = 0
        self.debounce = [True]
        self.lives = 3
        self.score = 0
        self.action_space = (0, 1, 2, 3, 4, 5)

    def update(self, action):
        global running, gameOver

        colls = asteroids.sprites()
        idx = self.rect.collidelist([sprite.rect for sprite in colls])

        out_of_bounds = self.rect.centerx > 700 or self.rect.centerx < 0 or self.rect.centery > 700 or self.rect.centery < 0

        if out_of_bounds:
            self.score -= 100

        if idx != -1 and self.lives > 0:
            self.lives -= 1
            plr.score -= 10
            colls[idx].kill()
        elif self.lives == 0 or out_of_bounds:
            gameOver = True
            asteroids.empty()
            bullets.empty()
            self.kill()
            print("Game over")
            print(f"Your score was : {self.score}")
        else:
            plr.score += 0.01

        keys = pygame.key.get_pressed()

        if action == 0:
            self.speed += self.accel
        if action == 1:
            self.speed -= self.accel
        if action == 2:
            self.angle += 5
        if action == 3:
            self.angle += -5
        if action == 4 and self.debounce[0]:
            bullets.add(Bullet(self.rect, self.angle))
            self.debounce[0] = False
            threading.Thread(target=cooldown, args=(self.debounce,)).start()
        if action == 5:
            pass

        self.image = pygame.transform.rotate(self.ogimage, self.angle)
        self.rect = self.image.get_rect(center=self.rect.center)
        vec_x = -self.speed * math.sin(self.angle * math.pi / 180.0)
        vec_y = -self.speed * math.cos(self.angle * math.pi / 180.0)
        self.rect.move_ip(round(vec_x), round(vec_y))


class Asteroid(pygame.sprite.Sprite):
    def __init__(self, radius):
        super().__init__()
        self.image = pygame.Surface([50, 50])
        self.rect = self.image.get_rect()
        self.angle = random.uniform(0, 2 * math.pi)
        radius = radius
        self.rect.x = radius * math.cos(self.angle) + 350
        self.rect.y = radius * math.sin(self.angle) + 350
        target_x = 350
        target_y = 350
        magnitude = math.sqrt((target_x - self.rect.x) ** 2 + (target_y - self.rect.y) ** 2)
        speed = random.uniform(2, 4)
        self.velocity_x = speed * (target_x - self.rect.x) / magnitude
        self.velocity_y = speed * (target_y - self.rect.y) / magnitude

    def update(self):
        idx = self.rect.collidelist([sprite.rect for sprite in bullets.sprites()])
        if idx != -1:
            bullets.sprites()[idx].kill()
            self.kill()
            plr.score += 5
        elif self.rect.x > 800 or self.rect.x < -100 or self.rect.y > 800 or self.rect.y < -100:
            self.kill()
        self.rect.move_ip(self.velocity_x, self.velocity_y)


def globalUpdate(groups):
    for group in groups:
        group.update()
        group.draw(screen)


plr = Bot(100, 100, 1)


def step(action):
    global running
    # poll for events
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    old_score = plr.score

    # update and draw sprites + screen + text
    screen.fill("white")
    globalUpdate([all_sprites, asteroids, bullets])
    agents.update(action)
    agents.draw(screen)

    scoreSurface = font.render(f"Score: {plr.score}", False, "black")
    livesSurface = font.render(f"Lives: {plr.lives}", False, "black")
    screen.blit(scoreSurface, (600, 20))
    screen.blit(livesSurface, (20, 20))

    #check if player died
    if gameOver:
        screen.blit(gameOverSurface, (260, 300))

    #swap buffers and increment clock
    pygame.display.flip()
    clock.tick(fps)

    observation = numpy.expand_dims(numpy.array(pygame.PixelArray(screen), dtype=numpy.float32), 0)
    reward = plr.score - old_score
    terminated = gameOver
    truncated = False
    _ = "skibidi"
    # print(numpy.array(observation, dtype=numpy.float32))

    return observation, reward, terminated, truncated, _


def reset():
    global gameOver
    pygame.time.wait(1000)
    gameOver = False
    plr.score = 0
    plr.lives = 3
    plr.rect.center = (350, 350)
    agents.add(plr)
    threading.Thread(target=spawnAsteroids).start()

    return numpy.expand_dims(numpy.array(pygame.PixelArray(screen), dtype=numpy.float32), 0)


# def main():
#     global running
#
#     i = 0
#     while running:
#         if gameOver:
#             reset()
#         step(i % 6)
#         i += 1
#
#     pygame.quit()
#
#
# if __name__ == "__main__":
#     main()
