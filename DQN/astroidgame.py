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
asteroids_active = False
seed = 0

#pygame.time.wait(12000)

font = pygame.font.SysFont("Arial", 12)
gmfont = pygame.font.SysFont("Arial", 40)
gameOverSurface = gmfont.render("GAME OVER", False, "black")

bullets = pygame.sprite.Group()
all_sprites = pygame.sprite.Group()
asteroids = pygame.sprite.Group()
agents = pygame.sprite.Group()
borders = pygame.sprite.Group()


def cooldown(debounce):
    pygame.time.wait(200)
    debounce[0] = True


def spawnAsteroids():
    global asteroids_active

    while running and not gameOver:
        asteroids.add(Asteroid(400))
        asteroids_active = True
        pygame.time.wait(800)
    asteroids_active = False


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


class Player(pygame.sprite.Sprite):
    def __init__(self, width, height, accel):
        super().__init__()
        self.accel = accel
        self.speed = 0
        self.ogimage = pygame.image.load("ship.png")
        self.ogimage = pygame.transform.scale(self.ogimage, (width, height))
        self.image = self.ogimage
        self.rect = self.image.get_rect(center=(350, 350))
        self.angle = 0
        self.debounce = [True]
        self.lives = 3
        self.score = 0
        self.out_of_bounds = 0
        self.vec_x = 0
        self.vec_y = 0

        #sensors
        self.num_sensors = 15
        self.og_sensors = [None] * self.num_sensors
        self.sensors = [None] * self.num_sensors
        self.sensor_angles = [0] * self.num_sensors
        self.sensor_rects = [0] * self.num_sensors
        for i in range(self.num_sensors):
            self.og_sensors[i] = pygame.transform.scale2x(pygame.image.load("sensor_blue.png"))
            self.sensor_angles[i] = i * 360.0 / self.num_sensors
            self.sensors[i] = pygame.transform.rotate(self.og_sensors[i], self.sensor_angles[i])
            self.sensor_rects[i] = self.sensors[i].get_rect()
            self.sensor_rects[i].center = (
                self.rect.centerx + 50 * math.sin(self.sensor_angles[i] * math.pi / 180.0),
                self.rect.centery + 50 * math.cos(self.sensor_angles[i] * math.pi / 180.0))

        #RL variables
        self.num_observations = self.num_sensors
        self.observation_x = [0] * self.num_sensors
        self.observation_y = [0] * self.num_sensors
        self.action_space = (0, 1, 2, 3)

    def reset(self):
        for i in range(self.num_sensors):
            self.og_sensors[i] = pygame.transform.scale2x(pygame.image.load("sensor_blue.png"))
            self.sensor_angles[i] = i * 360.0 / self.num_sensors
            self.sensors[i] = pygame.transform.rotate(self.og_sensors[i], self.sensor_angles[i])
            self.sensor_rects[i] = self.sensors[i].get_rect()
            self.sensor_rects[i].center = (
                self.rect.centerx + 50 * math.sin(self.sensor_angles[i] * math.pi / 180.0),
                self.rect.centery + 50 * math.cos(self.sensor_angles[i] * math.pi / 180.0))

    def update(self, action):
        global running, gameOver

        colls = asteroids.sprites()
        idx = self.rect.collidelist([sprite.rect for sprite in colls])

        self.out_of_bounds = self.rect.centerx > 700 or self.rect.centerx < 0 or self.rect.centery > 700 or self.rect.centery < 0
        if self.out_of_bounds:
            #self.score -= 200
            if self.rect.centerx > 700:
                self.rect.centerx = 0
            if self.rect.centerx < 0:
                self.rect.centerx = 700
            if self.rect.centery > 700:
                self.rect.centery = 0
            if self.rect.centery < 0:
                self.rect.centery = 700
            self.reset()
        if self.lives == 0:
            gameOver = True
            asteroids.empty()
            bullets.empty()
            self.lives = 0
            self.score -= 600
            self.angle = 0
            self.speed = 0
        elif idx != -1 and self.lives > 0:
            self.score -= 100
            self.lives -= 1
            colls[idx].kill()
        # else:
        #     self.score += 1 / (math.fabs(self.speed) + 100)

        keys = pygame.key.get_pressed()

        if action == 0:
            self.speed += self.accel
        if action == 1:
            self.speed += -self.accel
        if action == 2:
            self.angle += 5
        if action == 3:
            self.angle += -5
        if action == 4 and self.debounce[0]:
            bullets.add(Bullet(self.rect, self.angle))
            self.debounce[0] = False
            threading.Thread(target=cooldown, args=(self.debounce,)).start()

        #limiter
        if self.angle >= 360:
            self.angle -= 360
        self.speed = max(min(self.speed, 5), 0)

        #fix centering of rotated image
        self.image = pygame.transform.rotate(self.ogimage, self.angle)
        self.rect = self.image.get_rect(center=self.rect.center)
        self.vec_x = -self.speed * math.sin(self.angle * math.pi / 180.0)
        self.vec_y = -self.speed * math.cos(self.angle * math.pi / 180.0)
        self.rect.move_ip(round(self.vec_x), round(self.vec_y))

        #draw sensors and sense collision
        for i in range(len(self.sensors)):
            self.sensor_rects[i].move_ip(round(self.vec_x), round(self.vec_y))
            mask = pygame.mask.from_surface(self.sensors[i])
            for asteroid in colls: # + borders.sprites()):
                a_mask = pygame.mask.from_surface(asteroid.image)
                coll_pos = mask.overlap(a_mask, (asteroid.rect.x - self.sensor_rects[i].x,
                                                 asteroid.rect.y - self.sensor_rects[i].y))
                if coll_pos is not None:
                    self.observation_x[i] = coll_pos[0]
                    self.observation_y[i] = coll_pos[1]
                    #asteroid.kill()
                    self.score -= 1/(math.dist(self.rect.center, coll_pos) + 10)
                else:
                    self.observation_x[i] = -300
                    self.observation_y[i] = -300
                    # self.score += 0.000005


class Asteroid(pygame.sprite.Sprite):
    def __init__(self, radius):
        super().__init__()
        global seed
        self.image = pygame.Surface([50, 50])
        self.rect = self.image.get_rect()
        random.seed(seed)
        seed += 1
        self.angle = random.uniform(0, 2 * math.pi)
        speed = random.uniform(2, 4)
        radius = radius
        self.rect.x = radius * math.cos(self.angle) + 350
        self.rect.y = radius * math.sin(self.angle) + 350
        target_x = 350
        target_y = 350
        magnitude = math.sqrt((target_x - self.rect.x) ** 2 + (target_y - self.rect.y) ** 2)
        self.velocity_x = speed * (target_x - self.rect.x) / magnitude
        self.velocity_y = speed * (target_y - self.rect.y) / magnitude

    def update(self):
        idx = self.rect.collidelist([sprite.rect for sprite in bullets.sprites()])
        if idx != -1:
            bullets.sprites()[idx].kill()
            bot.score += 50
            self.kill()
        elif self.rect.x > 800 or self.rect.x < -100 or self.rect.y > 800 or self.rect.y < -100:
            self.kill()
        self.rect.move_ip(self.velocity_x, self.velocity_y)


class Border(pygame.sprite.Sprite):
    def __init__(self, width, height, left, top):
        super().__init__()
        self.image = pygame.Surface((width, height))
        self.image.fill("red")
        self.rect = self.image.get_rect(left=left, top=top)
        self.velocity_x = 0
        self.velocity_y = 0


def globalUpdate(groups, render):
    for group in groups:
        group.update()
        group.draw(screen)


bot = Player(10, 10, 0.1)


#plr = Player(100, 100, 0.1)

def step(action, render):
    global running
    # poll for events
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    old_score = bot.score

    screen.fill("white")
    # update and draw sprites + screen + text
    globalUpdate([all_sprites, asteroids, bullets, borders], render)
    agents.update(action)
    #print(len(asteroids.sprites()))
    if render:
        for i in range(bot.num_sensors):
            screen.blit(bot.sensors[i], bot.sensor_rects[i])
        agents.draw(screen)

        scoreSurface = font.render(f"Score: {bot.score}", False, "black")
        livesSurface = font.render(f"Lives: {bot.lives}", False, "black")
        screen.blit(scoreSurface, (600, 20))
        screen.blit(livesSurface, (20, 20))

        #swap buffers and increment clock
        pygame.display.flip()
        clock.tick()

    reward = bot.score - old_score
    observation = numpy.array(bot.observation_x + bot.observation_y)
    observation = numpy.append(observation, bot.rect.top)
    observation = numpy.append(observation, bot.rect.left)
    observation = numpy.append(observation, bot.rect.bottom)
    observation = numpy.append(observation, bot.rect.right)
    observation = numpy.append(observation, bot.angle)
    observation = numpy.append(observation, bot.vec_x)
    observation = numpy.append(observation, bot.vec_y)

    #print(bot.rect.top)

    terminated = gameOver

    return observation, reward, terminated, False, False


def reset():
    global gameOver, seed
    asteroids.empty()
    #pygame.time.wait(1000)
    gameOver = False
    bot.score = 0
    bot.lives = 3
    bot.rect.center = (350, 350)
    bot.reset()
    seed = 0
    if not bot.alive():
        agents.add(bot)
    if not asteroids_active:
        threading.Thread(target=spawnAsteroids).start()

    state = numpy.array([])
    for _ in range(37):
        state = numpy.append(state, 0)

    return state


b1 = Border(10, 700, 0, 0)
b2 = Border(10, 700, 690, 0)
b3 = Border(700, 10, 0, 0)
b4 = Border(700, 10, 0, 690)

borders.add(b1)
borders.add(b2)
borders.add(b3)
borders.add(b4)


def main():
    global running

    i = 0
    while running:
        if gameOver:
            reset()
        step(i % 6, True)
        i += 1

    pygame.quit()

# if __name__ == "__main__":
#     main()
