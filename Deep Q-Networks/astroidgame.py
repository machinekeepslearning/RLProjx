import math
import random
import threading
import numpy
import pygame

render = False
bounds = (800, 800)
pygame.init()
if render:
    screen = pygame.display.set_mode(bounds)
clock = pygame.time.Clock()
running = True
fps = 600
gameOver = True
asteroids_active = False
seed = 0
global_tick = 0

#pygame.time.wait(12000)

if render:
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
        asteroids.add(Asteroid(bounds[0] / 2 + 50))
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
        if self.rect.x > bounds[0] or self.rect.x < 0 or self.rect.y > bounds[1] or self.rect.y < 0:
            self.kill()
        vec_x = -10 * math.sin(self.angle * math.pi / 180.0)
        vec_y = -10 * math.cos(self.angle * math.pi / 180.0)
        self.rect.move_ip(round(vec_x), round(vec_y))


class Player(pygame.sprite.Sprite):
    def __init__(self, width, height, accel, sensor_length):
        super().__init__()
        self.accel = accel
        self.speed = 0
        self.ogimage = pygame.image.load("ship.png")
        self.ogimage = pygame.transform.scale(self.ogimage, (width, height))
        self.image = self.ogimage
        self.rect = self.image.get_rect(center=(bounds[0] / 2, bounds[1] / 2))
        self.angle = 0
        self.debounce = [True]
        self.max_lives = 1
        self.lives = self.max_lives
        self.score = 0
        self.out_of_bounds = 0
        self.vec_x = 0
        self.vec_y = 0

        #sensors
        self.num_sensors = 20
        self.sensor_size = 50
        self.og_sensors = pygame.transform.scale(pygame.image.load("sensor_blue.png"), (1, self.sensor_size))
        #self.og_sensors = pygame.image.load("sensor_blue.png")
        self.sensors = [None] * self.num_sensors
        self.sensor_angles = [0] * self.num_sensors
        self.sensor_rects = [0] * self.num_sensors
        offset = 0
        max_angle = 360
        for i in range(self.num_sensors):
            self.sensor_angles[i] = offset + i * max_angle / self.num_sensors
            self.sensors[i] = pygame.transform.rotate(self.og_sensors, self.sensor_angles[i])
            self.sensor_rects[i] = self.sensors[i].get_rect()
            self.sensor_rects[i].center = (
                self.rect.centerx + (self.sensor_size/2) * math.sin(self.sensor_angles[i] * math.pi / 180.0),
                self.rect.centery + (self.sensor_size/2) * math.cos(self.sensor_angles[i] * math.pi / 180.0))

        #RL variables
        self.num_observations = self.num_sensors
        self.observation_x = [0] * self.num_sensors
        self.observation_y = [0] * self.num_sensors
        self.action_space = (0, 1, 2, 3)

    def reset(self):
        self.score = 0
        self.lives = self.max_lives
        self.rect.center = (int(bounds[0] / 2), int(bounds[1] / 2))
        for i in range(self.num_sensors):
            self.sensor_rects[i].center = (
                self.rect.centerx + (self.sensor_size/2) * math.sin(self.sensor_angles[i] * math.pi / 180.0),
                self.rect.centery + (self.sensor_size/2) * math.cos(self.sensor_angles[i] * math.pi / 180.0))

    def update(self, action):
        global running, gameOver

        colls = asteroids.sprites()
        idx = self.rect.collidelist([sprite.rect for sprite in colls])

        self.out_of_bounds = self.rect.centerx > 700 or self.rect.centerx < 0 or self.rect.centery > 700 or self.rect.centery < 0
        if self.out_of_bounds:
            #self.score -= 200
            if self.rect.centerx > bounds[0]:
                self.rect.centerx = 0
            if self.rect.centerx < 0:
                self.rect.centerx = bounds[0]
            if self.rect.centery > bounds[1]:
                self.rect.centery = 0
            if self.rect.centery < 0:
                self.rect.centery = bounds[1]
            for i in range(self.num_sensors):
                self.sensor_rects[i].center = (
                    self.rect.centerx + (self.sensor_size/2) * math.sin(self.sensor_angles[i] * math.pi / 180.0),
                    self.rect.centery + (self.sensor_size/2) * math.cos(self.sensor_angles[i] * math.pi / 180.0))
        if self.lives == 0:
            gameOver = True
            asteroids.empty()
            bullets.empty()
            self.lives = 0
            #self.score -= 50
            self.angle = 0
            self.speed = 0
        elif idx != -1 and self.lives > 0:
            #self.score -= 10
            self.lives -= 1
            colls[idx].kill()
        # else:
        #     self.score += 1e-5

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

        # if keys[pygame.K_w]:
        #     self.speed += self.accel
        # if keys[pygame.K_s]:
        #     self.speed += -self.accel
        # if keys[pygame.K_a]:
        #     self.angle += 5
        # if keys[pygame.K_d]:
        #     self.angle += -5
        # if action == 4 and self.debounce[0]:
        #     bullets.add(Bullet(self.rect, self.angle))
        #     self.debounce[0] = False
        #     threading.Thread(target=cooldown, args=(self.debounce,)).start()

        #limiter
        if self.angle >= 360:
            self.angle = 0
        self.speed = max(min(self.speed, 2), -1)

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
            for asteroid in colls:  # + borders.sprites()):
                a_mask = pygame.mask.from_surface(asteroid.image)
                coll_pos = mask.overlap(a_mask, (asteroid.rect.x - self.sensor_rects[i].x,
                                                 asteroid.rect.y - self.sensor_rects[i].y))
                if coll_pos is not None:
                    self.observation_x[i] = coll_pos[0] / bounds[0]
                    self.observation_y[i] = coll_pos[1] / bounds[1]
                    #asteroid.kill()
                    self.score -= 1 / (math.dist(self.rect.center, coll_pos) + 10)
                else:
                    self.observation_x[i] = 0
                    self.observation_y[i] = 0
                    self.score += 5e-5

class Asteroid(pygame.sprite.Sprite):
    def __init__(self, radius):
        super().__init__()
        global seed
        self.image = pygame.Surface([20, 20])
        self.rect = self.image.get_rect()
        #random.seed(seed)
        #seed += 1
        self.angle = random.uniform(0, 2 * math.pi)
        speed = random.uniform(2, 3)
        radius = radius
        self.rect.x = radius * math.cos(self.angle) + bounds[0] / 2
        self.rect.y = radius * math.sin(self.angle) + bounds[1] / 2
        target_x = bot.rect.x  #bounds[0]/2
        target_y = bot.rect.y  #bounds[1]/2
        magnitude = math.sqrt((target_x - self.rect.x) ** 2 + (target_y - self.rect.y) ** 2)
        self.velocity_x = speed * (target_x - self.rect.x) / magnitude
        self.velocity_y = speed * (target_y - self.rect.y) / magnitude

    def update(self):
        idx = self.rect.collidelist([sprite.rect for sprite in bullets.sprites()])
        if idx != -1:
            bullets.sprites()[idx].kill()
            bot.score += 50
            self.kill()
        elif self.rect.x > bounds[0] + 100 or self.rect.x < -100 or self.rect.y > bounds[1] + 100 or self.rect.y < -100:
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


def globalUpdate(groups):
    global render
    for group in groups:
        group.update()
        if render:
            group.draw(screen)


bot = Player(10, 10, 0.01, 20)


#plr = Player(100, 100, 0.1)

def step(action):
    global running, render, global_tick, gameOver

    global_tick += 1
    # poll for events
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    old_score = bot.score

    # update and draw sprites + screen + text
    agents.update(action)
    if render: screen.fill("white")
    globalUpdate([all_sprites, asteroids, bullets, borders])
    #print(len(asteroids.sprites()))
    if render:
        for i in range(bot.num_sensors):
            screen.blit(bot.sensors[i], bot.sensor_rects[i])
        agents.draw(screen)

        scoreSurface = font.render(f"Score: {bot.score}", False, "black")
        livesSurface = font.render(f"Lives: {bot.lives}", False, "black")
        screen.blit(scoreSurface, (100, 20))
        screen.blit(livesSurface, (20, 20))

        #swap buffers and increment clock
        pygame.display.flip()
        clock.tick(fps)

    reward = bot.score - old_score
    observation = numpy.array(bot.observation_x + bot.observation_y)
    observation = numpy.append(observation, (bot.rect.top + 10) / bounds[1])
    observation = numpy.append(observation, (bot.rect.left - 10) / bounds[0])
    observation = numpy.append(observation, (bot.rect.bottom - 10) / bounds[1])
    observation = numpy.append(observation, (bot.rect.right + 10) / bounds[0])
    observation = numpy.append(observation, bot.angle / 360)
    observation = numpy.append(observation, bot.vec_x / bounds[0])
    observation = numpy.append(observation, bot.vec_y / bounds[1])
    #print(observation)
    #print(bot.rect.top)

    if (global_tick >= 5000):
        gameOver = True
        global_tick = 0
    terminated = gameOver

    return observation, reward, terminated, False, False


def reset():
    global gameOver, seed
    asteroids.empty()

    gameOver = False
    bot.reset()
    seed = 0
    if not bot.alive():
        agents.add(bot)
        print("added")
    if not asteroids_active:
        threading.Thread(target=spawnAsteroids).start()

    state = numpy.array([])
    for _ in range(47):
        state = numpy.append(state, 0)

    return state, 0


b1 = Border(10, bounds[1], 0, 0)
b2 = Border(10, bounds[1], bounds[0] - 10, 0)
b3 = Border(bounds[0], 10, 0, 0)
b4 = Border(bounds[0], 10, 0, bounds[1] - 10)

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
        step(i % 6)
        i += 1

    pygame.quit()


# if __name__ == "__main__":
#     main()
