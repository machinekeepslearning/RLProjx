import random
import math
import threading

import pygame
import numpy
import keyboard
import time

debug = False
render = False
running = True
gameover = False

asteroids = {}
xbounds = [0, 800]
ybounds = [0, 800]
bounds = (xbounds[1], ybounds[1])

globalId = 0
action_space = (0, 1, 2, 3)

if render:
    pygame.init()
    screen = pygame.display.set_mode(bounds)
    clock = pygame.time.Clock()
    screen.fill("black")


def circle_line(a, b, xc, yc, R):
    """
    Accepts a Line of form y=ax+b
    Accepts a Circle of form (y-yc)^2+(x-xc)^2=R^2
    returns (None, None), (None, None) if no solution
    returns solutions as (x1,y1),(x2,y2) if solution found
    """
    beta = a * xc + b - yc
    determinant = (a ** 2 * beta ** 2) - (a ** 2 + 1) * (beta ** 2 - R ** 2)
    if determinant < 0:
        return (None, None), (None, None)

    n1 = -a * beta
    n2 = math.sqrt(determinant)
    n3 = R * (a ** 2 + 1)

    cost1 = (n1 + n2) / n3
    cost2 = (n1 - n2) / n3

    x1 = R * cost1 + xc
    y1 = a * x1 + b

    x2 = R * cost2 + xc
    y2 = a * x2 + b

    return (x1, y1), (x2, y2)


def spawnAsteroids():
    global globalId

    while running:
        asteroids.update({globalId: Asteroid(
            speed=random.randint(150, 200),
            radius=random.randint(10, 30),
            id=globalId)})
        globalId += 1
        time.sleep(2)

def spawnSingleAsteroid():
    global globalId

    asteroids.update({globalId: Asteroid(
        speed=random.randint(50, 100),
        radius=random.randint(10, 30),
        id=globalId)})
    globalId += 1

class Asteroid:
    def __init__(self, speed, radius, id):
        self.color = "red"
        self.id = id

        #set size
        self.radius = radius

        #set position outside of frame
        self.pos_radius = 700
        self.start_angle = random.uniform(0, 2 * math.pi)
        self.center = numpy.array(
            [
                (xbounds[1] - xbounds[0]) / 2 + self.pos_radius * math.cos(self.start_angle),
                (ybounds[1] - ybounds[0]) / 2 + self.pos_radius * math.sin(self.start_angle)
            ],
            dtype=numpy.float32)

        #set movement
        self.speed = speed * 5e-4
        self.dir = random.uniform(0, 2 * math.pi)

        chase_num = random.randint(1, 10)
        if chase_num < 6:
            disp = (bot.center - self.center)
            self.velocity = disp/math.sqrt(numpy.dot(disp, disp)) * self.speed
        else:
            self.velocity = numpy.array([self.speed * math.cos(self.dir), self.speed * math.sin(self.dir)])

    def render(self):
        pygame.draw.circle(screen, self.color, (int(self.center[0]), int(self.center[1])), self.radius)

    def update(self):
        self.center[0] += self.velocity[0]
        self.center[1] += self.velocity[1]

        if (self.center[0] < (xbounds[0] - 300) or
                self.center[0] > (xbounds[1] + 300) or
                self.center[1] < (ybounds[0] - 300) or
                self.center[1] > (ybounds[1] + 300)):
            asteroids.pop(self.id)

        if render:
            self.render()


class Player:
    def __init__(self, radius, speed):
        self.max_lives = 10
        self.lives = self.max_lives
        self.angle = 0
        self.speed = speed * 5e-4
        self.center = numpy.array([(xbounds[1] - xbounds[0]) / 2, (ybounds[1] - ybounds[0]) / 2])
        self.radius = radius
        self.velocity = numpy.zeros((2,))
        self.action_space = (0, 1, 2, 3)

        self.collisions = []
        self.sort_indices = []
        self.coll_keys = []

        self.reward = 0
        self.old_score = 0
        self.score = 0

    def render(self):
        pygame.draw.circle(screen, "blue", (int(self.center[0]), int(self.center[1])), self.radius)

    def reset(self):
        self.lives = self.max_lives
        self.score = 0
        self.center = numpy.array([bounds[0] / 2, bounds[1] / 2])
        self.angle = 0
    def computeInputs(self):
        global gameover

        inputs = numpy.zeros((39,))

        inputs[30] = self.center[0] / bounds[0]
        inputs[31] = self.center[1] / bounds[1]
        inputs[32] = self.radius / bounds[0]
        inputs[33] = self.velocity[0] / bounds[0]
        inputs[34] = self.velocity[1] / bounds[1]
        inputs[35] = (xbounds[0] - self.center[0])/bounds[0]
        inputs[36] = (xbounds[1] - self.center[0])/bounds[0]
        inputs[37] = (ybounds[0] - self.center[1])/bounds[1]
        inputs[38] = (ybounds[1] - self.center[1])/bounds[1]

        asteroids_pos = numpy.empty((0, 2))
        asteroid_radii = numpy.empty((0,))
        asteroid_list = list(asteroids.values())
        for asteroid in asteroid_list:
            asteroids_pos = numpy.concatenate((asteroids_pos, [asteroid.center]))
            asteroid_radii = numpy.append(asteroid_radii, asteroid.radius)
        difference = asteroids_pos - bot.center
        distance = numpy.sqrt(numpy.sum(numpy.square(difference), axis=1))

        self.collisions = distance < self.radius + asteroid_radii

        if len(self.collisions) == 0:
            return inputs

        self.sort_indices = numpy.argsort(distance)
        self.coll_keys = numpy.array(list(asteroids.keys()))

        #Provide the distance, positions, radius, x-velocity, y-velocity
        #Identify top 5 nearest asteroids and if there are less than 5 asteroids, we set the input to -1
        #6 inputs per asteroid (5 asteroids)
        #5 inputs for self: center (x, y), self velocity (up, down), self radius
        for i in range(min(len(self.sort_indices), 5)):
            idx = int(self.sort_indices[i])
            inputs[i] = distance[idx] / bounds[0]
            inputs[i + 5] = asteroid_list[idx].center[0] / bounds[0]
            inputs[i + 10] = asteroid_list[idx].center[1] / bounds[1]
            inputs[i + 15] = asteroid_list[idx].radius / bounds[0]
            inputs[i + 20] = asteroid_list[idx].velocity[0] / bounds[0]
            inputs[i + 25] = asteroid_list[idx].velocity[1] / bounds[1]

        #Make sure to compute inputs before updating other things
        if (self.center[0] < xbounds[0] or
                self.center[0] > xbounds[1] or
                self.center[1] < ybounds[0] or
                self.center[1] > ybounds[1]):
            self.lives -= 5
            self.score -= 10
            bot.center[0] += 1000 * bot.velocity[0]
            bot.center[1] -= 1000 * bot.velocity[1]
            print(f"Hit border, {self.lives} Lives Remaining")

        elif sum(self.collisions) > 0:
            self.lives -= sum(self.collisions)
            self.score -= 2 * sum(self.collisions)
            for i in range(len(self.coll_keys)):
                if self.collisions[i] == 1:
                    asteroids.pop(self.coll_keys[i], None)
            print(f"Hit Asteroid, {self.lives} Lives Remaining")
        else:
            self.score += 0.01

        if self.lives < 1:
            gameover = True

        self.reward = self.score - self.old_score
        return inputs

    def update(self, action):
        global gameover

        self.old_score = self.score

        self.velocity[0] = self.speed * math.cos(self.angle)
        self.velocity[1] = self.speed * math.sin(self.angle)

        if action == 0:
            self.center += self.velocity
        if action == 1:
            self.center -= self.velocity
        if action == 2:
            self.angle += -1e-3
        if action == 3:
            self.angle += 1e-3

        # if keyboard.is_pressed('w'):
        #     self.center[0] += self.speed * math.cos(self.angle)
        #     self.center[1] += self.speed * math.sin(self.angle)
        # if keyboard.is_pressed('a'):
        #     self.angle += -1e-3
        # if keyboard.is_pressed('s'):
        #     self.center[0] -= self.speed * math.cos(self.angle)
        #     self.center[1] -= self.speed * math.sin(self.angle)
        # if keyboard.is_pressed('d'):
        #     self.angle += 1e-3

        self.angle %= 2 * math.pi

        if render:
            self.render()


#speed is measured in pixels per second


bot = Player(20.0, 170)


def step(action):
    global running

    if len(asteroids) < 10:
        spawnSingleAsteroid()

    if render:
        pygame.display.flip()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        screen.fill("black")

    bot.update(action)
    for roid in list(asteroids.values()):
        roid.update()

    observation = bot.computeInputs()

    reward = bot.reward

    terminated = gameover

    return observation, reward, terminated, False, False


def reset():
    global gameover

    gameover = False

    bot.reset()

    keys = list(asteroids.keys())
    for i in range(len(keys)):
        del asteroids[keys[i]]

    inputs = numpy.zeros((39,))

    return inputs, 0

# threading.Thread(target=spawnAsteroids).start()

if debug == True:
    #threading.Thread(target=spawnAsteroids).start()
    while running:
        step(1)

    if render:
        pygame.quit()
