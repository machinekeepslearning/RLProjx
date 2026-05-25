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

cooled_down = True

asteroids = {}
lasers = {}
xbounds = [0, 800]
ybounds = [0, 800]
bounds = (xbounds[1], ybounds[1])

globalId = 0
action_space = (0, 1, 2, 3, 4)

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


def cooldown():
    global cooled_down

    time.sleep(0.5)
    cooled_down = True


def spawnAsteroids(min_speed, max_speed, min_rad, max_rad, max_roids):
    global globalId

    while running:
        if len(asteroids) < max_roids:
            asteroids.update({globalId: Asteroid(
                speed=random.randint(min_speed, max_speed),
                radius=random.randint(min_rad, max_rad),
                id=globalId)})
            globalId += 1
        time.sleep(1)


def spawnSingleAsteroid():
    global globalId

    asteroids.update({globalId: Asteroid(
        speed=random.randint(200, 250),
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
        if chase_num < 8:
            disp = (bot.center - self.center)
            self.velocity = disp / math.sqrt(numpy.dot(disp, disp)) * self.speed
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

        disp = bot.laser_pos - self.center
        dist = numpy.sqrt(numpy.sum(numpy.square(disp), axis=1))

        collisions = dist < self.radius + bot.laser_rad

        laser_keys = list(lasers.keys())

        for i in range(len(collisions)):
            if collisions[i] == 1:
                bot.score += 5
                lasers.pop(laser_keys[i])
                asteroids.pop(self.id)
                break


class Player:
    def __init__(self, radius, speed, angular_speed):
        self.max_lives = 10
        self.lives = self.max_lives
        self.angle = 0
        self.speed = speed * 5e-4
        self.angular_speed = angular_speed * 5e-4
        self.center = numpy.array([(xbounds[1] - xbounds[0]) / 2, (ybounds[1] - ybounds[0]) / 2])
        self.radius = radius
        self.velocity = numpy.zeros((2,))
        self.action_space = (0, 1, 2, 3)
        self.direction_point = self.center + self.radius * numpy.array([math.cos(self.angle), math.sin(self.angle)])
        self.num_inputs = 9 + 30 + 25

        self.collisions = []
        self.sort_indices = []
        self.roid_coll_keys = []

        self.reward = 0
        self.old_score = 0
        self.score = 0

        self.laser_pos = numpy.empty((0, 2))
        self.laser_id = 0
        self.laser_rad = 2

    def fire(self):
        global lasers, cooled_down
        if cooled_down:
            cooled_down = False
            lasers.update({self.laser_id: Laser(self.center[0], self.center[1],
                                                self.angle, 1400,
                                                self.laser_rad, self.laser_id)})
            self.laser_id += 1
            threading.Thread(target=cooldown).start()

    def render(self):
        self.direction_point = self.center + self.radius * numpy.array([math.cos(self.angle), math.sin(self.angle)])
        pygame.draw.circle(screen, "blue", self.center.tolist(), self.radius)
        pygame.draw.line(screen, "yellow", self.center.tolist(), self.direction_point)

    def reset(self):
        self.center = numpy.array([bounds[0] / 2, bounds[1] / 2])
        self.reward = 0
        self.lives = self.max_lives
        self.score = 0
        self.angle = 0

    def computeInputs(self):
        global gameover

        inputs = numpy.zeros((9 + 30 + 25,))

        inputs[self.num_inputs - 9] = self.center[0] / bounds[0]
        inputs[self.num_inputs - 8] = self.center[1] / bounds[1]
        inputs[self.num_inputs - 7] = self.radius / bounds[0]
        inputs[self.num_inputs - 6] = self.velocity[0] / bounds[0]
        inputs[self.num_inputs - 5] = self.velocity[1] / bounds[1]
        inputs[self.num_inputs - 4] = (xbounds[0] - self.center[0]) / bounds[0]
        inputs[self.num_inputs - 3] = (xbounds[1] - self.center[0]) / bounds[0]
        inputs[self.num_inputs - 2] = (ybounds[0] - self.center[1]) / bounds[1]
        inputs[self.num_inputs - 1] = (ybounds[1] - self.center[1]) / bounds[1]

        # Provide the distance, positions, radius, x-velocity, y-velocity
        # Identify top 5 nearest asteroids and if there are less than 5 asteroids, we set the input to -1
        # 6 inputs per asteroid (5 asteroids)
        # 5 inputs for self: center (x, y), self velocity (up, down), self radius
        asteroid_list = list(asteroids.values())
        laser_list = list(lasers.values())

        asteroids_pos = numpy.empty((0, 2))
        asteroid_radii = numpy.empty((0,))
        for asteroid in asteroid_list:
            asteroids_pos = numpy.concatenate((asteroids_pos, [asteroid.center]))
            asteroid_radii = numpy.append(asteroid_radii, asteroid.radius)
        difference = asteroids_pos - bot.center
        distance = numpy.sqrt(numpy.sum(numpy.square(difference), axis=1))

        self.collisions = distance < self.radius + asteroid_radii

        if len(self.collisions) == 0:
            return inputs

        self.sort_indices = numpy.argsort(distance)
        self.roid_coll_keys = numpy.array(list(asteroids.keys()))

        # Provide Velocity, Position, radius
        # 5 inputs per bullet for 5 closest bullets

        self.laser_pos = numpy.empty((0, 2))
        for laser in laser_list:
            self.laser_pos = numpy.concatenate((self.laser_pos, [laser.center]))
        laser_difference = self.laser_pos - bot.center
        laser_distance = numpy.sqrt(numpy.sum(numpy.square(laser_difference), axis=1))
        laser_sort_indices = numpy.argsort(laser_distance)

        num_roid_checks = min(len(asteroid_list), 5)
        num_laser_checks = min(len(laser_list), 5)

        for i in range(num_roid_checks + num_laser_checks):
            if i < num_roid_checks:
                idx = int(self.sort_indices[i])
                inputs[i] = distance[idx] / bounds[0]
                inputs[i + 5] = asteroid_list[idx].center[0] / bounds[0]
                inputs[i + 10] = asteroid_list[idx].center[1] / bounds[1]
                inputs[i + 15] = asteroid_list[idx].radius / bounds[0]
                inputs[i + 20] = asteroid_list[idx].velocity[0] / bounds[0]
                inputs[i + 25] = asteroid_list[idx].velocity[1] / bounds[1]
            elif num_laser_checks > 0:
                j = i - num_roid_checks
                idx = int(laser_sort_indices[j])
                inputs[j + 30] = laser_list[idx].center[0] / bounds[0]
                inputs[j + 35] = laser_list[idx].center[1] / bounds[1]
                inputs[j + 40] = laser_list[idx].radius / bounds[0]
                inputs[j + 45] = laser_list[idx].velocity[0] / bounds[0]
                inputs[j + 50] = laser_list[idx].velocity[1] / bounds[1]

        #Make sure to compute inputs before updating other things
        if (self.center[0] < xbounds[0] or
                self.center[0] > xbounds[1] or
                self.center[1] < ybounds[0] or
                self.center[1] > ybounds[1]):
            self.lives -= 5
            self.score -= 100
            self.center[0] += 1000 * bot.velocity[0]
            self.center[1] -= 1000 * bot.velocity[1]
            print(f"Hit border, {self.lives} Lives Remaining")
        elif sum(self.collisions) > 0:
            self.lives -= sum(self.collisions)
            self.score -= 20 * sum(self.collisions)
            for i in range(len(self.roid_coll_keys)):
                if self.collisions[i] == 1:
                    asteroids.pop(self.roid_coll_keys[i], None)
            print(f"Hit Asteroid, {self.lives} Lives Remaining")
        else:
            self.score += 0.01

        self.reward = self.score - self.old_score

        if self.lives < 1:
            gameover = True

        return inputs

    def update(self, action):
        global gameover

        self.old_score = self.score

        self.velocity[0] = self.speed * math.cos(self.angle)
        self.velocity[1] = self.speed * math.sin(self.angle)

        if debug:
            if keyboard.is_pressed('w'):
                self.center[0] += self.speed * math.cos(self.angle)
                self.center[1] += self.speed * math.sin(self.angle)
            if keyboard.is_pressed('a'):
                self.angle += -self.angular_speed
            if keyboard.is_pressed('s'):
                self.center[0] -= self.speed * math.cos(self.angle)
                self.center[1] -= self.speed * math.sin(self.angle)
            if keyboard.is_pressed('d'):
                self.angle += self.angular_speed
            if keyboard.is_pressed('space'):
                self.fire()
        else:
            if action == 0:
                self.center += self.velocity
                # print("Moving forward")
            if action == 1:
                self.center -= self.velocity
                # print("Moving backward")
            if action == 2:
                self.angle += self.angular_speed
                # print("Rotating clockwise")
            if action == 3:
                self.angle -= self.angular_speed
                # print("Rotating counter clockwise")
            if action == 4:
                self.fire()

        self.angle %= 2 * math.pi

        if render:
            self.render()


class Laser:
    def __init__(self, x, y, angle, speed, radius, id):
        self.id = id
        self.center = numpy.array([x, y])
        self.speed = speed * 5e-4
        self.angle = angle
        self.velocity = self.speed * numpy.array([math.cos(self.angle), math.sin(self.angle)])
        self.radius = radius
        self.color = "yellow"

    def render(self):
        pygame.draw.circle(screen, self.color, self.center.tolist(), self.radius)

    def update(self):
        self.center += self.velocity
        if (self.center[0] < xbounds[0] or
                self.center[0] > xbounds[1] or
                self.center[1] < ybounds[0] or
                self.center[1] > ybounds[1]):
            lasers.pop(self.id)

        if render:
            self.render()


#speed is measured in pixels per second


bot = Player(20.0, 900, 20)

threading.Thread(target=spawnAsteroids, args=(700 - 200, 800 - 200, 10, 30, 20)).start()


def step(action):
    global running

    if render:
        pygame.display.flip()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        screen.fill("black")

    bot.update(action)
    for roid in list(asteroids.values()):
        roid.update()
    for laser in list(lasers.values()):
        laser.update()

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
    laser_keys = list(lasers.keys())
    for i in range(len(laser_keys)):
        del laser_keys[keys[i]]

    inputs = numpy.zeros((bot.num_inputs,))

    return inputs, 0


# threading.Thread(target=spawnAsteroids).start()

if debug:
    #threading.Thread(target=spawnAsteroids).start()
    while running:
        step(1)

    if render:
        pygame.quit()
