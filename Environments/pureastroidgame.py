import random
import math
import threading

import pygame
import numpy
import keyboard
import time
from helpers.collision_helpers import *

debug = False
render = False
running = True
gameover = False

screen = None

xbounds = [0, 800]
ybounds = [0, 800]
bounds = (xbounds[1], ybounds[1])

def env_render():
    global render, screen
    render = True

    pygame.init()
    screen = pygame.display.set_mode(bounds)
    screen.fill("black")

if debug:
    env_render()

cooled_down = True

asteroids = {}
lasers = {}

globalId = 0
action_space = (0, 1, 2, 3, 4)

reward_scale = 1




def cooldown():
    global cooled_down

    time.sleep(0.4)
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


class Laser(Circle):
    def __init__(self, x, y, angle, speed, radius, id, color):
        super().__init__(x, y, radius, color)
        self.id = id
        self.speed = speed * 5e-4
        self.angle = angle
        self.velocity = self.speed * numpy.array([math.cos(self.angle), math.sin(self.angle)])

    def render(self, surface):
        pygame.draw.circle(surface, self.color, self.center.tolist(), self.radius)

    def update(self):
        self.center += self.velocity

        if (self.center[0] < xbounds[0] or
                self.center[0] > xbounds[1] or
                self.center[1] < ybounds[0] or
                self.center[1] > ybounds[1]):
            lasers.pop(self.id)

        if render:
            self.render(screen)


class Asteroid(Circle):
    def __init__(self, speed, radius, id):
        self.start_angle = random.uniform(0, 2 * math.pi)
        self.pos_radius = 700
        x = (xbounds[1] - xbounds[0]) / 2.0 + self.pos_radius * math.cos(self.start_angle)
        y = (ybounds[1] - ybounds[0]) / 2.0 + self.pos_radius * math.sin(self.start_angle)
        super().__init__(x, y, radius, "red")

        self.id = id

        #set movement
        self.speed = speed * 5e-4
        self.dir = random.uniform(0, 2 * math.pi)

        chase_num = random.randint(1, 10)
        if chase_num < 7:
            disp = (bot.center - self.center)
            self.velocity = disp / math.sqrt(numpy.dot(disp, disp)) * self.speed
        else:
            self.velocity = numpy.array([self.speed * math.cos(self.dir), self.speed * math.sin(self.dir)])

    def render(self, surface):
        pygame.draw.circle(surface, self.color, (int(self.center[0]), int(self.center[1])), self.radius)

    def update(self):
        self.center[0] += self.velocity[0]
        self.center[1] += self.velocity[1]

        if (self.center[0] < (xbounds[0] - 300) or
                self.center[0] > (xbounds[1] + 300) or
                self.center[1] < (ybounds[0] - 300) or
                self.center[1] > (ybounds[1] + 300)):
            asteroids.pop(self.id)

        if render:
            self.render(screen)

        disp = bot.laser_pos - self.center
        dist = numpy.sqrt(numpy.sum(numpy.square(disp), axis=1))

        collisions = dist < self.radius + bot.laser_rad

        laser_keys = list(lasers.keys())

        for i in range(len(collisions)):
            if collisions[i] == 1:
                bot.reward += 1 * reward_scale
                print(f"Asteroid Destroyed! Current Reward: {bot.reward}")
                lasers.pop(laser_keys[i])
                asteroids.pop(self.id)
                break


class Player(Circle):
    def __init__(self, radius, speed, angular_speed, x, y, color):
        super().__init__(x, y, radius, color)
        self.max_lives = 10
        self.lives = self.max_lives
        self.angle = 0
        self.speed = speed * 5e-4
        self.angular_speed = angular_speed * 5e-4
        self.velocity = numpy.zeros((2,))
        self.action_space = (0, 1, 2, 3, 4)
        self.direction_point = self.center + self.radius * numpy.array([math.cos(self.angle), math.sin(self.angle)])
        #Self data + asteroid data + laser data
        self.num_inputs = 8 + 30 + 10

        self.asteroids_pos = numpy.empty((0, 2))
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
                                                self.angle, 5000,
                                                self.laser_rad, self.laser_id, "yellow")})
            self.laser_id += 1
            threading.Thread(target=cooldown).start()

    def render(self, surface):
        self.direction_point = self.center + self.radius * numpy.array([math.cos(self.angle), math.sin(self.angle)])
        pygame.draw.circle(surface, "blue", self.center.tolist(), self.radius)
        pygame.draw.line(surface, "yellow", self.center.tolist(), self.direction_point)
        for i in range(len(self.asteroids_pos)):
            pygame.draw.line(surface, "white", self.center.tolist(), self.asteroids_pos[i])
        for i in range(len(self.laser_pos)):
            pygame.draw.line(surface, "red", self.center.tolist(), self.laser_pos[i])

    def reset(self):
        self.center = numpy.array([bounds[0] / 2, bounds[1] / 2])
        self.reward = 0
        self.lives = self.max_lives
        self.score = 0
        self.angle = 0

    def computeInputs(self):
        global gameover

        inputs = numpy.zeros((self.num_inputs,)) - 1

        inputs[self.num_inputs - 8] = self.angle / (2 * math.pi)
        inputs[self.num_inputs - 7] = self.center[0] / bounds[0]
        inputs[self.num_inputs - 6] = self.center[1] / bounds[1]
        inputs[self.num_inputs - 5] = self.radius / self.radius
        inputs[self.num_inputs - 4] = self.velocity[0] / self.speed
        inputs[self.num_inputs - 3] = self.velocity[1] / self.speed
        inputs[self.num_inputs - 2] = (self.center[0] - bounds[0]) / bounds[0]
        inputs[self.num_inputs - 1] = (self.center[1] - bounds[1]) / bounds[1]

        # Provide the distance, positions, radius, x-velocity, y-velocity
        # Identify top 5 nearest asteroids and if there are less than 5 asteroids, we set the input to -1
        # 6 inputs per asteroid (5 asteroids)
        # 5 inputs for self: center (x, y), self velocity (up, down), self radius
        asteroid_list = list(asteroids.values())
        laser_list = list(lasers.values())

        self.asteroids_pos = numpy.empty((0, 2))
        asteroid_radii = numpy.empty((0,))
        for asteroid in asteroid_list:
            self.asteroids_pos = numpy.concatenate((self.asteroids_pos, [asteroid.center]))
            asteroid_radii = numpy.append(asteroid_radii, asteroid.radius)
        difference = self.asteroids_pos - bot.center
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
        num_laser_checks = min(len(laser_list), 2)

        for i in range(num_roid_checks):
            idx = int(self.sort_indices[i])
            inputs[i] = distance[idx] / bounds[0]
            inputs[i + 5] = asteroid_list[idx].center[0] / bounds[0]
            inputs[i + 10] = asteroid_list[idx].center[1] / bounds[1]
            inputs[i + 15] = asteroid_list[idx].radius / self.radius
            inputs[i + 20] = asteroid_list[idx].velocity[0] / self.speed
            inputs[i + 25] = asteroid_list[idx].velocity[1] / self.speed
        for i in range(num_laser_checks):
            idx = int(laser_sort_indices[i])
            inputs[i + 30] = laser_list[idx].center[0] / bounds[0]
            inputs[i + 32] = laser_list[idx].center[1] / bounds[1]
            inputs[i + 34] = laser_list[idx].radius / self.radius
            inputs[i + 36] = laser_list[idx].velocity[0] / self.speed
            inputs[i + 38] = laser_list[idx].velocity[1] / self.speed

        #Make sure to compute inputs before updating other things
        if (self.center[0] < 0 or
                self.center[0] > bounds[0] or
                self.center[1] < 0 or
                self.center[1] > bounds[1]):
            self.lives -= 2
            self.reward -= 4 * reward_scale
            self.center[0] = bounds[0] / 2
            self.center[1] = bounds[1] / 2
            print(f"Border Hit, current reward: {self.reward}")
        elif sum(self.collisions) > 0:
            self.lives -= sum(self.collisions)
            for i in range(len(self.roid_coll_keys)):
                if self.collisions[i] == 1:
                    asteroids.pop(self.roid_coll_keys[i], None)
                    self.reward -= 2 * reward_scale
            print(f"Hit Asteroids, current reward: {self.reward}")

        if self.lives < 1:
            gameover = True

        return inputs

    def update(self, action):
        global gameover

        self.reward = 0

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
            self.render(screen)


#speed is measured in pixels per second


bot = Player(20.0, 900, 20, bounds[0] / 2, bounds[1] / 2, "blue")

threading.Thread(target=spawnAsteroids, args=(700, 800, 10, 30, 20)).start()


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
        del lasers[laser_keys[i]]

    inputs = numpy.zeros((bot.num_inputs,)) - 1

    return inputs, 0


# threading.Thread(target=spawnAsteroids).start()

if debug:
    #threading.Thread(target=spawnAsteroids).start()
    while running:
        step(1)
        print(bot.score)

    if render:
        pygame.quit()
