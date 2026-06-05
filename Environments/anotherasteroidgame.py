import random
import math
import threading

import pygame
import numpy
import keyboard
import time
from helpers.geometry_helpers import *

debug = True
render = False
running = True
gameover = False
sensor_enabled = False

screen: pygame.Surface = None

collisions = []

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
# Laser, Asteroid
off_cooldown = {"Laser": True, "Asteroid": True}

asteroids = {}

lasers = {}

globalId = 0
action_space = (0, 1, 2, 3, 4)

reward = 0
reward_scale = 1

def toggle_sensors():
    global sensor_enabled

    sensor_enabled = not sensor_enabled


keyboard.add_hotkey("p", toggle_sensors)


def cooldown(cooldown_time, key):
    global off_cooldown

    time.sleep(cooldown_time)
    off_cooldown[key] = True


def spawnAsteroid(min_speed, max_speed, min_rad, max_rad, max_roids):
    global globalId, off_cooldown

    if len(asteroids) < max_roids and off_cooldown["Asteroid"]:
        off_cooldown["Asteroid"] = False
        asteroids.update({globalId: Asteroid(
            speed=random.randint(min_speed, max_speed),
            radius=random.randint(min_rad, max_rad),
            id=globalId)})
        globalId += 1

        threading.Thread(target=cooldown, args=(1, "Asteroid")).start()


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
            lasers.pop(self.id, None)

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
            asteroids.pop(self.id, None)

        if render:
            self.render(screen)


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
        self.num_inputs = 5 + 35 + 10

        self.asteroids_pos = numpy.empty((0, 2))
        self.collisions = []
        self.sort_indices = []
        self.roid_coll_keys = []

        self.laser_id = 0
        self.laser_rad = 2
        self.laser_speed = 1000

        self.sensor_length = 50
        self.num_sensors = 50
        self.sensor_unit_vectors = numpy.zeros((self.num_sensors, 2))
        self.sensor_angles = numpy.zeros((self.num_sensors,))
        self.sensor_color = [None] * self.num_sensors
        increment = (2 * math.pi) / self.num_sensors
        for i in range(self.num_sensors):
            self.sensor_angles[i] = i * increment
            ux = math.cos(self.sensor_angles[i])
            uy = math.sin(self.sensor_angles[i])
            self.sensor_unit_vectors[i] = (ux, uy)

            self.sensor_color[i] = "green"

    def fire(self):
        global lasers, off_cooldown
        if off_cooldown["Laser"]:
            off_cooldown["Laser"] = False
            lasers.update({self.laser_id: Laser(self.center[0], self.center[1],
                                                self.angle, self.laser_speed,
                                                self.laser_rad, self.laser_id, "yellow")})
            self.laser_id += 1

            threading.Thread(target=cooldown, args=(0.4, "Laser")).start()

    def render(self, surface):
        self.direction_point = self.center + self.radius * numpy.array([math.cos(self.angle), math.sin(self.angle)])
        pygame.draw.circle(surface, "blue", self.center.tolist(), self.radius)
        pygame.draw.line(surface, "yellow", self.center.tolist(), self.direction_point)

    def reset(self):
        self.center = numpy.array([bounds[0] / 2, bounds[1] / 2])
        self.lives = self.max_lives
        self.angle = 0

    def update(self, action):
        global gameover, reward

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

        if (self.center[0] < xbounds[0] or
                self.center[0] > xbounds[1] or
                self.center[1] < ybounds[0] or
                self.center[1] > ybounds[1]):
            reward -= 0.4
            bot.lives -= 1
            self.center[0] = bounds[0]/2
            self.center[1] = bounds[1] / 2

        if render:
            self.render(screen)


#speed is measured in pixels per second


bot = Player(20.0, 800, 20, bounds[0] / 2, bounds[1] / 2, "blue")


def globalUpdate():
    global reward

    #initialize object arrays
    asteroid_list = list(asteroids.values())
    asteroid_pos = get_positions(asteroid_list)
    asteroid_radii = get_radii(asteroid_list)

    #Sensing
    rel_asteroid_pos = asteroid_pos - bot.center
    dist = numpy.linalg.norm(rel_asteroid_pos, axis=1)
    dist = numpy.expand_dims(dist, -1)
    #Projection/Normals: axis 0: Asteroids, axis 1: Sensors
    projections = rel_asteroid_pos.dot(bot.sensor_unit_vectors.transpose())
    normals = numpy.sqrt(numpy.square(dist.repeat(bot.num_sensors, axis=1)) - numpy.square(projections))
    along_sensor = projections - numpy.sqrt(
        numpy.square(asteroid_radii.repeat(bot.num_sensors, axis=1)) - numpy.square(normals))
    min_along = numpy.min(along_sensor, where=(along_sensor > 0), axis=0, initial=1000)
    min_along = numpy.reshape(min_along, (50, 1))
    casts = numpy.multiply(min_along, bot.sensor_unit_vectors)

    #Laser-Asteroid Collisions
    laser_keys = list(lasers.keys())
    asteroid_keys = list(asteroids.keys())
    if len(lasers) > 0:
        laser_list = list(lasers.values())
        laser_pos = get_positions(laser_list)
        la_disp = numpy.zeros((len(asteroids), len(lasers), 2))
        for i in range(len(asteroids)):
            la_disp[i] = asteroid_pos[i] - laser_pos
        la_collisions = numpy.linalg.norm(la_disp, axis=2) < (asteroid_radii + bot.laser_rad)
        laser_collisions = numpy.any(la_collisions, axis=0)
        asteroid_collisions = numpy.any(la_collisions, axis=1)

        #Perform deletions
        for i in range(len(lasers)):
            if laser_collisions[i] == 1:
                lasers.pop(laser_keys[i], None)
                reward += 0.6
        for i in range(len(asteroids)):
            if asteroid_collisions[i] == 1:
                asteroids.pop(asteroid_keys[i], None)

    if len(asteroids) > 0:
        ba_disp = asteroid_pos - bot.center
        ba_dist = numpy.linalg.norm(ba_disp, axis=1)
        ba_collisions = (asteroid_radii + bot.radius).flatten() > ba_dist
        for i in range(len(asteroids)):
            if ba_collisions[i] == 1:
                asteroids.pop(asteroid_keys[i], None)
                reward -= 0.4
                bot.lives -= 1


    if render and sensor_enabled:
        for i in range(len(casts)):
            pygame.draw.line(screen, "green", bot.center, casts[i] + bot.center)

    return casts.flatten()/1000


def step(action):
    global running, reward

    reward = 0

    if render:
        pygame.display.flip()
        screen.fill("black")

    spawnAsteroid(100, 100, 10, 30, 20)

    bot.update(action)
    for roid in list(asteroids.values()):
        roid.update()
    for laser in list(lasers.values()):
        laser.update()

    observation = globalUpdate()

    terminated = True if bot.lives < 1 else False

    return observation, reward, terminated, False, False


def reset():
    global gameover, reward

    reward = 0
    gameover = False

    keys = list(asteroids.keys())
    for i in range(len(keys)):
        del asteroids[keys[i]]
    laser_keys = list(lasers.keys())
    for i in range(len(laser_keys)):
        del lasers[laser_keys[i]]

    bot.reset()

    inputs = numpy.zeros((bot.num_sensors,)) + 1000

    return inputs, 0


# threading.Thread(target=spawnAsteroids).start()

if debug:
    #threading.Thread(target=spawnAsteroids).start()
    while running:
        if render:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
        step(1)
        if reward != 0:
            print(reward)

    if render:
        pygame.quit()
