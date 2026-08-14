import math
import random

import numpy

from gymnasium_env.helpers.geometry_helpers import *

class Laser(Circle):
    def __init__(self, x, y, angle, speed, radius, id, color, bounds):
        base_scale = 5e-2 * bounds[0]/800.0
        super().__init__(x, y, radius, color)
        self.id = id
        self.speed = speed * base_scale * bounds[0]/800.0
        self.angle = angle
        self.velocity = self.speed * numpy.array([math.cos(self.angle), math.sin(self.angle)])
        self.bounds = bounds

    def render(self, surface):
        pygame.draw.circle(surface, self.color, self.center.tolist(), self.radius)

    def update(self):
        self.center += self.velocity


class Asteroid(Circle):
    def __init__(self, speed, radius, id, bounds):
        base_scale = 5e-2 * bounds[0] / 800.0
        self.start_angle = random.uniform(0, 2 * math.pi)
        self.pos_radius = 700 * bounds[0]/800.0
        self.bounds = numpy.array(bounds)
        x = bounds[0] / 2.0 + self.pos_radius * math.cos(self.start_angle)
        y = bounds[1] / 2.0 + self.pos_radius * math.sin(self.start_angle)

        super().__init__(x, y, radius * bounds[0]/800.0, "red")

        self.id = id

        #set movement
        self.speed = speed * base_scale * bounds[0]/800.0

        self.rand_angle = random.uniform(-30, 30)

        self.dir = rotate_ccw(self.bounds / 2 - self.center, self.rand_angle)

        self.velocity = self.speed * self.dir / numpy.linalg.norm(self.dir)

    def render(self, surface):
        pygame.draw.circle(surface, self.color, (int(self.center[0]), int(self.center[1])), self.radius)

    def update(self):
        self.center += self.velocity


class Agent(Circle):
    def __init__(self, radius, speed, angular_speed, x, y, color, bounds):
        base_scale = 5e-2 * bounds[0] / 800.0
        super().__init__(x, y, radius * bounds[0]/800.0, color)
        self.bounds = bounds
        self.start_pos = numpy.array([x, y])
        self.angle = 0
        self.speed = speed * base_scale * bounds[0]/800.0
        self.angular_speed = angular_speed * base_scale * bounds[0]/800.0
        self.velocity = numpy.zeros((2,))
        #Self data + asteroid data + laser data

        self.laser_id = 0
        self.laser_rad = 2

    def render(self, surface):
        direction_point = self.center + self.radius * numpy.array([math.cos(self.angle), math.sin(self.angle)])
        pygame.draw.circle(surface, "blue", self.center.tolist(), self.radius)
        pygame.draw.line(surface, "yellow", self.center.tolist(), direction_point)

    def reset(self):
        self.center = self.start_pos.copy()
        self.angle = 0

    def update(self):
        self.velocity[0] = self.speed * math.cos(self.angle)
        self.velocity[1] = self.speed * math.sin(self.angle)

        self.angle %= 2 * math.pi
