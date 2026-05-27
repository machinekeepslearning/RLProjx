import math
import pygame

import numpy

class Circle:
    def __init__(self, x, y, radius, color):
        self.center = numpy.array([x, y])
        self.radius = radius
        self.color = color
    def render(self, screen):
        pygame.draw.circle(screen, self.color, (int(self.center[0]), int(self.center[1])), self.radius)

def obj_distance_list(this, other_list):
    other_pos = numpy.empty((0, 2))
    for obj in other_list:
        other_pos = numpy.concatenate((other_pos, [obj.center]))
    disp = other_pos - this.center

    distances = numpy.sqrt(numpy.sum(numpy.square(disp), axis=1))

    return distances