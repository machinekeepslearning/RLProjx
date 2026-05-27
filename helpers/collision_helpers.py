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