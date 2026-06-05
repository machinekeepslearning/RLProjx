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

    def center(self):
        return self.center.copy()


def get_positions(obj_list):
    '''
    Return a matrix of object positions
    N is the size of the object list
    Matrix returned is of shape (N, 2)
    '''
    pos = numpy.zeros((len(obj_list), 2))
    for i in range(len(obj_list)):
        pos[i] = obj_list[i].center

    return pos


def get_radii(obj_list):
    '''
    Return a matrix of object radii
    N is the size of the object list
    Matrix returned is of shape (N, 1)
    '''
    rads = numpy.zeros((len(obj_list), 1))
    for i in range(len(obj_list)):
        rads[i] = obj_list[i].radius

    return rads
