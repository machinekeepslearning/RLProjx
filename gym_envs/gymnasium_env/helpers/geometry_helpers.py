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

    def get_center(self):
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


def is_out_of_bounds(pos_arr, bounds, expansion=0):
    bounds = numpy.array(bounds)
    check_arr = numpy.append(pos_arr < (0 - expansion, 0 - expansion), pos_arr > (bounds + expansion))
    return numpy.any(check_arr)


def rotate_ccw(vector, deg_angle):
    '''
    vector must be a 2D vector

    deg angle is an angle in degrees not radians
    '''
    if deg_angle > 0:
        angle = deg_angle * math.pi / 180
        rotation_matrix = numpy.array([[math.cos(angle), -math.sin(angle)],
                                       [math.sin(angle), math.cos(angle)]])
    else:
        angle = -deg_angle * math.pi/180
        rotation_matrix = numpy.array([[math.cos(angle), math.sin(angle)],
                                       [-math.sin(angle), math.cos(angle)]])
    return numpy.matmul(rotation_matrix, vector)