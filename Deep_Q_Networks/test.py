import numpy
import keyboard
import matplotlib.pyplot as plt

def f1(x):
    return x+5

def f2(x):
    return numpy.square(x)

test = numpy.zeros((5,))

test[0:2] = 1

print(test)