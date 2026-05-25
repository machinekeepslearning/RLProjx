import numpy
import keyboard

a = numpy.array([1, 2, 3, 4, 5, 6])


for i in range(10):
    global local

    for j in range(5):
        local = j+10

    print(local)