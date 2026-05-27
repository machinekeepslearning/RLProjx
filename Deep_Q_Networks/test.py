import numpy
import keyboard
import matplotlib.pyplot as plt

def f1(x):
    return x+5

def f2(x):
    return numpy.square(x)

xaxis = numpy.array([1, 2, 3, 4, 5, 6, 7])

fig, (ax1, ax2) = plt.subplots(2, 1)

ax1.plot(xaxis, f1(xaxis))

ax2.plot(xaxis, f2(xaxis))

plt.tight_layout()
plt.show()