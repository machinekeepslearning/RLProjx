import numpy
import keyboard
import matplotlib.pyplot as plt
import torch.backends.mkldnn

a = numpy.array([[1, 2],
                 [3, 4],
                 [5, 6]])


a1 = numpy.array([1, 2, 3, 4, 5])
a2 = numpy.array([1, 2, 3, 4, 5])

print(torch.backends.mkldnn.is_available())
print(torch.cpu._is_avx512_bf16_supported())