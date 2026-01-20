import numpy
import torch
import itertools

a = torch.tensor([[1.0, 2.0, 3.0],
                  [4.0, 5.0, 6.0]], dtype=torch.float32)

print(torch.nn.functional.softmax(a, 1))



