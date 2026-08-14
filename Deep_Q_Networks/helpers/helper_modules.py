import torch
from torch import nn
from tensordict import TensorDict


class IntToFloat(nn.Module):
    def forward(self, x: TensorDict):
        return x.to(torch.float32)
