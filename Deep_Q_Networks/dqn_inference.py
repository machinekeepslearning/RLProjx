import random
import time

import torch
import torch.nn as nn
from torch.nn import Sequential

from Environments.anotherasteroidgame import *
import keyboard

infer = True

baseline = False
#31 seconds
env_render()

n_actions = len(bot.action_space)

# state, info = env.reset()
state, _ = reset()
n_observations = len(state)

device = torch.device(
    "xpu" if torch.xpu.is_available() else
    "mps" if torch.backends.mps.is_available() else
    "cpu"
)

def quit():
    global infer

    infer = False


keyboard.add_hotkey('q', quit)

class DQN(nn.Module):

    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        self.sequence = Sequential(
            nn.Linear(n_observations, 128),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Linear(128, 128),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Linear(128, n_actions),
        )

    def forward(self, x):
        return self.sequence.forward(x)



policy_net = DQN(n_observations, n_actions).to(device)
target_net = DQN(n_observations, n_actions).to(device)

policy_net.load_state_dict(torch.load("Current_Run/asteroid_policy.pt", weights_only=True))
target_net.load_state_dict(torch.load("Current_Run/asteroid_target.pt", weights_only=True))


def select_action(state):

    with torch.no_grad():
        return policy_net(state).max(1).indices.view(1, 1)


state, _ = reset()
state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
start = time.time()

policy_net.eval()

while infer:

    action = select_action(state)

    observation, reward, terminated, truncated, _ = step(action.item())

    reward = torch.tensor([reward], device=device)
    done = terminated or truncated

    if terminated:
        next_state = None
    else:
        next_state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)

    state = next_state

    if done:
        state, _ = reset()
        state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        print(f"Lasted for {time.time() - start} seconds")
        start = time.time()

